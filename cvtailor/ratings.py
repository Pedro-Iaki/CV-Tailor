"""Rate CVs 1-5 on a local web page (apply.py rate).

Candidates are CVs to judge: saved applications, and selections from JSONL files
(for example the local-model benchmark). Each is rendered with the normal template,
next to its job posting. Ratings are blind: the model or config behind a CV
(its label) shows only after it has been rated, so the score isn't biased.

The server binds to 127.0.0.1 only; nothing leaves the machine. Ratings live in the
cv_ratings table of applications.db.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from . import db
from .master import load_master
from .render import make_render_input, render_html
from .schema import Master, Selection
from .validate import validate_selection


@dataclass
class Candidate:
    key: str          # stable id, e.g. "app:3" or "choices-v5|gpt-oss:20b|low|job.txt"
    title: str        # shown while rating: the posting, not the model
    job_text: str
    selection: Selection
    master: Master
    label: str = ""   # revealed after rating: model, settings, run
    group: str = ""   # for the summary table: ratings are averaged per group
    lang: str = "en"


def application_candidates(conn: sqlite3.Connection, master: Master) -> list[Candidate]:
    out = []
    for r in conn.execute("SELECT * FROM applications WHERE status != 'discarded' ORDER BY id"):
        app = db.row_to_dict(r)
        out.append(Candidate(
            key=f"app:{app['id']}", title=f"{app['company']} | {app['role_title']}",
            job_text=app["job_description"], selection=Selection.model_validate(app["selection"]),
            master=master, label=f"application #{app['id']} ({app['route']}, {app.get('model') or '-'})",
            group="applications", lang=app["lang"]))
    return out


def file_candidates(path: Path, default_master: Master) -> list[Candidate]:
    """JSONL, one CV per line: {key, title?, job_text, selection, master_path?, label?,
    group?, lang?}. master_path is resolved relative to the file."""
    masters: dict[str, Master] = {}
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        m = default_master
        if d.get("master_path"):
            mp = str((path.parent / d["master_path"]).resolve())
            m = masters.setdefault(mp, load_master(mp))
        out.append(Candidate(
            key=d["key"], title=d.get("title") or d["job_text"].strip().splitlines()[0][:80],
            job_text=d["job_text"], selection=Selection.model_validate(d["selection"]), master=m,
            label=d.get("label", ""), group=d.get("group", ""), lang=d.get("lang", "en")))
    return out


class RatingStore:
    """Ratings in applications.db. Opens a connection per operation, because the page's
    server answers requests on several threads and SQLite connections are per-thread."""

    def __init__(self, db_path: Path | str):
        self.db_path = db_path

    def all(self) -> dict[str, dict[str, Any]]:
        with closing(db.connect(self.db_path)) as conn:
            return {r["key"]: dict(r) for r in conn.execute("SELECT * FROM cv_ratings")}

    def save(self, c: Candidate, rating: int, note: str = "") -> None:
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError("rating must be 1 to 5")
        with closing(db.connect(self.db_path)) as conn:
            self._upsert(conn, c, rating, note)

    @staticmethod
    def _upsert(conn: sqlite3.Connection, c: Candidate, rating: int, note: str) -> None:
        conn.execute(
            "INSERT INTO cv_ratings (key, rating, note, label, grp, rated_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET rating=excluded.rating, note=excluded.note, "
            "label=excluded.label, grp=excluded.grp, rated_at=excluded.rated_at",
            (c.key, rating, note.strip() or None, c.label, c.group, db.now_iso()))


class RatingApp:
    """The page's data, independent of HTTP (so it can be tested directly)."""

    def __init__(self, candidates: list[Candidate], store: RatingStore):
        self.by_key = {c.key: c for c in candidates}
        self.order = [c.key for c in candidates]
        self.store = store

    def items(self) -> list[dict[str, Any]]:
        rated = self.store.all()
        out = []
        for n, key in enumerate(self.order, 1):
            c, r = self.by_key[key], rated.get(key)
            out.append({"n": n, "key": key, "title": c.title, "lang": c.lang,
                        "rating": r["rating"] if r else None, "note": (r or {}).get("note") or "",
                        "label": c.label if r else ""})  # blind until rated
        return out

    def cv_html(self, key: str, lang: str) -> str:
        c = self.by_key[key]
        errors = validate_selection(c.selection, c.master)
        if errors:
            return "<p>This selection is not valid for its master:</p><ul>" + "".join(
                f"<li>{e}</li>" for e in errors) + "</ul>"
        return render_html(make_render_input(c.master, c.selection, lang if lang in ("en", "pt") else c.lang))

    def job_text(self, key: str) -> str:
        return self.by_key[key].job_text

    def rate(self, key: str, rating: int, note: str = "") -> dict[str, Any]:
        c = self.by_key[key]
        self.store.save(c, int(rating), note)
        return {"key": key, "rating": int(rating), "label": c.label}

    def summary(self) -> list[dict[str, Any]]:
        """Mean rating per group, over the candidates on this page."""
        rated = self.store.all()
        groups: dict[str, list[int]] = {}
        for key in self.order:
            c = self.by_key[key]
            groups.setdefault(c.group or "(none)", [])
            if key in rated:
                groups[c.group or "(none)"].append(rated[key]["rating"])
        rows = [{"group": g, "rated": len(v), "total": sum(1 for k in self.order if (self.by_key[k].group or "(none)") == g),
                 "mean": round(sum(v) / len(v), 2) if v else None, "ratings": v} for g, v in groups.items()]
        return sorted(rows, key=lambda r: (r["mean"] is None, -(r["mean"] or 0)))


def make_handler(app: RatingApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _send(self, body: str | bytes, ctype: str, status: int = 200) -> None:
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    self._send(PAGE, "text/html; charset=utf-8")
                elif u.path == "/api/items":
                    self._send(json.dumps(app.items()), "application/json")
                elif u.path == "/api/summary":
                    self._send(json.dumps(app.summary()), "application/json")
                elif u.path.startswith("/cv/"):
                    key = unquote(u.path[4:])
                    self._send(app.cv_html(key, q.get("lang", [""])[0]), "text/html; charset=utf-8")
                elif u.path.startswith("/job/"):
                    self._send(app.job_text(unquote(u.path[5:])), "text/plain; charset=utf-8")
                else:
                    self._send("not found", "text/plain", 404)
            except KeyError:
                self._send("unknown CV", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/rate":
                self._send("not found", "text/plain", 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                self._send(json.dumps(app.rate(body["key"], body["rating"], body.get("note", ""))),
                           "application/json")
            except (KeyError, ValueError, TypeError) as e:
                self._send(json.dumps({"error": str(e)}), "application/json", 400)
    return Handler


def serve(app: RatingApp, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rate CVs</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#c4451c;--ok:#2f7d4f}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--muted:#9a998f;--line:#33332f;--accent:#e8774f;--ok:#5fbf86}}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink);display:grid;grid-template-columns:260px 1fr;height:100vh}
aside{border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
aside h1{font-size:15px;margin:14px 14px 4px}aside .sub{color:var(--muted);margin:0 14px 10px;font-size:12px}
.tabs{display:flex;gap:6px;margin:0 14px 10px}.tabs button{flex:1}
.item{display:flex;justify-content:space-between;gap:8px;padding:8px 14px;border-top:1px solid var(--line);cursor:pointer}
.item:hover{background:var(--bg)}.item.on{background:var(--bg);box-shadow:inset 3px 0 var(--accent)}
.item .t{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.item .r{font-weight:600;color:var(--ok);min-width:18px;text-align:right}
main{display:grid;grid-template-rows:auto 1fr;min-width:0}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
.bar .title{font-weight:600;margin-right:auto}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:6px 10px;cursor:pointer}
button.star{width:38px;font-weight:600}button.star.sel{background:var(--accent);border-color:var(--accent);color:#fff}
input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);min-width:220px}
.label{color:var(--muted);font-size:12px;width:100%}
.panes{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);min-height:0}
iframe{width:100%;height:100%;border:0;background:#fff}
pre{margin:0;padding:14px;white-space:pre-wrap;overflow:auto;border-left:1px solid var(--line);font:13px/1.45 system-ui,sans-serif}
table{border-collapse:collapse;margin:14px;width:calc(100% - 28px)}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
.hint{color:var(--muted);font-size:12px}
@media (max-width:800px){body{grid-template-columns:1fr}aside{max-height:30vh}.panes{grid-template-columns:1fr}pre{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<aside><h1>Rate CVs</h1><p class="sub" id="progress"></p>
<div class="tabs"><button id="nextUnrated">Next unrated</button><button id="showSummary">Summary</button></div>
<div id="list"></div></aside>
<main><div class="bar"><span class="title" id="title">Loading...</span>
<button id="lang">EN / PT</button><span id="stars"></span>
<input type="text" id="note" placeholder="Note (optional)"><button id="save">Save and next</button>
<div class="label" id="label"></div>
<div class="hint">Keys: 1-5 rate and go to the next, L switches language, N next unrated. 5 = would send as is, 1 = unusable.</div></div>
<div class="panes" id="panes"><iframe id="cv" title="CV"></iframe><pre id="job"></pre></div></main>
<script>
let items=[], cur=null, lang=null, picked=null;
const $=id=>document.getElementById(id);
async function load(){items=await (await fetch('/api/items')).json();drawList();if(!cur){const u=items.find(i=>!i.rating)||items[0];if(u)show(u.key)}}
function drawList(){const done=items.filter(i=>i.rating).length;$('progress').textContent=`${done} of ${items.length} rated`;
 $('list').innerHTML='';items.forEach(i=>{const d=document.createElement('div');d.className='item'+(i.key===cur?' on':'');
 d.innerHTML=`<span class="t">#${i.n} ${esc(i.title)}</span><span class="r">${i.rating||''}</span>`;d.onclick=()=>show(i.key);$('list').appendChild(d)})}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function show(key){cur=key;const i=items.find(x=>x.key===key);lang=lang||i.lang;picked=i.rating;$('panes').style.display='';
 const tbl=document.querySelector('main table');if(tbl)tbl.remove();
 $('title').textContent=`#${i.n}  ${i.title}`;$('note').value=i.note||'';$('label').textContent=i.label?('Made by: '+i.label):'Made by: hidden until rated';
 drawStars();drawList();$('cv').src=`/cv/${encodeURIComponent(key)}?lang=${lang}`;$('job').textContent=await (await fetch('/job/'+encodeURIComponent(key))).text()}
function drawStars(){$('stars').innerHTML='';for(let n=1;n<=5;n++){const b=document.createElement('button');b.className='star'+(picked===n?' sel':'');b.textContent=n;b.onclick=()=>rate(n);$('stars').appendChild(b)}}
async function rate(n){picked=n;drawStars();const r=await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:cur,rating:n,note:$('note').value})});
 if(!r.ok){alert('Not saved: '+(await r.text()));return}await load();next()}
function next(){const idx=items.findIndex(i=>i.key===cur);const u=items.slice(idx+1).concat(items.slice(0,idx+1)).find(i=>!i.rating);if(u)show(u.key);else{show(cur);summary()}}
async function summary(){const rows=await (await fetch('/api/summary')).json();$('panes').style.display='none';$('title').textContent='Summary: mean rating per setup';
 $('label').textContent='';const t=document.createElement('table');t.innerHTML='<tr><th>Setup</th><th>Rated</th><th>Mean</th><th>Ratings</th></tr>'+rows.map(r=>`<tr><td>${esc(r.group)}</td><td>${r.rated} / ${r.total}</td><td>${r.mean??'-'}</td><td>${r.ratings.join(', ')}</td></tr>`).join('');
 const old=document.querySelector('main table');if(old)old.remove();document.querySelector('main').appendChild(t)}
$('save').onclick=()=>picked&&rate(picked);$('nextUnrated').onclick=next;$('showSummary').onclick=summary;
$('lang').onclick=()=>{lang=lang==='en'?'pt':'en';show(cur)};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'){if(e.key==='Enter'&&picked)rate(picked);return}
 if(e.key>='1'&&e.key<='5')rate(+e.key);else if(e.key==='l'||e.key==='L')$('lang').click();else if(e.key==='n'||e.key==='N')next()});
load();
</script></body></html>
"""
