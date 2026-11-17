"""Opt-in text generation with item-by-item human review.

Flow for one application:
  create_proposals   LLM drafts up to N items (rephrase / merge / new), each checked in code
  review             the user answers each one: "y", "n", "y, <comment>" or "n, <comment>"
                       y            -> rephrase/merge: stored as a FACT (reusable, renderable)
                                       new:           stored as UNSAFE (never rendered until
                                                      promoted by hand with `apply.py unsafe promote`)
                       y, comment   -> revised by the LLM and shown again (nothing stored yet)
                       n[, comment] -> rejected; the comment is kept as feedback for later drafts
  apply_approved     newly approved facts replace their sources in this application's
                     selection, and the PDF is re-rendered (with page fit)

Nothing generated reaches a CV without an explicit "y" on the exact text, and new claims
need a second, manual step on top of that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import db
from .facts import add_fact, add_unsafe, next_id
from .llm import LLMError
from .master import TAG_RE, item_index, item_text
from .pipeline import (
    IO,
    Context,
    PipelineError,
    _fit,
    describe_selection,
    display_path,
    get_app,
    stored_selection,
    write_atomic,
)
from .schema import Master, Selection
from .validate import validate_selection

MAX_PROPOSALS = 5
MAX_REVISIONS = 3
MAX_TEXT = 400
REPLY_RE = re.compile(r"^\s*(y|yes|s|sim|n|no|nao|não)(?![a-z])\s*[,;:.\-]?\s*(.*)$", re.I | re.S)
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


class GenerationError(PipelineError):
    pass


def ensure_enabled(ctx: Context) -> None:
    if not ctx.cfg.allow_generation:
        raise GenerationError(
            "text generation is disabled. Set CV_ALLOW_GENERATION=true in .env to enable "
            "LLM-drafted wording (every draft still needs your approval)."
        )


def parse_reply(text: str) -> tuple[bool, Optional[str]]:
    """'y' / 'n' / 'y, comment' / 'n, comment' (also yes/no, s/sim, nao)."""
    m = REPLY_RE.match(text or "")
    if not m:
        raise ValueError(f"could not read reply {text!r}: answer y, n, 'y, <comment>' or 'n, <comment>'")
    approve = m.group(1).lower() in ("y", "yes", "s", "sim")
    comment = m.group(2).strip() or None
    return approve, comment


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUM_RE.findall(text or "")}


# ---------------------------------------------------------------- checking drafts

def check_proposal(raw: dict[str, Any], master: Master) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Validate an LLM draft. Returns (normalized proposal or None, notes)."""
    idx = item_index(master)
    owners = {e.id for e in [*master.jobs, *master.projects]}  # what a bullet can belong to
    kind = raw.get("kind")
    origin = raw.get("origin")
    en = " ".join(str(raw.get("en") or "").split())
    pt = " ".join(str(raw.get("pt") or "").split())
    if kind not in ("bullet", "summary") or origin not in ("rephrase", "merge", "new"):
        return None, ["invalid kind or origin"]
    if not en or not pt:
        return None, ["draft is missing its English or Portuguese text"]
    if len(en) > MAX_TEXT or len(pt) > MAX_TEXT:
        return None, ["draft is too long"]
    based = list(dict.fromkeys(raw.get("based_on") or []))
    for b in based:
        if b not in idx or idx[b].kind not in ("bullet", "summary"):
            return None, [f"unknown source id '{b}'"]
    # Shape checks come first: the checks below index based[0].
    if kind == "summary" and origin == "merge":
        return None, ["summaries can be rephrased, not merged"]
    if origin == "rephrase" and len(based) != 1:
        return None, ["a rephrase needs exactly one source"]
    if origin == "merge" and not 2 <= len(based) <= 3:
        return None, ["a merge needs 2-3 sources"]

    if kind == "summary":
        job_id = None
        if origin == "rephrase" and idx[based[0]].kind != "summary":
            return None, ["a summary rephrase must be based on a summary"]
    else:
        job_id = raw.get("job_id") or None
        if job_id not in owners:
            return None, [f"unknown job or project '{job_id}'"]
        if origin in ("rephrase", "merge") and any(
            idx[b].kind != "bullet" or idx[b].parent != job_id for b in based
        ):
            return None, ["rephrase/merge sources must be bullets of the same job or project"]

    notes: list[str] = []
    # Deterministic grounding guard: a reworded item may not contain numbers its sources lack.
    if origin in ("rephrase", "merge"):
        src = set()
        for b in based:
            src |= numbers(idx[b].obj.en) | numbers(idx[b].obj.pt)
        extra = (numbers(en) | numbers(pt)) - src
        if extra:
            origin = "new"
            notes.append(f"contains {', '.join(sorted(extra))}, not found in its sources: "
                         "treated as an unverified new claim")

    want = "summary" if kind == "summary" else "bullet"
    replaces = []
    if origin != "new":
        for r in dict.fromkeys(raw.get("replaces") or []):
            it = idx.get(r)
            if it and it.kind == want and (kind == "summary" or it.parent == job_id):
                replaces.append(r)

    tags = [t for t in dict.fromkeys(raw.get("tags") or []) if isinstance(t, str) and TAG_RE.match(t)]
    if not tags:
        for b in based:
            tags += [t for t in idx[b].obj.tags if t not in tags]
    if not tags:
        return None, ["draft has no valid tags"]

    return {
        "kind": kind,
        "job_id": job_id,
        "origin": origin,
        "destination": "unsafe" if origin == "new" else "fact",
        "based_on": based,
        "replaces": replaces,
        "tags": tags,
        "en": en,
        "pt": pt,
        "rationale": str(raw.get("rationale") or "").strip() or None,
        "notes": "; ".join(notes) or None,
    }, notes


# ---------------------------------------------------------------- storage helpers

_row = db.item_row_to_dict


def get_proposal(conn, pid: int) -> Optional[dict[str, Any]]:
    r = conn.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
    return _row(r) if r else None


def pending_proposals(conn, app_id: int) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM proposals WHERE application_id = ? AND status = 'pending' ORDER BY id",
                        (app_id,)).fetchall()
    return [_row(r) for r in rows]


def _insert_proposal(conn, app_id: int, p: dict[str, Any], *, model: str,
                     parent_id: Optional[int] = None, depth: int = 0) -> int:
    data = {**p, "application_id": app_id, "parent_id": parent_id, "depth": depth, "model": model,
            "created_at": db.now_iso()}
    data = {k: (json.dumps(v, ensure_ascii=False) if k in db.ITEM_JSON_COLUMNS else v) for k, v in data.items()}
    cols = ", ".join(data)
    cur = conn.execute(f"INSERT INTO proposals ({cols}) VALUES ({', '.join('?' for _ in data)})",
                       list(data.values()))
    return cur.lastrowid


def recent_feedback(conn, limit: int = 15) -> list[str]:
    rows = conn.execute(
        "SELECT status, origin, en, comment FROM proposals WHERE comment IS NOT NULL ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [f"- {r['status']} {r['origin']} draft \"{r['en'][:140]}\": {r['comment']}" for r in rows]


def _sources_text(master: Master, ids: list[str]) -> str:
    idx = item_index(master)
    lines = []
    for i in dict.fromkeys(ids):
        if i in idx:
            o = idx[i].obj
            lines.append(f"{i}\n  en: {o.en}\n  pt: {o.pt}")
    return "\n".join(lines) or "(none)"


# ---------------------------------------------------------------- proposals

def create_proposals(ctx: Context, app_id: int, io: IO = IO(), max_items: int = MAX_PROPOSALS) -> list[dict]:
    ensure_enabled(ctx)
    app = get_app(ctx, app_id)
    sel = stored_selection(ctx, app)
    master = ctx.master
    conn = ctx.db()

    unmatched = app.get("unmatched_requirements")
    context = "\n".join([
        "CURRENT SELECTION (id and English text):",
        *describe_selection(sel, master, "en"),
        "",
        "UNMATCHED REQUIREMENTS: " + (", ".join(unmatched) if unmatched else "(none recorded)"),
        "",
        "REVIEWER FEEDBACK ON EARLIER DRAFTS:",
        *(recent_feedback(conn) or ["(none)"]),
    ])
    try:
        raws, usage = ctx.get_llm().propose(app["job_description"], master, context, max_items)
    except LLMError as e:
        raise GenerationError(str(e)) from None

    out = []
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("UPDATE proposals SET status = 'superseded' WHERE application_id = ? AND status = 'pending'",
                     (app_id,))
        for raw in raws:
            p, notes = check_proposal(raw, master)
            if p is None:
                io.out(f"  dropped a draft: {'; '.join(notes)}")
                continue
            pid = _insert_proposal(conn, app_id, p, model=ctx.cfg.model)
            out.append(get_proposal(conn, pid))
        db.add_usage(conn, app_id, usage, ctx.cfg.model)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return out


def proposal_view(p: dict[str, Any], master: Master) -> dict[str, Any]:
    """JSON-friendly view of a proposal for MCP clients."""
    idx = item_index(master)
    return {
        "proposal_id": p["id"],
        "kind": p["kind"],
        "origin": p["origin"],
        "job_id": p["job_id"],
        "if_approved": ("FACT (reusable, used in this CV)" if p["destination"] == "fact"
                        else "UNSAFE (new claim; not used until the user promotes it in the terminal)"),
        "sources": [{"id": b, "text": item_text(idx[b], "en")} for b in p["based_on"] if b in idx],
        "replaces": p["replaces"],
        "en": p["en"],
        "pt": p["pt"],
        "rationale": p["rationale"],
        "notes": p["notes"],
        "revision": p["depth"],
    }


def format_proposal(p: dict[str, Any], master: Master, n: Optional[int] = None) -> list[str]:
    idx = item_index(master)
    where = f"{p['job_id']}" if p["kind"] == "bullet" else "summary"
    dest = ("approve -> FACT (reusable)" if p["destination"] == "fact"
            else "approve -> UNSAFE (new claim; unused until you run `apply.py unsafe promote`)")
    head = f"[{n}] " if n is not None else ""
    rev = f" (revision {p['depth']})" if p["depth"] else ""
    lines = [f"{head}#{p['id']} {p['origin']} | {where} | {dest}{rev}"]
    for b in p["based_on"]:
        if b in idx:
            lines.append(f"    from {b}: {item_text(idx[b], 'en')}")
    lines += [f"    EN: {p['en']}", f"    PT: {p['pt']}"]
    if p["rationale"]:
        lines.append(f"    why: {p['rationale']}")
    if p["notes"]:
        lines.append(f"    note: {p['notes']}")
    return lines


# ---------------------------------------------------------------- review

@dataclass
class ReviewResult:
    facts: list[str] = field(default_factory=list)       # stored fact ids
    unsafe: list[str] = field(default_factory=list)      # stored unsafe ids
    rejected: list[int] = field(default_factory=list)    # proposal ids
    revised: list[dict] = field(default_factory=list)    # new pending proposals to review
    notes: list[str] = field(default_factory=list)


def review(ctx: Context, app_id: int, replies: dict[int, str]) -> ReviewResult:
    """Apply the user's replies to pending proposals. Replies are validated before any
    change is made; each proposal is then handled in its own transaction."""
    ensure_enabled(ctx)
    conn = ctx.db()
    parsed: dict[int, tuple[bool, Optional[str], dict, str]] = {}
    for pid, reply in replies.items():
        p = get_proposal(conn, int(pid))
        if p is None or p["application_id"] != app_id:
            raise ValueError(f"proposal #{pid} does not belong to application #{app_id}")
        if p["status"] != "pending":
            raise ValueError(f"proposal #{pid} was already reviewed ({p['status']})")
        approve, comment = parse_reply(reply)
        if approve and comment and p["depth"] >= MAX_REVISIONS:
            raise ValueError(f"proposal #{pid} reached the revision limit ({MAX_REVISIONS}); answer y or n")
        parsed[int(pid)] = (approve, comment, p, reply)

    res = ReviewResult()
    for pid, (approve, comment, p, reply) in parsed.items():
        if approve and comment:
            _revise(ctx, app_id, p, reply, comment, res)
        elif approve:
            _store(ctx, app_id, p, reply, res)
        else:
            conn.execute(
                "UPDATE proposals SET status = 'rejected', reply = ?, comment = ?, reviewed_at = ? WHERE id = ?",
                (reply, comment, db.now_iso(), pid),
            )
            res.rejected.append(pid)
    if res.facts:
        ctx.invalidate_master()
    return res


def _revise(ctx: Context, app_id: int, p: dict, reply: str, comment: str, res: ReviewResult) -> None:
    master = ctx.master
    draft = {k: p[k] for k in ("kind", "origin", "based_on", "replaces", "tags", "en", "pt", "rationale")}
    draft["job_id"] = p["job_id"] or ""
    try:
        raw, usage = ctx.get_llm().revise(master, draft, comment, _sources_text(master, p["based_on"] + p["replaces"]))
    except LLMError as e:
        res.notes.append(f"#{p['id']}: revision failed ({e}); it is still pending")
        return
    new, notes = check_proposal(raw, master)
    conn = ctx.db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        db.add_usage(conn, app_id, usage, ctx.cfg.model)
        if new is None:
            res.notes.append(f"#{p['id']}: the revision was invalid ({'; '.join(notes)}); it is still pending")
        else:
            child = _insert_proposal(conn, app_id, new, model=ctx.cfg.model, parent_id=p["id"], depth=p["depth"] + 1)
            conn.execute(
                "UPDATE proposals SET status = 'revised', reply = ?, comment = ?, reviewed_at = ? WHERE id = ?",
                (reply, comment, db.now_iso(), p["id"]),
            )
            res.revised.append(get_proposal(conn, child))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _store(ctx: Context, app_id: int, p: dict, reply: str, res: ReviewResult) -> None:
    conn = ctx.db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        new_id = next_id(conn, ctx.master, p["kind"], p["job_id"])
        comments = [r["comment"] for r in conn.execute(
            "WITH RECURSIVE chain(id, parent_id, comment) AS ("
            " SELECT id, parent_id, comment FROM proposals WHERE id = ?"
            " UNION ALL SELECT q.id, q.parent_id, q.comment FROM proposals q JOIN chain c ON q.id = c.parent_id)"
            " SELECT comment FROM chain WHERE comment IS NOT NULL", (p["id"],))]
        item = {
            "id": new_id, "kind": p["kind"], "job_id": p["job_id"], "origin": p["origin"],
            "based_on": p["based_on"], "tags": p["tags"], "en": p["en"], "pt": p["pt"],
            "comment": " | ".join(reversed(comments)) or None,
            "proposal_id": p["id"], "source_application_id": app_id,
        }
        if p["destination"] == "fact":
            add_fact(conn, item, approved_via="review")
            res.facts.append(new_id)
        else:
            add_unsafe(conn, item, p["rationale"])
            res.unsafe.append(new_id)
        conn.execute(
            "UPDATE proposals SET status = 'approved', reply = ?, stored_id = ?, reviewed_at = ? WHERE id = ?",
            (reply, new_id, db.now_iso(), p["id"]),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------- apply to the CV

def _apply_one(sel: dict, p: dict, project_ids: frozenset = frozenset()) -> None:
    fid = p["stored_id"]
    if p["kind"] == "summary":
        sel["summary_id"] = fid
        return
    owner = p["job_id"]  # a job id or a project id
    if owner in project_ids and owner not in sel["project_ids"]:
        sel["project_ids"].append(owner)  # an approved project bullet brings its project in
    lst = sel["bullets"].setdefault(owner, [])
    # Remove every selected source too, not only `replaces`: a fact may never be
    # selected together with an item it rewords (selection rule 8).
    targets = [i for i in dict.fromkeys(p["replaces"] + p["based_on"]) if i in lst]
    if targets:
        pos = min(lst.index(t) for t in targets)
        for t in targets:
            lst.remove(t)
        lst.insert(pos, fid)
    else:
        lst.append(fid)


def apply_approved(ctx: Context, app_id: int, io: IO = IO()) -> Optional[Path]:
    """Put newly approved facts into the application's CV and re-render it."""
    conn = ctx.db()
    rows = [_row(r) for r in conn.execute(
        "SELECT * FROM proposals WHERE application_id = ? AND status = 'approved' AND destination = 'fact' "
        "AND applied = 0 ORDER BY id", (app_id,))]
    if not rows:
        return None
    ctx.invalidate_master()
    master = ctx.master
    app = get_app(ctx, app_id)
    sel = stored_selection(ctx, app).model_dump()
    used = list(app.get("generated_ids") or [])
    for p in rows:
        trial = json.loads(json.dumps(sel))
        _apply_one(trial, p, frozenset(x.id for x in master.projects))
        errors = validate_selection(Selection.model_validate(trial), master)
        if errors:
            io.out(f"  {p['stored_id']} saved as a fact but not used in #{app_id}: {errors[0]}")
            continue
        sel = trial
        used.append(p["stored_id"])
    fit = _fit(ctx, Selection.model_validate(sel), app["lang"], trim=True, io=io)
    path = Path(app["pdf_path"])
    conn.execute("BEGIN IMMEDIATE")
    try:
        db.update_application(conn, app_id, {
            "selection": fit.selection.model_dump(),
            "trimmed": fit.removed,
            "generated_ids": [u for u in used if u not in fit.removed],
        })
        conn.executemany("UPDATE proposals SET applied = 1 WHERE id = ?", [(p["id"],) for p in rows])
        write_atomic(path, fit.rendered.pdf)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    io.out(f"#{app_id} re-rendered -> {display_path(path)}")
    return path
