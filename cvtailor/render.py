"""Jinja2 -> HTML -> PDF, with page counting.

Two PDF engines: a headless Chromium browser (Edge, preinstalled on Windows; Chrome
elsewhere), which needs nothing bundled, and WeasyPrint, which needs Pango/GTK.
CV_PDF_ENGINE picks one; the default uses a browser when one is found.

The renderer only ever sees a RenderInput. WeasyPrint is imported lazily so the rest
of the tool (and the test-suite) works on machines without Pango installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .config import TEMPLATES_DIR
from .langs import LANGUAGES
from .langs import name as lang_name
from .master import item_index, job_sort_key, parse_date, shown_skill_items
from .schema import Master, RenderInput, Selection, has_text, text_of


class MissingTextError(ValueError):
    """The CV would show items that have no text in its language."""

    def __init__(self, lang: str, missing: list[str]):
        self.lang, self.missing = lang, missing
        shown = missing[:12] + ([f"and {len(missing) - 12} more"] if len(missing) > 12 else [])
        super().__init__(f"no {lang_name(lang)} text yet for: " + ", ".join(shown)
                         + f". Add it in the Master tab (or master.yaml), or make the CV in another language.")


@lru_cache(maxsize=4)
def _load_i18n_cached(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_i18n(templates_dir: Path = TEMPLATES_DIR) -> dict[str, Any]:
    return _load_i18n_cached(str(templates_dir / "i18n.yaml"))


def make_render_input(master: Master, selection: Selection, lang: str,
                      templates_dir: Path = TEMPLATES_DIR) -> RenderInput:
    i18n = load_i18n(templates_dir)
    if lang not in i18n or lang not in LANGUAGES:
        raise ValueError(f"no CV template words for language '{lang}' (known: {', '.join(LANGUAGES)})")
    return RenderInput(master=master, selection=selection, lang=lang, i18n=i18n[lang])


# ---------------------------------------------------------------- context

def format_date(value: Optional[str], t: dict[str, Any], open_label: str) -> str:
    if not value:
        return open_label
    p = parse_date(value)
    if p is None:
        return value
    year, month = p
    if month is None:
        return str(year)
    return t["date_format"].format(month=t["months"][month - 1], mm=f"{month:02d}", year=year)


def format_range(start: str, end: Optional[str], t: dict[str, Any], open_label: str) -> str:
    return f"{format_date(start, t, open_label)}{t['range_sep']}{format_date(end, t, open_label)}"


def build_context(ri: RenderInput) -> dict[str, Any]:
    m, sel, lang, t = ri.master, ri.selection, ri.lang, ri.i18n
    idx = item_index(m)
    missing: list[str] = []  # everything shown must have text in this language

    def txt(value, what: str = "") -> str:
        if not has_text(value, lang):
            missing.append(what or getattr(value, "id", "?"))
        return text_of(value, lang)

    def links(items, what: str) -> list[dict[str, str]]:
        return [{"label": txt(link.label, f"{what} link"), "url": link.url} for link in items]

    def bullets(owner: str) -> list[str]:
        return [txt(idx[b].obj, b) for b in sel.bullets.get(owner, [])]

    def description(o) -> str:
        alt = sel.descriptions.get(o.id)
        return txt(idx[alt].obj, alt) if alt else txt(o.description, f"{o.id} description")

    pr = m.profile
    summary = idx[sel.summary_id].obj
    # A picked headline option wins; otherwise a summary can bring its own headline.
    alt = next((h for h in pr.headline_alts if h.id == sel.headline_id), None)
    headline = (txt(alt, alt.id) if alt else
                txt(summary.headline, f"{summary.id} headline") if summary.headline is not None
                else txt(pr.headline, "profile headline"))
    profile = {
        "name": pr.name,
        "headline": headline,
        "contact": [x for x in (pr.email, pr.phone, pr.location) if x],
        "links": links(pr.links, "profile"),
    }

    jobs = sorted(m.jobs, key=lambda j: job_sort_key(j.start, j.end), reverse=True)
    exp = []
    for j in jobs:
        exp.append({
            "title": txt(j.title, f"{j.id} title"),
            "company": j.company or "",
            "location": j.location or "",
            "dates": format_range(j.start, j.end, t, t["present"]),
            "description": description(j),
            "bullets": bullets(j.id),
        })

    projects = []
    for pid in sel.project_ids:
        p = idx[pid].obj
        projects.append({
            "name": txt(p.name, f"{p.id} name"),
            "dates": format_range(p.start, p.end, t, t["present"]) if p.start else "",
            "links": links(p.links, p.id),
            "description": description(p),
            "bullets": bullets(pid),
        })

    # Each skill line is "label: items". Hands-on and language categories keep their
    # own label; working and familiar categories are merged into one line per tier.
    # Languages come last.
    def shown(cid: str) -> list[str]:  # the selection may drop or reorder a category's items
        return [txt(i, f"{cid} item") for i in shown_skill_items(idx[cid].obj, sel.skill_items.get(cid))]

    def own_lines(tier: str) -> list[dict[str, str]]:
        return [{"label": txt(idx[cid].obj.category, cid), "text": ", ".join(shown(cid))}
                for cid in getattr(sel.skill_category_order, tier)]

    skill_lines = own_lines("hands_on")
    for tier in ("working", "familiar"):
        items: list[str] = []
        for cid in getattr(sel.skill_category_order, tier):
            items.extend(shown(cid))
        if items:
            skill_lines.append({"label": t["skills"][tier], "text": ", ".join(items)})
    skill_lines += own_lines("languages")

    education = []
    for e in sorted(m.education, key=lambda e: job_sort_key(e.start, e.end), reverse=True):
        dates = format_range(e.start, e.end, t, t["in_progress"])
        if e.expected:
            dates += f" ({t['expected']})"
        education.append({
            "degree": txt(e.degree, f"{e.id} degree"),
            "institution": txt(e.institution, f"{e.id} institution"),
            "note": txt(e.note, f"{e.id} note"),
            "dates": dates,
            "details": bullets(e.id),
        })

    present = {
        "summary": True,
        "experience": bool(exp),
        "projects": bool(projects),
        "skills": bool(skill_lines),
        "education": bool(education),
    }
    summary_text = txt(summary, summary.id)
    if missing:
        raise MissingTextError(lang, list(dict.fromkeys(missing)))
    return {
        "lang": lang,
        "html_lang": LANGUAGES[lang].html,
        "t": t,
        "profile": profile,
        "sections": [s for s in m.layout.sections if present[s]],
        "summary": summary_text,
        "experience": exp,
        "projects": projects,
        "projects_heading": t["headings"]["project" if len(projects) == 1 else "projects"],
        "skill_lines": skill_lines,
        "education": education,
    }


@lru_cache(maxsize=4)
def _env(templates_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(ri: RenderInput, templates_dir: Path = TEMPLATES_DIR) -> str:
    if not isinstance(ri, RenderInput):  # enforce the boundary at runtime too
        raise TypeError("render_html only accepts a RenderInput")
    env = _env(str(templates_dir))
    css = (templates_dir / "cv.css").read_text(encoding="utf-8")
    return env.get_template("cv.html.j2").render(css=css, **build_context(ri))


# ---------------------------------------------------------------- PDF

class PdfEngineMissing(RuntimeError):
    """No PDF engine is available (neither a Chromium browser nor WeasyPrint)."""


class WeasyPrintMissing(PdfEngineMissing):
    pass


def html_to_pdf(html: str, base_url: Path = TEMPLATES_DIR) -> tuple[int, bytes]:
    try:
        from weasyprint import HTML  # noqa: WPS433
    except (ImportError, OSError) as e:
        raise WeasyPrintMissing(
            "WeasyPrint could not load (Pango/GTK libraries missing?). "
            "See README -> Installing WeasyPrint. Original error: " + str(e).splitlines()[0]
        ) from None
    doc = HTML(string=html, base_url=str(base_url)).render()
    return len(doc.pages), doc.write_pdf()


@dataclass
class Rendered:
    pages: int
    pdf: bytes
    html: str


class Renderer(Protocol):
    def render(self, ri: RenderInput) -> Rendered: ...


class WeasyRenderer:
    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir

    def render(self, ri: RenderInput) -> Rendered:
        html = render_html(ri, self.templates_dir)
        pages, pdf = html_to_pdf(html, self.templates_dir)
        return Rendered(pages, pdf, html)


# ---------------------------------------------------------------- PDF via a browser

_BROWSER_CANDIDATES = [
    # Edge ships with Windows 10 and 11; its print-to-PDF uses the same Chromium engine
    # as the app window (WebView2), so the preview and the PDF match.
    r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
    r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
    r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
_PAGE_RE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def find_browser() -> Optional[str]:
    """A Chromium-based browser that can print to PDF headlessly, or None."""
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for cand in _BROWSER_CANDIDATES:
        path = Path(os.path.expandvars(cand))
        if "%" not in str(path) and path.exists():
            return str(path)
    return None


def count_pdf_pages(pdf: bytes) -> int:
    """Pages in a PDF written by Chromium (page objects are not packed in object streams)."""
    return len(_PAGE_RE.findall(pdf))


class BrowserRenderer:
    """PDF through a headless Chromium browser (Edge on Windows). Needs nothing bundled,
    and writes its temporary profile in a temp folder, not in the user's profile."""

    def __init__(self, browser: Optional[str] = None, templates_dir: Path = TEMPLATES_DIR,
                 timeout: float = 60.0):
        self.browser = browser or find_browser()
        if not self.browser:
            raise PdfEngineMissing("no Edge or Chrome found to print PDFs; install one, or WeasyPrint")
        self.templates_dir = templates_dir
        self.timeout = timeout

    def render(self, ri: RenderInput) -> Rendered:
        html = render_html(ri, self.templates_dir)
        with tempfile.TemporaryDirectory(prefix="cvtailor-pdf-") as tmp:
            src, out = Path(tmp) / "cv.html", Path(tmp) / "cv.pdf"
            src.write_text(html, encoding="utf-8")
            cmd = [self.browser, "--headless", "--disable-gpu", "--no-first-run",
                   "--no-pdf-header-footer", "--print-to-pdf-no-header",
                   f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={out}", src.as_uri()]
            try:
                subprocess.run(cmd, capture_output=True, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired:
                raise PdfEngineMissing(f"the browser did not print the PDF within {self.timeout:.0f} s") from None
            if not out.exists() or out.stat().st_size == 0:
                raise PdfEngineMissing(f"the browser ({Path(self.browser).name}) did not write a PDF")
            pdf = out.read_bytes()
        return Rendered(count_pdf_pages(pdf), pdf, html)


def make_renderer(engine: str = "auto") -> Renderer:
    """'browser' (Edge/Chrome), 'weasyprint', or 'auto': a browser if one is found,
    else WeasyPrint."""
    engine = (engine or "auto").lower()
    if engine == "weasyprint":
        return WeasyRenderer()
    if engine in ("browser", "edge", "chrome"):
        return BrowserRenderer()
    browser = find_browser()
    return BrowserRenderer(browser) if browser else WeasyRenderer()
