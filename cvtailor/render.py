"""Jinja2 -> HTML -> PDF (WeasyPrint), with page counting.

The renderer only ever sees a RenderInput. WeasyPrint is imported lazily so the rest
of the tool (and the test-suite) works on machines without Pango installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .config import TEMPLATES_DIR
from .master import item_index, job_sort_key, parse_date
from .schema import Master, RenderInput, Selection, skill_item_text, text_of


@lru_cache(maxsize=4)
def _load_i18n_cached(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_i18n(templates_dir: Path = TEMPLATES_DIR) -> dict[str, Any]:
    return _load_i18n_cached(str(templates_dir / "i18n.yaml"))


def make_render_input(master: Master, selection: Selection, lang: str,
                      templates_dir: Path = TEMPLATES_DIR) -> RenderInput:
    return RenderInput(master=master, selection=selection, lang=lang,
                       i18n=load_i18n(templates_dir)[lang])


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

    def txt(value) -> str:
        return text_of(value, lang)

    def links(items) -> list[dict[str, str]]:
        return [{"label": txt(link.label), "url": link.url} for link in items]

    def bullets(owner: str) -> list[str]:
        return [getattr(idx[b].obj, lang) for b in sel.bullets.get(owner, [])]

    def description(o) -> str:
        alt = sel.descriptions.get(o.id)
        return getattr(idx[alt].obj, lang) if alt else txt(o.description)

    pr = m.profile
    summary = idx[sel.summary_id].obj
    profile = {
        "name": pr.name,
        # a summary can bring its own headline
        "headline": txt(summary.headline if summary.headline is not None else pr.headline),
        "contact": [x for x in (pr.email, pr.phone, pr.location) if x],
        "links": links(pr.links),
    }

    jobs = sorted(m.jobs, key=lambda j: job_sort_key(j.start, j.end), reverse=True)
    exp = []
    for j in jobs:
        exp.append({
            "title": j.title.get(lang),
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
            "name": txt(p.name),
            "dates": format_range(p.start, p.end, t, t["present"]) if p.start else "",
            "links": links(p.links),
            "description": description(p),
            "bullets": bullets(pid),
        })

    # Each skill line is "label: items". Hands-on and language categories keep their
    # own label; working and familiar categories are merged into one line per tier.
    # Languages come last.
    def own_lines(tier: str) -> list[dict[str, str]]:
        return [{"label": idx[cid].obj.category.get(lang),
                 "text": ", ".join(skill_item_text(i, lang) for i in idx[cid].obj.items)}
                for cid in getattr(sel.skill_category_order, tier)]

    skill_lines = own_lines("hands_on")
    for tier in ("working", "familiar"):
        items: list[str] = []
        for cid in getattr(sel.skill_category_order, tier):
            items.extend(skill_item_text(i, lang) for i in idx[cid].obj.items)
        if items:
            skill_lines.append({"label": t["skills"][tier], "text": ", ".join(items)})
    skill_lines += own_lines("languages")

    education = []
    for e in sorted(m.education, key=lambda e: job_sort_key(e.start, e.end), reverse=True):
        dates = format_range(e.start, e.end, t, t["in_progress"])
        if e.expected:
            dates += f" ({t['expected']})"
        education.append({
            "degree": e.degree.get(lang),
            "institution": txt(e.institution),
            "note": txt(e.note),
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
    return {
        "lang": lang,
        "t": t,
        "profile": profile,
        "sections": [s for s in m.layout.sections if present[s]],
        "summary": getattr(summary, lang),
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

class WeasyPrintMissing(RuntimeError):
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
