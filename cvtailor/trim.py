"""Deterministic page-fit trimming. Removes content only, never alters text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .master import item_index, item_text, parse_date
from .render import Rendered, make_render_input
from .schema import Master, Selection


class PageFitError(Exception):
    pass


@dataclass
class FitResult:
    selection: Selection
    rendered: Rendered
    removed: list[str] = field(default_factory=list)


def _start_key(master: Master, owner: str) -> tuple:
    for e in [*master.jobs, *master.projects, *master.education]:
        if e.id == owner:
            p = parse_date(e.start or "") or (0, None)
            return (p[0], p[1] or 1)
    return (0, 0)


def next_removal(sel: Selection, master: Master) -> tuple[str, str] | None:
    """Return (kind, id) of the next item to drop, or None if nothing can go.

    Bullets go first, from the job, project or education entry with the most bullets
    (tie: the one that started earlier). Jobs keep min_bullets_per_job; projects and
    education details can go down to none.
    Then whole projects go, last selected first.
    """
    lim = master.limits
    jobs = {j.id for j in master.jobs}
    candidates = [
        (owner, ids) for owner, ids in sel.bullets.items()
        if len(ids) > (lim.min_bullets_per_job if owner in jobs else 0)
    ]
    if candidates:
        owner, ids = min(candidates, key=lambda c: (-len(c[1]), _start_key(master, c[0])))
        return "bullet", ids[-1]
    if sel.project_ids:
        return "project", sel.project_ids[-1]
    return None


def remove_item(sel: Selection, kind: str, id_: str) -> Selection:
    data = sel.model_dump()
    if kind == "bullet":
        for ids in data["bullets"].values():
            if id_ in ids:
                ids.remove(id_)
    else:
        data["project_ids"].remove(id_)
        data["bullets"].pop(id_, None)
        data["descriptions"].pop(id_, None)
    return Selection.model_validate(data)


def fit_to_page(
    sel: Selection,
    master: Master,
    lang: str,
    render: Callable,
    *,
    trim: bool = True,
    on_remove: Callable[[str, str], None] | None = None,
) -> FitResult:
    """render(RenderInput) -> Rendered. Trims until pages <= limits.max_pages."""
    max_pages = master.limits.max_pages
    idx = item_index(master)
    removed: list[str] = []
    while True:
        rendered = render(make_render_input(master, sel, lang))
        if rendered.pages <= max_pages:
            return FitResult(sel, rendered, removed)
        if not trim:
            raise PageFitError(
                f"the CV is {rendered.pages} pages ({lang.upper()}), limit is {max_pages}. "
                "Use `apply.py edit <id>` to choose what to drop, or run without --no-trim."
            )
        nxt = next_removal(sel, master)
        if nxt is None:
            raise PageFitError(
                f"the CV is still {rendered.pages} pages ({lang.upper()}) with every job at the "
                "minimum bullets and no projects; shorten the master text or raise max_pages."
            )
        kind, id_ = nxt
        sel = remove_item(sel, kind, id_)
        snippet = item_text(idx[id_], lang)[:60]
        removed.append(id_)
        if on_remove:
            on_remove(id_, snippet)
