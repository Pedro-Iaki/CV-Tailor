"""Annotated-YAML editing of a selection in $EDITOR."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .master import item_index, item_text
from .schema import TIERS, Master, Selection, text_of

ERROR_MARK = "# !! "
HEADER = """\
# Edit the selection, then save and close the editor.
# - Reorder lines to change order; delete a line to drop an item.
# - Uncomment a '# - id' line to add an item (keep the indentation).
# - Comments after '#' are ignored; ids are what count.
# - Save an empty file to abort without changes.
"""


def _note(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def to_annotated_yaml(sel: Selection, master: Master, errors: Optional[list[str]] = None) -> str:
    idx = item_index(master)
    out: list[str] = []
    if errors:
        out.append(ERROR_MARK + "The edit was rejected. Fix these problems and save again:")
        out += [f"{ERROR_MARK}  - {e}" for e in errors]
        out.append("")
    out.append(HEADER)

    def line(indent: str, id_: str, commented: bool = False, dash: bool = True) -> str:
        it = idx.get(id_)
        note = _note(item_text(it, "en")) if it else "(unknown id)"
        alt = getattr(it.obj, "alt_of", None) if it else None
        if alt:
            note = f"[alt of {', '.join(alt)}] " + note
        body = f"- {id_}" if dash else id_
        prefix = f"{indent}# " if commented else indent
        return f"{prefix}{body}   # {note}"

    # summary
    s = idx.get(sel.summary_id)
    note = _note(item_text(s, "en")) if s else "(unknown id)"
    out.append(f"summary_id: {sel.summary_id}   # {note}")
    for alt in master.summaries:
        if alt.id != sel.summary_id:
            out.append(f"# summary_id: {alt.id}   # {_note(alt.en)}")

    # projects
    out.append("project_ids:")
    for pid in sel.project_ids:
        out.append(line("  ", pid))
    for p in master.projects:
        if p.id not in sel.project_ids:
            out.append(line("  ", p.id, commented=True))

    # bullets: every job, then every project that has bullets. A project with no
    # chosen bullets has its key commented out too; uncomment it to add some.
    out.append("bullets:   # a project needs to be in project_ids to show its bullets")
    out.append("           # versions of the same item ('alt of' below) can't be selected together")
    jobs = {j.id for j in master.jobs}
    owners = [*master.jobs, *(o for o in [*master.projects, *master.education] if o.bullets)]
    for o in owners:
        chosen = sel.bullets.get(o.id, [])
        hide_key = o.id not in jobs and o.id not in sel.bullets
        key = f"  # {o.id}:" if hide_key else f"  {o.id}:"
        out.append(f"{key}   # {_note(item_text(idx[o.id], 'en'))}")
        for bid in chosen:
            out.append(line("    ", bid))
        for b in o.bullets:
            if b.id not in chosen:
                out.append(line("    ", b.id, commented=True))
    known = {o.id for o in owners}
    for owner, chosen in sel.bullets.items():  # keep unknown keys visible
        if owner not in known:
            out.append(f"  {owner}:")
            out += [line("    ", b) for b in chosen]

    # description alternatives
    with_alts = [o for o in [*master.jobs, *master.projects] if o.description_alts]
    known_desc = {o.id for o in with_alts}
    stray = {k: v for k, v in sel.descriptions.items() if k not in known_desc}
    if with_alts or stray:
        out.append("descriptions:   # job or project -> one of its description alternatives; 'default' keeps the default")
        for o in with_alts:
            options = {"default": text_of(o.description, "en"), **{d.id: d.en for d in o.description_alts}}
            chosen = sel.descriptions.get(o.id, "default")
            out.append(f"  {o.id}: {chosen}   # {_note(options.get(chosen, '(unknown id)'))}")
            for opt, text in options.items():
                if opt != chosen:
                    out.append(f"  # {o.id}: {opt}   # {_note(text)}")
        for owner, did in stray.items():  # keep unknown keys visible
            out.append(f"  {owner}: {did}")

    # skills
    out.append("skill_category_order:")
    for tier in TIERS:
        chosen = getattr(sel.skill_category_order, tier)
        out.append(f"  {tier}:")
        for cid in chosen:
            out.append(line("    ", cid))
        for c in master.skills.tier(tier):
            if c.id not in chosen:
                out.append(line("    ", c.id, commented=True))

    # skill items: per chosen category, which of its own items show and in what order
    out.append("skill_items:   # per category: reorder or delete items; a commented-out category shows all")
    cats = {c.id: c for t in TIERS for c in master.skills.tier(t)}
    for tier in TIERS:
        for cid in getattr(sel.skill_category_order, tier):
            c = cats.get(cid)
            if c is None:
                continue
            own = [text_of(i, "en") for i in c.items]
            if cid in sel.skill_items:
                chosen = sel.skill_items[cid]
                out.append(f"  {cid}:")
                out += [f"    - {json.dumps(t, ensure_ascii=False)}" for t in chosen]
                out += [f"    # - {json.dumps(t, ensure_ascii=False)}" for t in own if t not in chosen]
            else:
                out.append(f"  # {cid}:")
                out += [f"  #   - {json.dumps(t, ensure_ascii=False)}" for t in own]
    for cid, chosen in sel.skill_items.items():  # keep unknown keys visible
        if cid not in cats:
            out.append(f"  {cid}:")
            out += [f"    - {json.dumps(t, ensure_ascii=False)}" for t in chosen]
    return "\n".join(out) + "\n"


class EditAborted(Exception):
    pass


def strip_error_header(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith(ERROR_MARK)).lstrip("\n") + "\n"


def parse_annotated(text: str) -> Selection:
    """Parse edited YAML. Raises EditAborted on an empty file, ValueError on bad input."""
    meaningful = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not meaningful:
        raise EditAborted("empty file, edit aborted")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML error: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("the file must be a YAML mapping")
    try:
        return Selection.model_validate(data)
    except ValidationError as e:
        raise ValueError("; ".join(
            f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()
        )) from None


def editor_command() -> list[str]:
    ed = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if ed:
        return shlex.split(ed, posix=(os.name != "nt"))
    return ["notepad"] if sys.platform.startswith("win") else ["nano"]


def open_in_editor(text: str, suffix: str = ".yaml") -> str:
    fd, name = tempfile.mkstemp(prefix="cv-selection-", suffix=suffix)
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        subprocess.call([*editor_command(), str(path)])
        return path.read_text(encoding="utf-8-sig")
    finally:
        try:
            path.unlink()
        except OSError:
            pass
