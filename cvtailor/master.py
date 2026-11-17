"""Load and validate the master YAML; build the tag vocabulary and the LLM index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from pydantic import ValidationError

from .schema import (
    TIERS,
    Bilingual,
    Master,
    SkillCategory,
    skill_item_text,
    text_of,
)

TAG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DATE_RE = re.compile(r"^(\d{4})(?:-(0[1-9]|1[0-2]))?$")


class MasterError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("master file is invalid:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------- loading

def _format_pydantic(err: ValidationError) -> list[str]:
    out = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        out.append(f"{loc}: {e['msg']}")
    return out


def parse_master(data: Any) -> Master:
    if not isinstance(data, dict):
        raise MasterError(["master file must be a YAML mapping"])
    try:
        master = Master.model_validate(data)
    except ValidationError as e:
        raise MasterError(_format_pydantic(e)) from None
    errors = check_master(master)
    if errors:
        raise MasterError(errors)
    return master


def load_master(path: Path | str) -> Master:
    path = Path(path)
    if not path.exists():
        raise MasterError([f"master file not found: {path} (start from examples/master.example.yaml)"])
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise MasterError([f"YAML parse error: {e}"]) from None
    return parse_master(data)


# ---------------------------------------------------------------- semantic checks

def parse_date(s: str) -> Optional[tuple[int, Optional[int]]]:
    m = DATE_RE.match(s or "")
    if not m:
        return None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def date_lt(a: tuple[int, Optional[int]], b: tuple[int, Optional[int]]) -> bool:
    """a < b, comparing months only when both dates have them."""
    if a[0] != b[0]:
        return a[0] < b[0]
    if a[1] is None or b[1] is None:
        return False
    return a[1] < b[1]


def _iter_skill_categories(master: Master) -> Iterable[tuple[str, SkillCategory]]:
    for tier in TIERS:
        for cat in master.skills.tier(tier):
            yield tier, cat


def check_master(m: Master) -> list[str]:
    errors: list[str] = []

    # 1. unique ids
    seen: dict[str, str] = {}

    def see(id_: str, where: str) -> None:
        if not id_ or not str(id_).strip():
            errors.append(f"{where}: empty id")
            return
        if id_ in seen:
            errors.append(f"duplicate id '{id_}' ({seen[id_]} and {where})")
        else:
            seen[id_] = where

    for s in m.summaries:
        see(s.id, "summaries")
    for j in m.jobs:
        see(j.id, "jobs")
        for b in j.bullets:
            see(b.id, f"bullets of {j.id}")
    for p in m.projects:
        see(p.id, "projects")
        for b in p.bullets:
            see(b.id, f"bullets of {p.id}")
    for e in m.education:
        see(e.id, "education")
        for b in e.bullets:
            see(b.id, f"bullets of {e.id}")
    for o in [*m.jobs, *m.projects]:
        for d in o.description_alts:
            see(d.id, f"description_alts of {o.id}")
    for tier, c in _iter_skill_categories(m):
        see(c.id, f"skills.{tier}")

    # 2. non-empty bilingual text
    def need_text(obj: Any, where: str) -> None:
        if isinstance(obj, str):  # a Text given as one string for both languages
            if not obj.strip():
                errors.append(f"{where}: empty")
            return
        for lang in ("en", "pt"):
            if not str(getattr(obj, lang, "") or "").strip():
                errors.append(f"{where}: '{lang}' is empty")

    def maybe_text(obj: Any, where: str) -> None:
        if obj is not None:
            need_text(obj, where)

    if m.profile.headline is not None:
        need_text(m.profile.headline, "profile headline")
    for i, link in enumerate(m.profile.links):
        need_text(link.label, f"profile link {i + 1} label")

    for s in m.summaries:
        need_text(s, f"summary {s.id}")
        maybe_text(s.headline, f"summary {s.id} headline")
    for o in [*m.jobs, *m.projects]:
        for d in o.description_alts:
            need_text(d, f"description {d.id}")
        if o.description_alts and o.description is None:
            errors.append(f"{o.id}: description_alts need a default description")
    for j in m.jobs:
        need_text(j.title, f"job {j.id} title")
        maybe_text(j.description, f"job {j.id} description")
        for b in j.bullets:
            need_text(b, f"bullet {b.id}")
    for p in m.projects:
        need_text(p.name, f"project {p.id} name")
        maybe_text(p.description, f"project {p.id} description")
        for i, link in enumerate(p.links):
            need_text(link.label, f"project {p.id} link {i + 1} label")
        for b in p.bullets:
            need_text(b, f"bullet {b.id}")
        if p.description is None and not p.bullets:
            errors.append(f"project {p.id}: needs a description or bullets")
    for e in m.education:
        need_text(e.degree, f"education {e.id} degree")
        need_text(e.institution, f"education {e.id} institution")
        maybe_text(e.note, f"education {e.id} note")
        for b in e.bullets:
            need_text(b, f"bullet {b.id}")
        if e.expected and e.end is None:
            errors.append(f"education {e.id}: 'expected' needs an end date")
    for tier, c in _iter_skill_categories(m):
        need_text(c.category, f"skill category {c.id}")
        for i, item in enumerate(c.items):
            if isinstance(item, Bilingual):
                need_text(item, f"skill category {c.id} item {i + 1}")
            elif not item.strip():
                errors.append(f"skill category {c.id} item {i + 1}: empty")

    # 3. skill items unique across tiers and overall, except between a category and
    # its alternatives (which are never shown together)
    conflicts = alt_conflicts(m)
    first_seen: dict[str, list[tuple[str, str]]] = {}
    for tier, c in _iter_skill_categories(m):
        for item in c.items:
            key = skill_item_text(item, "en").strip().lower()
            if not key:
                continue
            clash = [(t, cid) for t, cid in first_seen.get(key, []) if cid not in conflicts.get(c.id, ())]
            first_seen.setdefault(key, []).append((tier, c.id))
            if clash:
                prev_tier, prev_cat = clash[0]
                if prev_tier != tier:
                    errors.append(
                        f"skill '{skill_item_text(item)}' appears in more than one tier "
                        f"({prev_tier}/{prev_cat} and {tier}/{c.id})"
                    )
                else:
                    errors.append(
                        f"skill '{skill_item_text(item)}' appears more than once "
                        f"({prev_cat} and {c.id})"
                    )

    # 3b. alternatives point at existing items of the same kind (bullets: same owner)
    idx = item_index(m)
    for it in idx.values():
        for target in getattr(it.obj, "alt_of", None) or []:
            t = idx.get(target)
            kind = it.kind.replace("_", " ")
            where = f"{kind} {it.id}: alt_of '{target}'"
            if target == it.id:
                errors.append(f"{where} is the item itself")
            elif t is None:
                errors.append(f"{where} does not exist")
            elif it.kind == "description":
                if t.kind != "bullet" or t.parent != it.parent:
                    errors.append(f"{where} must be a bullet of {it.parent}")
            elif t.kind != it.kind:
                errors.append(f"{where} is a {t.kind.replace('_', ' ')}, not a {kind}")
            elif it.kind == "bullet" and t.parent != it.parent:
                errors.append(f"{where} belongs to {t.parent}, not {it.parent}")

    # 4. tags
    def check_tags(tags: list[str], where: str) -> None:
        if not tags:
            errors.append(f"{where}: tags list is empty")
        for t in tags:
            if not isinstance(t, str) or not TAG_RE.match(t):
                errors.append(f"{where}: invalid tag '{t}' (use lowercase a-z, 0-9 and single hyphens)")

    for s in m.summaries:
        check_tags(s.tags, f"summary {s.id}")
    for j in m.jobs:
        for b in j.bullets:
            check_tags(b.tags, f"bullet {b.id}")
    for p in m.projects:
        check_tags(p.tags, f"project {p.id}")
        for b in p.bullets:
            check_tags(b.tags, f"bullet {b.id}")
    for e in m.education:
        for b in e.bullets:
            check_tags(b.tags, f"bullet {b.id}")
    for o in [*m.jobs, *m.projects]:
        for d in o.description_alts:
            check_tags(d.tags, f"description {d.id}")

    # 5. dates
    def check_dates(start: str, end: Optional[str], where: str) -> None:
        ps = parse_date(start)
        if ps is None:
            errors.append(f"{where}: start '{start}' is not YYYY-MM or YYYY")
        if end is None:
            return
        pe = parse_date(end)
        if pe is None:
            errors.append(f"{where}: end '{end}' is not YYYY-MM or YYYY")
        elif ps is not None and date_lt(pe, ps):
            errors.append(f"{where}: end {end} is earlier than start {start}")

    for j in m.jobs:
        check_dates(j.start, j.end, f"job {j.id}")
    for e in m.education:
        check_dates(e.start, e.end, f"education {e.id}")
    for p in m.projects:
        if p.start is not None:
            check_dates(p.start, p.end, f"project {p.id}")
        elif p.end is not None:
            errors.append(f"project {p.id}: end date without a start date")

    # 6. tag aliases
    used_tags = tag_vocabulary(m)
    alias_owner: dict[str, str] = {}
    for tag, aliases in m.tag_aliases.items():
        if tag not in used_tags:
            errors.append(f"tag_aliases: '{tag}' is not a tag used anywhere in the file")
        for a in aliases or []:
            key = " ".join(str(a).lower().split())
            if not key:
                errors.append(f"tag_aliases.{tag}: empty alias")
                continue
            if key in alias_owner:
                errors.append(
                    f"tag_aliases: alias '{a}' is used by both '{alias_owner[key]}' and '{tag}'"
                )
            else:
                alias_owner[key] = tag

    # 7. limits
    lim = m.limits
    for name in ("min_bullets_per_job", "max_bullets_per_job", "max_bullets_per_project",
                 "max_total_bullets", "max_projects", "max_pages"):
        if getattr(lim, name) < 1:
            errors.append(f"limits.{name} must be a positive integer")
    if lim.min_bullets_per_job > lim.max_bullets_per_job:
        errors.append("limits: min_bullets_per_job is greater than max_bullets_per_job")
    if lim.min_bullets_per_job * len(m.jobs) > lim.max_total_bullets:
        errors.append(
            f"limits: min_bullets_per_job ({lim.min_bullets_per_job}) x jobs ({len(m.jobs)}) "
            f"exceeds max_total_bullets ({lim.max_total_bullets})"
        )
    for j in m.jobs:
        if len(j.bullets) < lim.min_bullets_per_job:
            errors.append(
                f"job {j.id} has {len(j.bullets)} bullets, fewer than min_bullets_per_job "
                f"({lim.min_bullets_per_job})"
            )

    # 8. layout
    for name in sorted({s for s in m.layout.sections if m.layout.sections.count(s) > 1}):
        errors.append(f"layout.sections: '{name}' is listed more than once")

    if not m.summaries:
        errors.append("summaries: at least one summary is required")
    if not m.jobs:
        errors.append("jobs: at least one job is required")
    return errors


# ---------------------------------------------------------------- lookups

def tag_vocabulary(m: Master) -> set[str]:
    tags: set[str] = set()
    for s in m.summaries:
        tags.update(s.tags)
    for j in m.jobs:
        for b in j.bullets:
            tags.update(b.tags)
    for p in m.projects:
        tags.update(p.tags)
        for b in p.bullets:
            tags.update(b.tags)
    for e in m.education:
        for b in e.bullets:
            tags.update(b.tags)
    for o in [*m.jobs, *m.projects]:
        for d in o.description_alts:
            tags.update(d.tags)
    return tags


@dataclass
class Item:
    id: str
    kind: str  # summary | job | bullet | project | education | skill_category | description
    obj: Any
    parent: Optional[str] = None  # bullets, descriptions: id of the job, project or education entry
    tier: Optional[str] = None


def item_index(m: Master) -> dict[str, Item]:
    out: dict[str, Item] = {}
    for s in m.summaries:
        out[s.id] = Item(s.id, "summary", s)
    for j in m.jobs:
        out[j.id] = Item(j.id, "job", j)
        for b in j.bullets:
            out[b.id] = Item(b.id, "bullet", b, parent=j.id)
        for d in j.description_alts:
            out[d.id] = Item(d.id, "description", d, parent=j.id)
    for p in m.projects:
        out[p.id] = Item(p.id, "project", p)
        for b in p.bullets:
            out[b.id] = Item(b.id, "bullet", b, parent=p.id)
        for d in p.description_alts:
            out[d.id] = Item(d.id, "description", d, parent=p.id)
    for e in m.education:
        out[e.id] = Item(e.id, "education", e)
        for b in e.bullets:
            out[b.id] = Item(b.id, "bullet", b, parent=e.id)
    for tier, c in _iter_skill_categories(m):
        out[c.id] = Item(c.id, "skill_category", c, tier=tier)
    return out


def item_text(item: Item, lang: str = "en") -> str:
    o = item.obj
    if item.kind in ("summary", "bullet", "description"):
        return getattr(o, lang)
    if item.kind == "project":
        return text_of(o.name, lang)
    if item.kind == "job":
        return f"{o.title.get(lang)} @ {o.company}" if o.company else o.title.get(lang)
    if item.kind == "education":
        return f"{o.degree.get(lang)}, {text_of(o.institution, lang)}"
    if item.kind == "skill_category":
        return f"{o.category.get(lang)}: " + ", ".join(skill_item_text(i, lang) for i in o.items)
    return ""


def shown_skill_items(cat: SkillCategory, chosen: Optional[list[str]]) -> list[Any]:
    """The items of a skill category a selection shows: `chosen` (exact English texts,
    in order) or, when None, every item in master order. Unknown texts are skipped
    here; validate_selection reports them."""
    if chosen is None:
        return list(cat.items)
    by_en = {skill_item_text(i, "en"): i for i in cat.items}
    return [by_en[t] for t in chosen if t in by_en]


def alt_conflicts(m: Master) -> dict[str, set[str]]:
    """id -> ids it may never be selected with.

    An item conflicts with the items it is an alternative of (master `alt_of`) or a
    rewording of (approved facts, `based_on`), and with other alternatives or
    rewordings of the same items. Skill categories are the exception to the last
    part: two categories that replace the same one can be halves of one alternative
    layout (a "Hands-on" line and a "Working knowledge" line), so they only conflict
    with what they list in alt_of.
    """
    out: dict[str, set[str]] = {}
    by_target: dict[str, list[str]] = {}

    def link(a: str, b: str) -> None:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)

    for it in item_index(m).values():
        targets = [*(getattr(it.obj, "alt_of", None) or []), *(getattr(it.obj, "based_on", None) or [])]
        for target in targets:
            if target != it.id:
                link(it.id, target)
                if it.kind != "skill_category":
                    by_target.setdefault(target, []).append(it.id)
    for ids in by_target.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if a != b:
                    link(a, b)
    return out


def job_sort_key(start: str, end: Optional[str]) -> tuple:
    """Sort key for reverse-chronological order (use with reverse=True)."""
    e = parse_date(end) if end else (9999, 12)
    s = parse_date(start) or (0, 0)
    e = e or (0, 0)
    return (e[0], e[1] or 12, s[0], s[1] or 1)


# ---------------------------------------------------------------- LLM index

def _never_with(id_: str, conflicts: dict[str, set[str]]) -> dict[str, Any]:
    # Alternatives and approved rewordings, listed on both sides so the model sees
    # the exclusion whichever item it looks at.
    return {"never_with": sorted(conflicts[id_])} if conflicts.get(id_) else {}


def _index_item(it: Any, conflicts: dict[str, set[str]]) -> dict[str, Any]:
    d = {"id": it.id, "tags": it.tags, "text": it.en}
    if getattr(it, "headline", None):
        d["headline"] = text_of(it.headline)  # shown under the name when this summary is picked
    return d | _never_with(it.id, conflicts)


def _descriptions(o: Any, conflicts: dict[str, set[str]]) -> dict[str, Any]:
    if not o.description_alts:
        return {"text": text_of(o.description)} if o.description is not None else {}
    return {"description_options": [
        {"id": "default", "text": text_of(o.description)},
        *({"id": d.id, "tags": d.tags, "text": d.en} | _never_with(d.id, conflicts)
          for d in o.description_alts),
    ]}


def build_llm_index(m: Master) -> str:
    """Compact, deterministic JSON index sent to the selection model. No profile data."""
    cx = alt_conflicts(m)
    idx = {
        "summaries": [_index_item(s, cx) for s in m.summaries],
        "jobs": [
            {
                "id": j.id,
                "company": j.company,
                "title": j.title.en,
                "start": j.start,
                "end": j.end or "present",
                **_descriptions(j, cx),
                "bullets": [_index_item(b, cx) for b in j.bullets],
            }
            for j in m.jobs
        ],
        "projects": [
            {
                "id": p.id,
                "name": text_of(p.name),
                "tags": p.tags,
                **_descriptions(p, cx),
                "bullets": [_index_item(b, cx) for b in p.bullets],
            }
            for p in m.projects
        ],
        # Education is always shown; only its optional detail lines are selectable.
        "education_details": [
            {"id": e.id, "degree": e.degree.en, "bullets": [_index_item(b, cx) for b in e.bullets]}
            for e in m.education if e.bullets
        ],
        "skill_categories": [
            {
                "id": c.id,
                "tier": tier,
                "category": c.category.en,
                "items": [skill_item_text(i, "en") for i in c.items],
                **_never_with(c.id, cx),
            }
            for tier, c in _iter_skill_categories(m)
        ],
        "limits": {
            "min_bullets_per_job": m.limits.min_bullets_per_job,
            "max_bullets_per_job": m.limits.max_bullets_per_job,
            "max_bullets_per_project": m.limits.max_bullets_per_project,
            "max_total_bullets": m.limits.max_total_bullets,
            "max_projects": m.limits.max_projects,
        },
    }
    return json.dumps(idx, ensure_ascii=False, separators=(",", ":"))
