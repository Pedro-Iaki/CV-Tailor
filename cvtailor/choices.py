"""Choice format for the selection call: alternatives become single choices.

The plain schema (llm.build_selection_schema) lists every id as an enum and leaves the
"never together" rules to the validator. Small local models break those rules often:
they pick two versions of one bullet, or mix two skill layouts. This format makes
most of those mistakes impossible to express:

- A group of bullet versions (bullets that are all alternatives of each other) is
  one slot. The model lists the slot id (the group's first id) under "bullets" and
  picks the wording in "bullet_versions".
- Skill categories that conflict are offered as whole layouts: "skills" is one of
  several object shapes (JSON schema anyOf), each allowing only the categories of one
  compatible layout. Grammar-constrained decoding (Ollama, llama.cpp) enforces it.
- "key_requirements" comes first: a few words per requirement, a cheap scratchpad
  for models that think little or not at all. It is dropped after parsing.

decode() turns the answer back into a plain selection dict, a fixed mapping that
neither adds nor removes ids. The result is validated like any other selection.
Used only for local servers; the Anthropic path keeps the plain format.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from typing import Any

from .master import alt_conflicts, build_llm_index, item_index
from .schema import TIERS, Master, skill_item_text

MAX_LAYOUTS = 24  # above this, skills fall back to the plain per-tier arrays
_MAX_COMPONENT = 12  # skill conflict groups larger than this are not enumerated

SCRATCHPAD = {"type": "array", "items": {"type": "string"}, "maxItems": 5}


@dataclass
class Choices:
    groups: dict[str, list[str]]  # slot id -> bullet ids (all mutually exclusive), slot id first
    layouts: list[dict[str, list[str]]]  # each: tier -> allowed category ids, master order


def analyse(m: Master) -> Choices:
    cx = alt_conflicts(m)
    idx = item_index(m)
    order = {id_: i for i, id_ in enumerate(idx)}

    # Bullet groups: connected bullets of one owner in which every pair conflicts.
    # Approved merges conflict with each source, but the sources may still go
    # together, so those groups are not cliques and stay with the validator.
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for id_, it in idx.items():
        if it.kind != "bullet" or id_ in seen or not cx.get(id_):
            continue
        comp, todo = set(), [id_]
        while todo:
            x = todo.pop()
            if x in comp:
                continue
            comp.add(x)
            todo += [y for y in cx.get(x, ()) if idx[y].kind == "bullet" and idx[y].parent == it.parent]
        seen |= comp
        members = sorted(comp, key=order.get)
        if len(members) > 1 and all(b in cx[a] for a, b in itertools.combinations(members, 2)):
            groups[members[0]] = members

    # Skill layouts: per group of conflicting categories, every maximal set of
    # categories that may go together; layouts are the combinations across groups.
    cats = [id_ for id_, it in idx.items() if it.kind == "skill_category"]
    free = [c for c in cats if not cx.get(c)]
    comps: list[list[str]] = []
    seen = set()
    for c in cats:
        if c in seen or not cx.get(c):
            continue
        comp, todo = set(), [c]
        while todo:
            x = todo.pop()
            if x not in comp:
                comp.add(x)
                todo += [y for y in cx[x] if idx[y].kind == "skill_category"]
        seen |= comp
        comps.append(sorted(comp, key=order.get))
    options: list[list[list[str]]] = []
    for comp in comps:
        if len(comp) > _MAX_COMPONENT:
            return Choices(groups, [])
        sets = [set(s) for n in range(len(comp), 0, -1) for s in itertools.combinations(comp, n)
                if not any(b in cx[a] for a, b in itertools.combinations(s, 2))]
        maximal = [s for s in sets if not any(s < t for t in sets)]
        options.append([sorted(s, key=order.get) for s in maximal])
    combos = list(itertools.product(*options)) if comps else []
    if len(combos) > MAX_LAYOUTS:
        return Choices(groups, [])
    layouts = []
    for combo in combos:
        allowed = set(free).union(*combo)
        layouts.append({t: [c.id for c in m.skills.tier(t) if c.id in allowed] for t in TIERS})
    return Choices(groups, layouts)


def layout_id(i: int) -> str:
    return f"layout-{i + 1}"


# ---------------------------------------------------------------- index

def build_index(m: Master, ch: Choices | None = None) -> str:
    """The plain index, with bullet groups shown as one item with "versions" and the
    skill layouts listed. Limits come first."""
    ch = ch or analyse(m)
    idx = json.loads(build_llm_index(m))
    member_of = {b: slot for slot, ms in ch.groups.items() for b in ms}

    def regroup(bullets: list[dict]) -> list[dict]:
        by_id = {b["id"]: b for b in bullets}
        out = []
        for b in bullets:
            slot = member_of.get(b["id"])
            if slot is None:
                out.append(b)
            elif slot == b["id"]:
                versions = []
                for v in ch.groups[slot]:
                    v = dict(by_id[v])
                    outside = [x for x in v.pop("never_with", []) if x not in ch.groups[slot]]
                    versions.append(v | ({"never_with": outside} if outside else {}))
                out.append({"id": slot, "versions": versions})
        return out

    for key in ("jobs", "projects", "education_details"):
        for o in idx[key]:
            o["bullets"] = regroup(o["bullets"])
    out: dict[str, Any] = {"limits": idx.pop("limits"), **idx}
    if ch.layouts:
        for c in out["skill_categories"]:
            c.pop("never_with", None)
        out["skill_layouts"] = [{"id": layout_id(i), **{t: ids for t, ids in lay.items() if ids}}
                                for i, lay in enumerate(ch.layouts)]
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- schema

MAX_RANK_BRANCHES = 64  # above this, count limits are left to validation


def _ranks(keys: list[str], top: int, at_least_one: bool, min_count: int = 0) -> dict[str, Any]:
    """An object giving every key a rank, 0 (left out) to `top`. Object keys can't
    repeat, so nothing can be listed twice.

    Counts go into anyOf branches: each branch requires a set of keys to be ranked
    (one key with at_least_one, `min_count` keys when that is larger); when there are
    more keys than `top`, each branch also forces enough keys to 0 that at most `top`
    can be ranked. Branch counts are capped (MAX_RANK_BRANCHES); beyond that the
    validator checks the counts."""
    def obj(picks: tuple[str, ...] = (), zeros: tuple[str, ...] = ()) -> dict[str, Any]:
        def enum(k: str) -> list[int]:
            return [0] if k in zeros else list(range(1 if k in picks else 0, top + 1))
        return {
            "type": "object",
            "properties": {k: {"type": "integer", "enum": enum(k)} for k in keys},
            "required": list(keys),
            "additionalProperties": False,
        }
    need = min(max(min_count, 1 if at_least_one else 0), top, len(keys))
    need_zero = max(0, len(keys) - top)
    zero_sets = list(itertools.combinations(keys, need_zero)) if need_zero else [()]
    pick_sets = list(itertools.combinations(keys, need)) if need else [()]
    branches = [(p, z) for z in zero_sets for p in pick_sets if not set(p) & set(z)]
    if len(branches) > MAX_RANK_BRANCHES:  # too many shapes: keep only the minimum
        branches = [(p, ()) for p in pick_sets]
        if len(branches) > MAX_RANK_BRANCHES:  # even that is too many: one key each
            branches = [((k,), ()) for k in keys] if need else [((), ())]
    if len(branches) == 1:
        return obj(*branches[0])
    return {"anyOf": [obj(p, z) for p, z in branches]}


def _force_zero(rank_schema: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """A copy of a _ranks schema in which `keys` can only be 0. Branches that then
    can't be satisfied (they required one of those keys) are dropped."""
    def fix(obj: dict[str, Any]) -> dict[str, Any] | None:
        props = dict(obj["properties"])
        for k in keys & props.keys():
            if 0 not in props[k]["enum"]:
                return None  # this branch requires k to be ranked
            props[k] = {"type": "integer", "enum": [0]}
        return {**obj, "properties": props}
    if not keys:
        return rank_schema
    if "anyOf" in rank_schema:
        branches = [b for b in (fix(o) for o in rank_schema["anyOf"]) if b is not None]
        return branches[0] if len(branches) == 1 else {"anyOf": branches}
    return fix(rank_schema) or rank_schema


def build_schema(m: Master, plain: dict[str, Any], count_limits: bool,
                 ch: Choices | None = None) -> dict[str, Any]:
    """Transforms the plain schema (llm.build_selection_schema) into the choice format."""
    ch = ch or analyse(m)
    s = copy.deepcopy(plain)
    props = s["properties"]
    hidden = {b for ms in ch.groups.values() for b in ms[1:]}
    owners = props["bullets"]["properties"]
    for owner, p in owners.items():
        if "enum" not in p["items"]:
            continue
        ids = [b for b in p["items"]["enum"] if b not in hidden]
        top = min(p.get("maxItems", len(ids)), len(ids))
        owners[owner] = _ranks(ids, top, at_least_one=p.get("minItems", 0) >= 1)

    new: dict[str, Any] = {"key_requirements": SCRATCHPAD, "summary_id": props.pop("summary_id")}
    if "headline_id" in props:
        new["headline_id"] = props.pop("headline_id")
    if ch.groups:
        new["bullet_versions"] = {
            "type": "object",
            "properties": {slot: {"type": "string", "enum": ms} for slot, ms in ch.groups.items()},
            "required": list(ch.groups),
            "additionalProperties": False,
        }
    # Projects: rank, bullets and description together. A hidden project (rank 0) can
    # only have 0 for every bullet and the default description, so bullets of a
    # project that isn't shown can't be expressed.
    props.pop("project_ids")
    descs = (props.get("descriptions") or {}).get("properties", {})
    max_p = min(m.limits.max_projects, len(m.projects))
    cx = alt_conflicts(m)
    projects: dict[str, Any] = {}
    for p in m.projects:
        bullets = owners.pop(p.id, None)
        desc = descs.pop(p.id, None)
        rank_on = {"rank": {"type": "integer", "enum": list(range(1, max_p + 1))}}
        off: dict[str, Any] = {"rank": {"type": "integer", "enum": [0]}}
        if bullets is not None:
            keys = bullets["required"] if "required" in bullets else bullets["anyOf"][0]["required"]
            off["bullets"] = {"type": "object", "properties": {k: {"type": "integer", "enum": [0]} for k in keys},
                              "required": keys, "additionalProperties": False}
        shown_shapes: list[dict[str, Any]] = []
        if desc is None:
            shown_shapes.append({**rank_on, **({"bullets": bullets} if bullets is not None else {})})
        else:
            off["description"] = {"type": "string", "enum": ["default"]}
            # One shape per description option: bullets it already covers are forced to 0.
            for opt in desc["enum"]:
                shape: dict[str, Any] = dict(rank_on)
                if bullets is not None:
                    shape["bullets"] = _force_zero(bullets, cx.get(opt, set()))
                shape["description"] = {"type": "string", "enum": [opt]}
                shown_shapes.append(shape)
        shapes = [off, *shown_shapes] if max_p else [off]
        projects[p.id] = {"anyOf": [{"type": "object", "properties": sh, "required": list(sh),
                                     "additionalProperties": False} for sh in shapes]}
    props["bullets"]["required"] = [k for k in props["bullets"]["required"] if k in owners]
    if "descriptions" in props:
        props["descriptions"]["required"] = list(descs)
        if not descs:
            props.pop("descriptions")
    if projects:
        new["projects"] = {"type": "object", "properties": projects, "required": list(projects),
                           "additionalProperties": False}
    new["bullets"] = props.pop("bullets")
    skills_key = "skill_category_order"
    if ch.layouts:
        props.pop("skill_category_order")
        skills_key = "skills"
        cats = {c.id: c for t in TIERS for c in m.skills.tier(t)}
        branches = []
        for i, lay in enumerate(ch.layouts):
            tiers = {}
            for t in TIERS:
                ids = lay[t]
                arr: dict[str, Any] = ({"type": "array", "items": {"type": "string", "enum": ids}}
                                       if ids else {"type": "array", "items": {"type": "string"}})
                if count_limits:  # required categories of this layout set the minimum
                    arr = {**arr, "minItems": sum(1 for c in ids if cats[c].required), "maxItems": len(ids)}
                tiers[t] = arr
            # Per category: a rank for each of its own items (exact English text), so
            # items can be dropped (0) or reordered but never repeated or moved.
            item_props = {}
            for cid in (c for t in TIERS for c in lay[t]):
                own = [skill_item_text(x, "en") for x in cats[cid].items]
                item_props[cid] = _ranks(own, len(own), at_least_one=False,
                                         min_count=cats[cid].min_items or 0)
            branches.append({
                "type": "object",
                "properties": {"layout": {"type": "string", "enum": [layout_id(i)]}, **tiers,
                               "items": {"type": "object", "properties": item_props,
                                         "required": list(item_props), "additionalProperties": False}},
                "required": ["layout", *TIERS, "items"],
                "additionalProperties": False,
            })
        new["skills"] = {"anyOf": branches}
    else:
        new[skills_key] = props.pop("skill_category_order")
    new.update(props)  # descriptions (if any), unmatched_requirements, suggestions
    s["properties"] = new
    s["required"] = list(new)  # everything, in the order the model should write it
    return s


# ---------------------------------------------------------------- decode

def decode(data: dict[str, Any], master: Master | None = None) -> dict[str, Any]:
    """Choice-format answer -> plain selection dict. Never adds or drops an id; it only
    swaps each slot id for the version picked for it.

    Skill items: the schema asks for items of every category in the layout, so items
    of categories left out of the tiers are dropped, and (given the master) a list
    equal to the full master order is dropped as well, since that is the default."""
    d = dict(data)
    d.pop("key_requirements", None)
    versions = d.pop("bullet_versions", None) or {}
    projects = d.pop("projects", None)
    if isinstance(projects, dict):  # {pid: {rank, bullets?, description?}}
        entries = {pid: e for pid, e in projects.items() if isinstance(e, dict)}
        d["project_ids"] = _ranked({pid: e.get("rank") for pid, e in entries.items()})
        bullets = dict(d.get("bullets") or {})
        descriptions = dict(d.get("descriptions") or {})
        for pid in d["project_ids"]:
            if "bullets" in entries[pid]:
                bullets[pid] = entries[pid]["bullets"]
            if "description" in entries[pid]:
                descriptions[pid] = entries[pid]["description"]
        d["bullets"], d["descriptions"] = bullets, descriptions
    if isinstance(d.get("bullets"), dict) and isinstance(versions, dict):
        d["bullets"] = {o: [versions.get(b, b) if isinstance(b, str) else b for b in _ranked(ids)]
                        for o, ids in d["bullets"].items()}
    if "skills" in d:
        skills = d.pop("skills") or {}
        if not isinstance(skills, dict):
            d["skill_category_order"] = skills
            return d
        order = {t: skills.get(t) or [] for t in TIERS}
        d["skill_category_order"] = order
        items = skills.get("items")
        if isinstance(items, dict):
            chosen = {c for t in TIERS for c in order[t]}
            full = ({c.id: [skill_item_text(x, "en") for x in c.items] for t in TIERS for c in master.skills.tier(t)}
                    if master is not None else {})
            ranked = {cid: _ranked(v) for cid, v in items.items()}
            d["skill_items"] = {cid: texts for cid, texts in ranked.items()
                                if cid in chosen and texts != full.get(cid)}
    return d


def _ranked(value: Any) -> list[Any]:
    """{bullet: rank} -> bullets with rank > 0, by rank. Equal ranks keep the order the
    answer listed them in (the schema's order, which is the master order). A plain
    list passes through unchanged."""
    if isinstance(value, dict):
        ranked = [(r, i, b) for i, (b, r) in enumerate(value.items())
                  if isinstance(r, int) and not isinstance(r, bool) and r > 0]
        return [b for _, _, b in sorted(ranked)]
    return list(value or [])


# ---------------------------------------------------------------- prompt

def system_prompt(base: str) -> str:
    """The plain selection prompt with the list and alternatives rules rewritten for
    this format."""
    for old, new in _REWRITES:
        if old not in base:  # pragma: no cover - guarded by a test
            raise RuntimeError(f"selection prompt changed; update choices._REWRITES: {old[:40]!r}")
        base = base.replace(old, new)
    return base


_OLD_RULE = """\
- Some items are alternative versions of each other: an item's "never_with" lists
  the ids it must never be selected together with (anywhere in the selection). Pick
  the version that fits the job best. For skills, alternatives are whole layouts:
  either the categories on one side or the other, never a mix.
"""

_NEW_RULE = """\
- Under "bullets", every job, project and education entry gives EACH of its bullets a
  number: 0 leaves it out, 1 puts it first, 2 second, and so on. Use each number at
  most once per entry. Every job needs at least one bullet.
- A bullet with "versions" is ONE bullet written in several ways. To use it, number
  its "id" under "bullets" like any other bullet, and pick the wording that fits the
  job best under "bullet_versions" (every such bullet needs a pick there, used or not).
- An item may list "never_with": ids it must never be selected together with.
- Skills: choose ONE entry of "skill_layouts" and put its id in skills.layout. Then
  list only that layout's categories, each in its own tier, most relevant first.
  You may leave out categories that do not help. Pick the layout whose categories
  cover the job's requirements best. A layout with a "working" or "familiar" tier
  lists some skills at a lower experience level: prefer it when the job asks for
  skills that only appear there, rather than leaving them out.
- Under skills.items, every category of the chosen layout gives EACH of its skills a
  number the same way: 0 leaves it out, 1 puts it first, and so on, most relevant
  to the job first. Keep at least one skill per category you list, and at least
  "min_items" where a category has it. A "required" category is always listed.
"""

_OLD_INTRO = "Hard rules:\n"
_NEW_INTRO = (
    "First, under \"key_requirements\", write the 3 to 5 most important requirements of the\n"
    "job in a few words each. Base every choice on them.\n\nHard rules:\n"
)

# (old, new) pairs applied to llm.SELECTION_SYSTEM_PROMPT, in order.
_REWRITES = [
    (_OLD_RULE, _NEW_RULE),
    (_OLD_INTRO, _NEW_INTRO),
    ("""\
- Projects are optional. Pick them in "project_ids". A picked project's bullets
  go under "bullets" keyed by the project id (from none up to the project
  maximum). A project you do not pick gets an empty list.
""", """\
- Projects are optional. Under "projects", give each project a rank: 0 leaves it
  out (its bullets then all get 0 and its description stays "default"), 1 shows it
  first, 2 second. A shown project numbers its own bullets and picks its
  description there.
- Count before answering: respect the maximum bullets per job and per project, and
  the total across jobs, projects and education.
"""),
    ('needs one option id under\n  "descriptions" (keyed by the job or project id)',
     'needs one option id: a job\'s under\n  "descriptions" (keyed by the job id), a project\'s in its "projects" entry'),
    ("education id; otherwise give it an empty list.",
     "education id; otherwise give them 0."),
]
