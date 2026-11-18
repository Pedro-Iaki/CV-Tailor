"""Approved variants: save, load, content hashes and state (ok / drifted / stale)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .master import item_index
from .schema import TIERS, Master, Selection, Variant, skill_item_text, text_of
from .validate import validate_selection

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class VariantError(Exception):
    pass


@dataclass
class LoadedVariant:
    name: str
    path: Path
    state: str  # ok | drifted | stale
    variant: Optional[Variant] = None
    reasons: list[str] = field(default_factory=list)
    changed_ids: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.state in ("ok", "drifted") and self.variant is not None

    def state_text(self) -> str:
        if self.state == "stale":
            return "stale: " + "; ".join(self.reasons)
        if self.state == "drifted":
            return "drifted: " + ", ".join(self.changed_ids)
        return "ok"


def selection_item_ids(sel: Selection) -> list[str]:
    ids = [sel.summary_id]
    for b in sel.bullets.values():
        ids.extend(b)
    ids.extend(sel.project_ids)
    ids.extend(sel.descriptions.values())
    for t in TIERS:
        ids.extend(getattr(sel.skill_category_order, t))
    return ids


def item_hash(master: Master, id_: str) -> str:
    it = item_index(master)[id_]
    o = it.obj
    langs = master.langs  # en+pt hashes are the same as before languages were configurable
    if it.kind == "skill_category":
        parts = [o.category.get(lang) for lang in langs]
        parts += ["|".join(skill_item_text(i, lang) for lang in langs) for i in o.items]
    elif it.kind == "project":  # its bullets are hashed on their own
        parts = [text_of(o.name, lang) + "|" + text_of(o.description, lang) for lang in langs]
        parts += ["|".join([*(text_of(link.label, lang) for lang in langs), link.url]) for link in o.links]
    elif it.kind == "summary" and o.headline is not None:
        parts = [*(o.get(lang) for lang in langs), *(text_of(o.headline, lang) for lang in langs)]
    else:
        parts = [o.get(lang) for lang in langs]
    return "sha256:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def compute_hashes(sel: Selection, master: Master) -> dict[str, str]:
    return {i: item_hash(master, i) for i in selection_item_ids(sel)}


def check_variant(v: Variant, master: Master) -> tuple[str, list[str], list[str]]:
    errors = validate_selection(v.selection, master)
    if errors:
        return "stale", errors, []
    current = compute_hashes(v.selection, master)
    changed = sorted(i for i, h in current.items() if v.item_hashes.get(i) != h)
    return ("drifted" if changed else "ok"), [], changed


def load_variant_file(path: Path, master: Master) -> LoadedVariant:
    name = path.stem
    try:
        v = Variant.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        return LoadedVariant(name, path, "stale", None, [f"unreadable variant file: {str(e).splitlines()[0]}"])
    state, reasons, changed = check_variant(v, master)
    return LoadedVariant(v.name, path, state, v, reasons, changed)


def load_variants(variants_dir: Path, master: Master) -> list[LoadedVariant]:
    if not variants_dir.exists():
        return []
    return [load_variant_file(p, master) for p in sorted(variants_dir.glob("*.json"))]


def get_variant(variants_dir: Path, name: str, master: Master) -> Optional[LoadedVariant]:
    p = variants_dir / f"{name}.json"
    return load_variant_file(p, master) if p.exists() else None


def save_variant(variants_dir: Path, *, name: str, selection: Selection, job_tags: list[str],
                 master: Master, source_application_id: Optional[int],
                 replace: bool = False) -> Path:
    if not NAME_RE.match(name):
        raise VariantError("variant name may only contain letters, digits, '.', '_' and '-'")
    errors = validate_selection(selection, master)
    if errors:
        raise VariantError("selection is not valid against the current master:\n  - " + "\n  - ".join(errors))
    variants_dir.mkdir(parents=True, exist_ok=True)
    path = variants_dir / f"{name}.json"
    if path.exists() and not replace:
        raise VariantError(f"variant '{name}' already exists (use --replace to overwrite)")
    v = Variant(
        name=name,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source_application_id=source_application_id,
        job_tags=sorted(job_tags),
        selection=selection,
        item_hashes=compute_hashes(selection, master),
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(v.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path
