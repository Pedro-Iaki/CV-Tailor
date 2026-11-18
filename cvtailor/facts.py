"""Approved generated text: the facts table (renderable) and unsafe items (not renderable).

Trust levels, highest first:
  master.yaml      written by the user
  facts            LLM drafts the user approved that only reword/merge existing items,
                   or unsafe items the user promoted by hand
  unsafe_items     approved drafts that make new claims; NEVER merged into the master,
                   so they cannot be selected or rendered until promoted
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Optional

from . import db
from .master import MasterError, check_master, item_index
from .schema import Master

ITEM_COLUMNS = ("id", "kind", "job_id", "origin", "based_on", "tags", "texts", "comment",
                "proposal_id", "source_application_id", "created_at")


class FactError(Exception):
    pass


_row = db.item_row_to_dict


def _insert(conn: sqlite3.Connection, table: str, item: dict[str, Any]) -> None:
    data = {k: (json.dumps(v, ensure_ascii=False) if k in db.ITEM_JSON_COLUMNS else v)
            for k, v in item.items()}
    cols = ", ".join(data)
    qs = ", ".join("?" for _ in data)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(data.values()))


# ---------------------------------------------------------------- reads

def load_facts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [_row(r) for r in conn.execute("SELECT * FROM facts ORDER BY approved_at, id")]


def load_unsafe(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [_row(r) for r in conn.execute("SELECT * FROM unsafe_items ORDER BY created_at, id")]


def get_unsafe(conn: sqlite3.Connection, item_id: str) -> Optional[dict[str, Any]]:
    r = conn.execute("SELECT * FROM unsafe_items WHERE id = ?", (item_id,)).fetchone()
    return _row(r) if r else None


# ---------------------------------------------------------------- merge into the master

def merge_facts(master: Master, facts: list[dict[str, Any]]) -> tuple[Master, list[str]]:
    """Return the master with facts added as summaries/bullets, plus warnings for facts
    that no longer fit (e.g. their job was removed from master.yaml)."""
    if not facts:
        return master, []
    data = master.model_dump()
    jobs = {e["id"]: e for e in [*data["jobs"], *data["projects"]]}  # bullet owners
    warnings: list[str] = []
    for f in facts:
        item = {"id": f["id"], "tags": f["tags"], **f["texts"],
                "origin": f["origin"], "based_on": f["based_on"]}
        if f["kind"] == "summary":
            data["summaries"].append(item)
        elif f["job_id"] in jobs:
            jobs[f["job_id"]]["bullets"].append(item)
        else:
            warnings.append(f"fact {f['id']}: job or project '{f['job_id']}' is not in the master any more; ignored")
    merged = Master.model_validate(data)
    errors = check_master(merged)
    if errors:
        raise MasterError([f"approved facts conflict with the master: {e}" for e in errors]
                          + ["fix master.yaml or remove the fact with `apply.py facts remove <id>`"])
    return merged, warnings


# ---------------------------------------------------------------- writes

def next_id(conn: sqlite3.Connection, master: Master, kind: str, job_id: Optional[str]) -> str:
    stem = "summary" if kind == "summary" else re.sub(r"^(job|proj)-", "", job_id or "item")
    taken = set(item_index(master))
    taken |= {r[0] for r in conn.execute("SELECT id FROM facts UNION SELECT id FROM unsafe_items")}
    n = 1
    while f"gen-{stem}-{n:02d}" in taken:
        n += 1
    return f"gen-{stem}-{n:02d}"


def add_fact(conn: sqlite3.Connection, item: dict[str, Any], approved_via: str) -> None:
    now = db.now_iso()
    _insert(conn, "facts", {**{k: item.get(k) for k in ITEM_COLUMNS},
                            "created_at": item.get("created_at") or now,
                            "approved_at": now, "approved_via": approved_via})


def add_unsafe(conn: sqlite3.Connection, item: dict[str, Any], rationale: Optional[str]) -> None:
    _insert(conn, "unsafe_items", {**{k: item.get(k) for k in ITEM_COLUMNS},
                                   "created_at": item.get("created_at") or db.now_iso(),
                                   "rationale": rationale})


def promote(conn: sqlite3.Connection, item_id: str, *, texts: Optional[dict[str, str]] = None,
            tags: Optional[list[str]] = None, langs: Optional[list[str]] = None) -> dict[str, Any]:
    """Move an unsafe item to the facts table (inside the caller's transaction).

    Only the CLI calls this, after the user confirms in the terminal. It is deliberately
    not exposed as an MCP tool, so a model can never promote its own claims."""
    item = get_unsafe(conn, item_id)
    if item is None:
        raise FactError(f"unsafe item '{item_id}' not found")
    if texts is not None:
        item["texts"] = {k: str(v) for k, v in texts.items()}
    if tags is not None:
        item["tags"] = tags
    empty = [c for c in (langs or list(item["texts"])) if not str(item["texts"].get(c) or "").strip()]
    if empty or not item["texts"]:
        raise FactError(f"the text in {', '.join(empty) or 'every language'} must not be empty")
    add_fact(conn, item, approved_via="promote")
    conn.execute("DELETE FROM unsafe_items WHERE id = ?", (item_id,))
    return item


def drop_unsafe(conn: sqlite3.Connection, item_id: str) -> None:
    if conn.execute("DELETE FROM unsafe_items WHERE id = ?", (item_id,)).rowcount == 0:
        raise FactError(f"unsafe item '{item_id}' not found")


def remove_fact(conn: sqlite3.Connection, item_id: str) -> None:
    if conn.execute("DELETE FROM facts WHERE id = ?", (item_id,)).rowcount == 0:
        raise FactError(f"fact '{item_id}' not found")
