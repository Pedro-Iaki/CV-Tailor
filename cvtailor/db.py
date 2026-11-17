"""SQLite access with numbered migrations (PRAGMA user_version)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

STATUSES = ("applied", "screen", "interview", "offer", "rejected", "ghosted", "discarded")
STAGE_RANK = {"applied": 0, "screen": 1, "interview": 2, "offer": 3}

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE applications (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      company TEXT NOT NULL,
      role_title TEXT NOT NULL,
      company_key TEXT NOT NULL,
      job_url TEXT,
      lang TEXT NOT NULL,
      job_description TEXT NOT NULL,
      job_hash TEXT NOT NULL,
      job_tags TEXT NOT NULL,
      route TEXT NOT NULL,
      route_score REAL,
      route_candidates TEXT,
      variant_name TEXT,
      selection TEXT NOT NULL,
      llm_selection TEXT,
      trimmed TEXT,
      edited INTEGER NOT NULL DEFAULT 0,
      unmatched_requirements TEXT,
      suggestions TEXT,
      pdf_path TEXT NOT NULL,
      model TEXT,
      input_tokens INTEGER,
      cache_read_tokens INTEGER,
      cache_write_tokens INTEGER,
      output_tokens INTEGER,
      status TEXT NOT NULL DEFAULT 'applied',
      furthest_stage INTEGER NOT NULL DEFAULT 0,
      status_updated_at TEXT,
      notes TEXT
    );
    CREATE TABLE status_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      application_id INTEGER NOT NULL REFERENCES applications(id),
      status TEXT NOT NULL,
      changed_at TEXT NOT NULL,
      note TEXT
    );
    CREATE INDEX idx_app_job_hash ON applications(job_hash);
    CREATE INDEX idx_app_company_key ON applications(company_key);
    """,
    # 2: opt-in generated text. Proposals are reviewed one by one; approved rephrasings
    # and merges become facts (renderable), approved new claims become unsafe items
    # (never renderable until promoted by hand).
    """
    CREATE TABLE proposals (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      application_id INTEGER NOT NULL REFERENCES applications(id),
      parent_id INTEGER REFERENCES proposals(id),
      depth INTEGER NOT NULL DEFAULT 0,
      kind TEXT NOT NULL,
      job_id TEXT,
      origin TEXT NOT NULL,
      destination TEXT NOT NULL,
      based_on TEXT NOT NULL,
      replaces TEXT NOT NULL,
      tags TEXT NOT NULL,
      en TEXT NOT NULL,
      pt TEXT NOT NULL,
      rationale TEXT,
      notes TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      reply TEXT,
      comment TEXT,
      stored_id TEXT,
      applied INTEGER NOT NULL DEFAULT 0,
      model TEXT,
      created_at TEXT NOT NULL,
      reviewed_at TEXT
    );
    CREATE INDEX idx_proposals_app ON proposals(application_id);
    CREATE TABLE facts (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      job_id TEXT,
      origin TEXT NOT NULL,
      based_on TEXT NOT NULL,
      tags TEXT NOT NULL,
      en TEXT NOT NULL,
      pt TEXT NOT NULL,
      comment TEXT,
      proposal_id INTEGER,
      source_application_id INTEGER,
      created_at TEXT NOT NULL,
      approved_at TEXT NOT NULL,
      approved_via TEXT NOT NULL
    );
    CREATE TABLE unsafe_items (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      job_id TEXT,
      origin TEXT NOT NULL,
      based_on TEXT NOT NULL,
      tags TEXT NOT NULL,
      en TEXT NOT NULL,
      pt TEXT NOT NULL,
      comment TEXT,
      rationale TEXT,
      proposal_id INTEGER,
      source_application_id INTEGER,
      created_at TEXT NOT NULL
    );
    ALTER TABLE applications ADD COLUMN generated_ids TEXT;
    """,
]

JSON_COLUMNS = ("job_tags", "route_candidates", "selection", "llm_selection", "trimmed",
                "unmatched_requirements", "suggestions", "generated_ids")
ITEM_JSON_COLUMNS = ("based_on", "replaces", "tags")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.astimezone()


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.execute("BEGIN")
        try:
            for stmt in [s for s in sql.split(";") if s.strip()]:
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {i}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _dump(v: Any) -> Any:
    return json.dumps(v, ensure_ascii=False) if v is not None else None


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for c in JSON_COLUMNS:
        if c in d and d[c] is not None:
            d[c] = json.loads(d[c])
    return d


def item_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """A proposals / facts / unsafe_items row with its JSON columns decoded."""
    d = dict(row)
    for c in ITEM_JSON_COLUMNS:
        if d.get(c) is not None:
            d[c] = json.loads(d[c])
    return d


# ---------------------------------------------------------------- writes

def insert_application(conn: sqlite3.Connection, fields: dict[str, Any]) -> int:
    """Insert inside the caller's transaction. JSON columns are serialized here."""
    data = {k: (_dump(v) if k in JSON_COLUMNS else v) for k, v in fields.items()}
    data.setdefault("created_at", now_iso())
    data.setdefault("status_updated_at", data["created_at"])
    cols = ", ".join(data)
    qs = ", ".join("?" for _ in data)
    cur = conn.execute(f"INSERT INTO applications ({cols}) VALUES ({qs})", list(data.values()))
    app_id = cur.lastrowid
    conn.execute(
        "INSERT INTO status_history (application_id, status, changed_at, note) VALUES (?, ?, ?, ?)",
        (app_id, data.get("status", "applied"), data["created_at"], "created"),
    )
    return app_id


def update_application(conn: sqlite3.Connection, app_id: int, fields: dict[str, Any]) -> None:
    data = {k: (_dump(v) if k in JSON_COLUMNS else v) for k, v in fields.items()}
    sets = ", ".join(f"{k} = ?" for k in data)
    conn.execute(f"UPDATE applications SET {sets} WHERE id = ?", [*data.values(), app_id])


def set_status_many(conn: sqlite3.Connection, app_ids: list[int], status: str,
                    note: Optional[str] = None) -> None:
    """Set one status on several applications in one transaction. Raises ValueError,
    with nothing changed, if any id is missing or the status is unknown."""
    missing = missing_ids(conn, app_ids)
    if missing:
        raise ValueError("not found: " + ", ".join(f"#{i}" for i in missing))
    conn.execute("BEGIN")
    try:
        for i in app_ids:
            set_status(conn, i, status, note)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def set_status(conn: sqlite3.Connection, app_id: int, status: str, note: Optional[str] = None,
               when: Optional[str] = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status '{status}' (use one of: {', '.join(STATUSES)})")
    row = conn.execute("SELECT furthest_stage FROM applications WHERE id = ?", (app_id,)).fetchone()
    if row is None:
        raise KeyError(f"application #{app_id} not found")
    when = when or now_iso()
    furthest = max(row["furthest_stage"], STAGE_RANK.get(status, 0))
    conn.execute(
        "UPDATE applications SET status = ?, furthest_stage = ?, status_updated_at = ? WHERE id = ?",
        (status, furthest, when, app_id),
    )
    conn.execute(
        "INSERT INTO status_history (application_id, status, changed_at, note) VALUES (?, ?, ?, ?)",
        (app_id, status, when, note),
    )


# ---------------------------------------------------------------- reads

def get_application(conn: sqlite3.Connection, app_id: int) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
    return row_to_dict(row) if row else None


def find_by_hash(conn: sqlite3.Connection, job_hash: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM applications WHERE job_hash = ? AND status != 'discarded' ORDER BY id LIMIT 1",
        (job_hash,),
    ).fetchone()
    return row_to_dict(row) if row else None


def find_recent_company(conn: sqlite3.Connection, key: str, days: int = 60,
                        now: Optional[datetime] = None) -> list[dict[str, Any]]:
    now = now or datetime.now().astimezone()
    cutoff = now - timedelta(days=days)
    rows = conn.execute(
        "SELECT * FROM applications WHERE company_key = ? AND status != 'discarded' ORDER BY id",
        (key,),
    ).fetchall()
    return [row_to_dict(r) for r in rows if parse_iso(r["created_at"]) >= cutoff]


def list_applications(conn: sqlite3.Connection, *, status: Optional[str] = None,
                      since: Optional[str] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM applications WHERE 1=1"
    args: list[Any] = []
    if status:
        sql += " AND status = ?"
        args.append(status)
    if since:
        sql += " AND substr(created_at, 1, 10) >= ?"
        args.append(since)
    sql += " ORDER BY id DESC"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    return [row_to_dict(r) for r in conn.execute(sql, args).fetchall()]


def status_history(conn: sqlite3.Connection, app_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM status_history WHERE application_id = ? ORDER BY id", (app_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def age_days(created_at: str, now: Optional[datetime] = None) -> int:
    now = now or datetime.now().astimezone()
    return (now - parse_iso(created_at)).days


def missing_ids(conn: sqlite3.Connection, ids: Iterable[int]) -> list[int]:
    return [i for i in ids if conn.execute("SELECT 1 FROM applications WHERE id = ?", (i,)).fetchone() is None]


def add_usage(conn: sqlite3.Connection, app_id: int, usage: Any, model: Optional[str] = None) -> None:
    """Add LLM token usage (e.g. from text generation) to an application's totals."""
    conn.execute(
        """UPDATE applications SET
             input_tokens = COALESCE(input_tokens, 0) + ?,
             cache_read_tokens = COALESCE(cache_read_tokens, 0) + ?,
             cache_write_tokens = COALESCE(cache_write_tokens, 0) + ?,
             output_tokens = COALESCE(output_tokens, 0) + ?,
             model = COALESCE(model, ?)
           WHERE id = ?""",
        (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens, usage.output_tokens,
         model, app_id),
    )
