from datetime import datetime, timedelta

import pytest

from cvtailor import db
from cvtailor.stats import compute_stats


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "a.db")


def _insert(conn, days_ago=30, **kw):
    created = (datetime.now().astimezone() - timedelta(days=days_ago)).isoformat(timespec="seconds")
    fields = dict(created_at=created, company="Acme", role_title="DE", company_key="acme|de", lang="en",
                  job_description="jd", job_hash=f"h{kw.get('job_hash', id(kw))}", job_tags=[],
                  route="generated", selection={}, pdf_path="x.pdf")
    fields.update(kw)
    return db.insert_application(conn, fields)


def test_migrations_set_user_version(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    db.migrate(conn)  # idempotent


def test_interview_then_rejected_counts_as_interview(conn):
    a = _insert(conn)
    db.set_status(conn, a, "screen")
    db.set_status(conn, a, "interview")
    db.set_status(conn, a, "rejected")
    row = db.get_application(conn, a)
    assert row["status"] == "rejected" and row["furthest_stage"] == 2
    assert [h["status"] for h in db.status_history(conn, a)] == ["applied", "screen", "interview", "rejected"]
    s = compute_stats(db.list_applications(conn))
    assert s.overall.interview == 1 and s.overall.screen == 1 and s.overall.n == 1


def test_set_status_many_is_all_or_nothing(conn):
    a = _insert(conn, days_ago=5)
    b = _insert(conn, days_ago=5)
    with pytest.raises(ValueError, match="not found: #99"):
        db.set_status_many(conn, [a, 99], "screen")
    with pytest.raises(ValueError, match="unknown status"):
        db.set_status_many(conn, [a, b], "hired")
    assert {r["status"] for r in db.list_applications(conn)} == {"applied"}
    db.set_status_many(conn, [a, b], "screen", "call")
    assert {r["furthest_stage"] for r in db.list_applications(conn)} == {1}


def test_discarded_and_pending_excluded(conn):
    _insert(conn, days_ago=30)                # counted, no reply
    _insert(conn, days_ago=3)                 # pending (young, applied)
    c = _insert(conn, days_ago=30)            # discarded
    d = _insert(conn, days_ago=3)             # young but already screened -> counted
    db.set_status(conn, c, "discarded")
    db.set_status(conn, d, "screen")
    s = compute_stats(db.list_applications(conn), min_age=14)
    assert s.overall.n == 2 and s.pending == 1 and s.discarded == 1
    assert s.overall.screen == 1
    assert s.overall.low_sample


def test_discard_can_be_undone(conn):
    a = _insert(conn)
    db.set_status(conn, a, "discarded")
    db.set_status(conn, a, "applied")
    assert db.get_application(conn, a)["status"] == "applied"


def test_groups_and_n(conn):
    for i in range(12):
        a = _insert(conn, variant_name="platform-v1" if i < 10 else None, lang="pt" if i % 2 else "en",
                    route="reused" if i < 10 else "generated")
        if i < 3:
            db.set_status(conn, a, "interview")
    s = compute_stats(db.list_applications(conn))
    by_variant = {g.key: g for g in s.by_variant}
    assert by_variant["platform-v1"].n == 10 and not by_variant["platform-v1"].low_sample
    assert by_variant["platform-v1"].interview == 3
    assert by_variant["(one-off)"].n == 2 and by_variant["(one-off)"].low_sample
    assert {g.key for g in s.by_lang} == {"en", "pt"}
    assert {g.key for g in s.by_route} == {"reused", "generated"}


def test_find_by_hash_ignores_discarded_and_recent_company(conn):
    a = _insert(conn, days_ago=10, job_hash="same")
    assert db.find_by_hash(conn, "same")["id"] == a
    assert db.find_recent_company(conn, "acme|de", 60)
    db.set_status(conn, a, "discarded")
    assert db.find_by_hash(conn, "same") is None
    _insert(conn, days_ago=90, job_hash="old", company_key="old|co")
    assert db.find_recent_company(conn, "old|co", 60) == []
