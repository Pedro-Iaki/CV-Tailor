"""CLI wiring for the database commands (no LLM, no PDF)."""

from datetime import datetime, timedelta

import pytest
from typer.testing import CliRunner

import apply
from cvtailor import db
from tests.conftest import EXAMPLE_MASTER, FakeClient


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CV_MASTER", str(EXAMPLE_MASTER))
    monkeypatch.setenv("CV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CV_OUTPUT_DIR", str(tmp_path / "out"))
    conn = db.connect(tmp_path / "data" / "applications.db")
    for i, days in enumerate([40, 35, 2]):
        created = (datetime.now().astimezone() - timedelta(days=days)).isoformat(timespec="seconds")
        db.insert_application(conn, dict(
            created_at=created, company=f"Co{i}", role_title="DE", company_key=f"co{i}|de", lang="en",
            job_description="jd", job_hash=f"h{i}", job_tags=[], route="generated", selection={},
            pdf_path="x.pdf"))
    conn.close()
    return tmp_path


def test_status_multiple_ids_and_list(env):
    r = CliRunner().invoke(apply.app, ["status", "1", "2", "interview", "--note", "call"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(apply.app, ["status", "1", "rejected"])
    assert r.exit_code == 0
    out = CliRunner().invoke(apply.app, ["list"]).output
    assert "rejected" in out and "interview" in out
    stats = CliRunner().invoke(apply.app, ["stats"]).output
    assert "counted: 2 | pending (applied < 14 days): 1" in stats
    assert "100% (2/2)" in stats  # interview+ still counts #1 after rejection


def test_status_rejects_unknown(env):
    r = CliRunner().invoke(apply.app, ["status", "1", "hired"])
    assert r.exit_code != 0
    r = CliRunner().invoke(apply.app, ["status", "99", "screen"])
    assert r.exit_code != 0


def test_mark_ghosted_and_discard(env):
    r = CliRunner().invoke(apply.app, ["mark-ghosted", "--older-than", "30", "--yes"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(apply.app, ["discard", "3"])
    assert r.exit_code == 0
    conn = db.connect(env / "data" / "applications.db")
    assert [a["status"] for a in sorted(db.list_applications(conn), key=lambda a: a["id"])] == \
        ["ghosted", "ghosted", "discarded"]


def test_batch_reports_ok_skipped_and_failed(tmp_path, monkeypatch, make_ctx, selection_dict, job_text):
    sel = dict(selection_dict, unmatched_requirements=[], suggestions=[])
    ctx = make_ctx(client=FakeClient(selections=[sel]))
    monkeypatch.setattr(apply, "_ctx", lambda: ctx)
    first, dup = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text(job_text, encoding="utf-8")
    dup.write_text(job_text, encoding="utf-8")  # same posting -> skipped, not prompted
    r = CliRunner().invoke(apply.app, ["batch", str(first), str(dup), str(tmp_path / "missing.txt")])
    assert r.exit_code == 0, r.output
    names = ("a.txt", "b.txt", "missing.txt")
    rows = {ln.split()[0]: ln for ln in r.output.splitlines() if ln.split() and ln.split()[0] in names}
    assert rows["a.txt"].rstrip().endswith("ok")
    assert "skipped: duplicate of #1" in rows["b.txt"]
    assert "failed: file not found" in rows["missing.txt"]
    assert len(db.list_applications(ctx.db())) == 1


def test_validate_master(env):
    r = CliRunner().invoke(apply.app, ["validate-master"])
    assert r.exit_code == 0 and "master ok" in r.output
