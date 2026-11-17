import asyncio
import copy

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from cvtailor import mcp_server as srv
from tests.conftest import FakeClient


def answer(selection_dict):
    d = copy.deepcopy(selection_dict)
    d.update({"unmatched_requirements": ["dbt"], "suggestions": ["add a kafka bullet"]})
    return d


@pytest.fixture
def ctx(make_ctx, selection_dict, monkeypatch):
    c = make_ctx(client=FakeClient(selections=[answer(selection_dict)] * 3))
    monkeypatch.setattr(srv, "make_context", lambda: c)
    return c


def test_server_registers_all_tools():
    tools = asyncio.run(srv.build_server().list_tools())
    names = {t.name for t in tools}
    assert names == {fn.__name__ for fn, _ in srv.TOOLS}
    tailor = next(t for t in tools if t.name == "tailor_cv")
    assert tailor.input_schema["required"] == ["job_description"]
    assert next(t for t in tools if t.name == "get_stats").annotations.read_only_hint is True


def test_tailor_then_track(ctx, job_text):
    out = srv.tailor_cv(job_text)
    assert out["app_id"] == 1 and out["route"] == "generated"
    assert out["suggestions"] == ["add a kafka bullet"] and out["unmatched_requirements"] == ["dbt"]
    assert any(l.startswith("#1 -> ") for l in out["log"])
    assert srv.list_applications()[0]["company"] == "Acme Pagamentos"
    srv.set_status([1], "interview", note="call with hiring manager")
    srv.set_status([1], "rejected")
    detail = srv.get_application(1)
    assert detail["status"] == "rejected"
    assert [h["status"] for h in detail["status_history"]] == ["applied", "interview", "rejected"]
    assert any("Built Databricks pipelines" in l for l in detail["selection_text"])
    stats = srv.get_stats(min_age=0)
    assert stats["overall"]["interview_plus"] == 1 and stats["overall"]["low_sample"]


def test_same_company_needs_confirmation(ctx, job_text):
    srv.tailor_cv(job_text)
    with pytest.raises(ToolError, match="confirm_same_company=true"):
        srv.tailor_cv(job_text + "\nHybrid.")
    with pytest.raises(ToolError, match="allow_duplicate=true"):
        srv.tailor_cv(job_text)
    assert srv.tailor_cv(job_text + "\nHybrid.", confirm_same_company=True)["app_id"] == 2


def test_meta_failure_hint(make_ctx, monkeypatch, job_text):
    c = make_ctx(client=FakeClient(meta_error=RuntimeError("down")))
    monkeypatch.setattr(srv, "make_context", lambda: c)
    with pytest.raises(ToolError, match="Pass company, role and lang"):
        srv.tailor_cv(job_text)


def test_update_selection_validates(ctx, job_text, selection_dict):
    srv.tailor_cv(job_text)
    bad = copy.deepcopy(selection_dict)
    del bad["bullets"]["job-intern"]
    with pytest.raises(ToolError, match="nothing changed"):
        srv.update_selection(1, bad)
    good = copy.deepcopy(selection_dict)
    good["bullets"]["job-acme"][-1] = "acme-streaming-06"
    assert srv.update_selection(1, good)["edited"] is True
    assert "acme-streaming-06" in srv.get_application(1)["selection"]["bullets"]["job-acme"]


def test_approve_render_and_variants(ctx, job_text):
    srv.tailor_cv(job_text)
    srv.approve_variant(1, "platform-v1")
    [v] = srv.list_variants()
    assert v["name"] == "platform-v1" and v["state"] == "ok" and v["applications"] == 1
    out = srv.render_application(1, "pt")
    assert out["pdf_path"].endswith("_PT.pdf")


def test_master_index_has_no_contact_data(ctx):
    idx = srv.get_master_index()
    assert idx["jobs"][0]["id"] == "job-acme"
    assert "ana.souza@example.com" not in str(idx) and "Ana Souza" not in str(idx)


def test_unknown_status_is_a_tool_error(ctx, job_text):
    srv.tailor_cv(job_text)
    with pytest.raises(ToolError, match="unknown status"):
        srv.set_status([1], "hired")
    with pytest.raises(ToolError, match="not found"):
        srv.set_status([99], "screen")


def test_error_message_reaches_the_model(ctx):
    with pytest.raises(ToolError, match="application #42 not found"):
        asyncio.run(srv.build_server().call_tool("get_application", {"app_id": 42}))
