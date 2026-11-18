"""Drafting alternative wordings ("+ version" in the Master tab): opt-in, checked
against the item's sources, and never written to the master by the drafting step."""

import pytest

from cvtailor import generation
from cvtailor.generation import GenerationError, check_version, draft_versions, version_sources
from tests.conftest import FakeClient

GOOD = {"en": "Cut batch latency from 6 h to 40 min with Databricks pipelines in PySpark over 2 TB/day of payment events.",
        "pt": "Reduzi a latência de 6 h para 40 min com pipelines Databricks em PySpark sobre 2 TB/dia de eventos de pagamento.",
        "note": "outcome first"}
INVENTED = {"en": "Built Databricks pipelines with Snowflake for 3 TB/day of payment events.",
            "pt": "Construí pipelines Databricks com Snowflake para 3 TB/dia de eventos de pagamento.",
            "note": "adds Snowflake"}


def test_drafting_needs_the_opt_in(make_ctx):
    with pytest.raises(GenerationError, match="disabled"):
        draft_versions(make_ctx(), "acme-dbx-pipeline-01")


def test_drafts_are_checked_and_flagged(make_ctx):
    client = FakeClient(versions=[[GOOD, INVENTED, GOOD]])
    ctx = make_ctx(client=client, allow_generation=True)
    out = draft_versions(ctx, "acme-dbx-pipeline-01", note="lead with the outcome")
    assert [d["note"] for d in out] == ["outcome first", "adds Snowflake"]  # the repeat is dropped
    assert out[0]["warnings"] == []
    assert any("3" in w for w in out[1]["warnings"]) and any("snowflake" in w for w in out[1]["warnings"])
    user = client.calls[0]["messages"][0]["content"]
    assert "lead with the outcome" in user and "acme-dbx-pipeline-01" in user
    assert "Acme Pagamentos" in user  # where the item sits


def test_regenerate_passes_earlier_drafts(make_ctx):
    client = FakeClient(versions=[[GOOD]])
    ctx = make_ctx(client=client, allow_generation=True)
    with pytest.raises(GenerationError, match="no usable drafts"):  # only repeats came back
        draft_versions(ctx, "acme-dbx-pipeline-01", previous=[GOOD])
    assert GOOD["en"] in client.calls[0]["messages"][0]["content"]


def test_only_bullets_and_summaries(make_ctx, master):
    assert version_sources(master, "sum-platform")[3].startswith("the profile summary")
    item, others, versions, where = version_sources(master, "datacorp-dbt-01")
    assert item.id == "datacorp-dbt-01" and all(o.id != item.id for o in others) and versions == []
    with pytest.raises(GenerationError, match="not a bullet or summary"):
        version_sources(master, "job-acme")


def test_check_version_ignores_sentence_starts():
    class S:
        en, pt = "Built pipelines in Python.", "Construí pipelines em Python."
    d = check_version({"en": "Delivered pipelines in Python. Shipped fast.", "pt": "Entreguei pipelines em Python.", "note": ""}, [S])
    assert d["warnings"] == []
    assert check_version({"en": "", "pt": "x"}, [S]) is None


def test_desktop_draft_job_writes_nothing(make_ctx, tmp_path, monkeypatch):
    import dataclasses
    import shutil
    from cvtailor.desktop import Api
    from cvtailor.pipeline import Context
    from tests.conftest import EXAMPLE_MASTER
    (tmp_path / "data").mkdir(exist_ok=True)
    mp = tmp_path / "data" / "master.yaml"
    shutil.copy(EXAMPLE_MASTER, mp)
    before = mp.read_bytes()
    client = FakeClient(versions=[[GOOD]])

    def factory():
        base = make_ctx(client=client, allow_generation=True)
        ctx = Context(dataclasses.replace(base.cfg, master_path=mp))
        ctx.llm = base.llm
        return ctx
    api = Api(factory)
    assert api.builder_state()["allow_generation"] is True
    jid = api.builder_draft_versions("acme-dbx-pipeline-01", "shorter", [])
    import time
    for _ in range(200):
        j = api.job_status(jid)
        if j["done"]:
            break
        time.sleep(0.02)
    assert j["error"] is None and j["result"][0]["en"] == GOOD["en"]
    assert any("Drafting with" in line for line in j["lines"])
    assert mp.read_bytes() == before


def test_local_client_streams_the_answer(monkeypatch):
    import io
    import json
    from cvtailor import local_llm
    from cvtailor.llm import LLM
    answer = json.dumps({"drafts": [GOOD]})
    pieces = [answer[:30], answer[30:90], answer[90:]]
    events = [b'data: ' + json.dumps({"choices": [{"delta": {"content": p}, "finish_reason": None}]}).encode() + b"\n\n"
              for p in pieces]
    events += [b'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}).encode() + b"\n\n",
               b'data: ' + json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}}).encode() + b"\n\n",
               b"data: [DONE]\n\n"]
    sent = {}

    def urlopen(req, timeout=None):
        sent.update(json.loads(req.data))
        return io.BytesIO(b"".join(events))
    monkeypatch.setattr(local_llm.urllib.request, "urlopen", urlopen)
    seen = []
    llm = LLM("m", None, client=local_llm.LocalClient("http://x/v1"))
    drafts, usage = llm.versions("ctx", n=3, reasoning="none", on_text=seen.append)
    assert sent["stream"] is True and sent["reasoning_effort"] == "none"
    assert drafts == [GOOD] and usage.output_tokens == 7
    assert seen[-1] == answer and len(seen) == 3 and seen[0] == pieces[0]
