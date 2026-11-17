import copy

import pytest

from cvtailor.llm import LLM, LLMError, SELECTION_SYSTEM_PROMPT, build_selection_schema
from tests.conftest import FakeClient


def test_schema_enumerates_master_ids(master):
    s = build_selection_schema(master)
    props = s["properties"]
    assert props["summary_id"]["enum"] == [x.id for x in master.summaries]
    owners = [*master.jobs, *(p for p in master.projects if p.bullets)]
    assert set(props["bullets"]["required"]) == {o.id for o in owners}
    assert "proj-airflow-demo" in props["bullets"]["required"]  # the example project with bullets
    assert props["bullets"]["additionalProperties"] is False
    for o in owners:
        assert props["bullets"]["properties"][o.id]["items"]["enum"] == [b.id for b in o.bullets]
    assert props["project_ids"]["items"]["enum"] == [p.id for p in master.projects]
    for tier in ("hands_on", "working", "familiar"):
        got = props["skill_category_order"]["properties"][tier]["items"]["enum"]
        assert got == [c.id for c in master.skills.tier(tier)]
    assert s["additionalProperties"] is False


def test_schema_handles_empty_tier(master_dict):
    from cvtailor.master import parse_master
    master_dict["skills"]["familiar"] = []
    s = build_selection_schema(parse_master(master_dict))
    assert "enum" not in s["properties"]["skill_category_order"]["properties"]["familiar"]["items"]


def _llm_sel(selection_dict, **extra):
    d = copy.deepcopy(selection_dict)
    d.update({"unmatched_requirements": ["kafka"], "suggestions": ["add dbt bullet"]})
    d.update(extra)
    return d


def test_select_success_and_request_shape(master, selection_dict, job_text):
    client = FakeClient(selections=[_llm_sel(selection_dict)])
    sel, usage = LLM("m", "low", client=client).select(job_text, master)
    assert sel.summary_id == "sum-platform"
    assert sel.unmatched_requirements == ["kafka"]
    assert usage.cache_read_tokens == 800
    call = client.calls[0]
    assert call["system"][0]["text"] == SELECTION_SYSTEM_PROMPT
    assert call["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "Ana Souza" not in call["system"][1]["text"]  # no profile data sent
    assert "ana.souza@example.com" not in str(call)
    assert call["messages"][0]["content"].startswith("JOB DESCRIPTION:")
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in call


def test_select_retries_once_with_errors(master, selection_dict, job_text):
    bad = _llm_sel(selection_dict)
    bad["bullets"]["job-intern"] = ["acme-streaming-06"]  # wrong job
    client = FakeClient(selections=[bad, _llm_sel(selection_dict)])
    sel, usage = LLM("m", "low", client=client).select(job_text, master)
    assert len(client.calls) == 2
    assert "belongs to job-acme, not job-intern" in client.calls[1]["messages"][0]["content"]
    assert usage.input_tokens == 2000


def test_select_fails_after_second_invalid(master, selection_dict, job_text):
    bad = _llm_sel(selection_dict)
    del bad["bullets"]["job-intern"]
    client = FakeClient(selections=[bad, "not json"])
    with pytest.raises(LLMError, match="failed validation twice"):
        LLM("m", "low", client=client).select(job_text, master)


def test_select_caps_lists(master, selection_dict, job_text):
    many = _llm_sel(selection_dict, unmatched_requirements=list("abcdefg"), suggestions=list("wxyz"))
    sel, _ = LLM("m", None, client=FakeClient(selections=[many])).select(job_text, master)
    assert len(sel.unmatched_requirements) == 5 and len(sel.suggestions) == 3


def test_meta_language_and_url(job_text):
    client = FakeClient(meta={"company": "Acme", "role_title": "DE", "language": "pt",
                              "job_url": "https://invented.example/x"})
    meta, _ = LLM("m", "low", client=client).extract_meta(job_text)
    assert meta.language == "pt"
    assert meta.job_url is None  # not literally in the text
    assert "Ana" not in str(client.calls[0])
