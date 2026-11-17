"""Local OpenAI-compatible backend. The HTTP call is faked; no server is needed."""

import io
import json
import urllib.error

import pytest

from cvtailor import local_llm
from cvtailor.config import Config
from cvtailor.llm import LLM, LLMError, build_selection_schema
from cvtailor.local_llm import LocalClient
from tests.conftest import EXAMPLE_SELECTION

SELECTION = json.loads(EXAMPLE_SELECTION.read_text(encoding="utf-8"))


class FakeHTTP:
    """Replaces urllib.request.urlopen; answers each call with the next queued reply."""

    def __init__(self, monkeypatch, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        monkeypatch.setattr(local_llm.urllib.request, "urlopen", self)

    def __call__(self, req, timeout=None):
        self.requests.append({"url": req.full_url, "headers": dict(req.header_items()),
                              "body": json.loads(req.data), "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())


def chat(content, finish="stop", usage=(120, 30)):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def selection_reply():
    sel = dict(SELECTION, unmatched_requirements=[], suggestions=[])
    return chat(json.dumps(sel))


def test_select_through_local_server(monkeypatch, master):
    http = FakeHTTP(monkeypatch, selection_reply())
    llm = LLM("qwen3:8b", None, client=LocalClient("http://localhost:11434/v1/", timeout=30),
              count_limits=True)
    sel, usage = llm.select("Senior data engineer, Databricks", master)

    assert sel.summary_id == SELECTION["summary_id"]
    assert (usage.input_tokens, usage.output_tokens) == (120, 30)
    req = http.requests[0]
    assert req["url"] == "http://localhost:11434/v1/chat/completions"
    assert req["timeout"] == 30
    assert "Authorization" not in req["headers"]
    body = req["body"]
    assert body["model"] == "qwen3:8b" and body["stream"] is False
    system, user = body["messages"]
    assert system["role"] == "system" and "MASTER RESUME INDEX" in system["content"]
    assert user == {"role": "user", "content": "JOB DESCRIPTION:\nSenior data engineer, Databricks"}
    # the selection schema, with count limits, goes out as response_format
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema == build_selection_schema(master, count_limits=True)
    assert "cache_control" not in json.dumps(body) and "effort" not in json.dumps(body)


def test_count_limits_in_schema(master):
    lim = master.limits
    plain = build_selection_schema(master)
    assert "minItems" not in json.dumps(plain)  # the Anthropic API rejects these
    s = build_selection_schema(master, count_limits=True)["properties"]
    for j in master.jobs:
        got = s["bullets"]["properties"][j.id]
        assert got["minItems"] == min(lim.min_bullets_per_job, len(j.bullets))
        assert got["maxItems"] == min(lim.max_bullets_per_job, len(j.bullets))
    proj = s["bullets"]["properties"]["proj-airflow-demo"]
    assert (proj["minItems"], proj["maxItems"]) == (0, min(lim.max_bullets_per_project, 2))
    assert s["project_ids"]["maxItems"] == lim.max_projects


def test_think_block_stripped_and_api_key_sent(monkeypatch, master):
    meta = {"company": "Acme", "role_title": "Data Engineer", "language": "en", "job_url": None}
    http = FakeHTTP(monkeypatch, chat("<think>\nhmm\n</think>\n" + json.dumps(meta)))
    llm = LLM("qwen3:8b", None, client=LocalClient("http://h/v1", api_key="k"))
    got, _ = llm.extract_meta("Acme hires a Data Engineer")
    assert got.company == "Acme"
    assert http.requests[0]["headers"]["Authorization"] == "Bearer k"


def test_truncated_answer_is_an_error(monkeypatch, master):
    FakeHTTP(monkeypatch, chat('{"company": "Ac', finish="length"))
    llm = LLM("m", None, client=LocalClient("http://h/v1"))
    with pytest.raises(LLMError, match="max_tokens"):
        llm.extract_meta("job")


def test_server_down_and_http_error(monkeypatch):
    FakeHTTP(monkeypatch,
             urllib.error.URLError("connection refused"),
             urllib.error.HTTPError("http://h/v1/chat/completions", 404, "nf", {},
                                    io.BytesIO(b'{"error":"model \\"qwen\\" not found"}')))
    llm = LLM("qwen", None, client=LocalClient("http://h/v1"))
    with pytest.raises(LLMError, match="could not reach the local LLM.*CV_LLM_BASE_URL"):
        llm.extract_meta("job")
    with pytest.raises(LLMError, match="HTTP 404.*not found"):
        llm.extract_meta("job")


def test_config_and_context_pick_local_backend(monkeypatch, make_ctx):
    monkeypatch.setenv("CV_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CV_MODEL", "qwen3:8b")
    monkeypatch.setenv("CV_LLM_TIMEOUT", "90")
    cfg = Config.from_env()
    assert (cfg.llm_base_url, cfg.model, cfg.llm_timeout) == ("http://localhost:11434/v1", "qwen3:8b", 90.0)

    ctx = make_ctx()
    ctx.llm = None
    ctx.cfg.llm_base_url = "http://localhost:11434/v1"
    llm = ctx.get_llm()
    assert isinstance(llm.client, LocalClient)
    assert llm.count_limits and llm.effort is None
