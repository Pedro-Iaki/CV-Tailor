"""Careful mode: hints, free-text analysis, selection, one review round."""

import io
import json

from cvtailor import careful
from cvtailor.careful import careful_select, embedding_hints, examples_block, tag_hints
from cvtailor.llm import LLM
from cvtailor.local_llm import LocalClient
from cvtailor.pipeline import NewOptions, run_new
from cvtailor.tags import match_tags
from tests.conftest import FakeClient
from tests.test_local_llm import FakeHTTP, chat

ANALYSIS = "Requirements: Spark, Delta. Coverage: acme-dbx-pipeline-01 strong."


def sel_reply(selection_dict, **over):
    return chat(json.dumps({**selection_dict, "unmatched_requirements": [], "suggestions": [], **over}))


def local_llm(reasoning=None):
    return LLM("m", None, client=LocalClient("http://h/v1", reasoning=reasoning), count_limits=True)


def test_full_round(monkeypatch, master, selection_dict, job_text):
    revised = {**selection_dict, "summary_id": "sum-analytics"}
    http = FakeHTTP(monkeypatch, chat(ANALYSIS), sel_reply(selection_dict),
                    chat("Use sum-analytics: the job is analytics-heavy."), sel_reply(revised))
    examples = [{"name": "Other job", "job": "Analytics engineer, dbt.", "notes": "Chose sum-analytics."}]
    res = careful_select(local_llm("none"), job_text, master, matches=match_tags(job_text, master),
                         examples=examples, reasoning="")
    assert res.revised and res.selection.summary_id == "sum-analytics"
    assert res.notes == ["analysis: 6 words, with a worked example", "review: revised"]
    a, s1, r, s2 = [q["body"] for q in http.requests]
    # analysis: free text, thinking left to the model ("" = default), hints and example included
    assert "response_format" not in a and "reasoning_effort" not in a
    assert "RELEVANCE HINTS" in a["messages"][1]["content"] and "WORKED EXAMPLE (Other job)" in a["messages"][0]["content"]
    # selection: the client's own setting, with the analysis as context
    assert s1["reasoning_effort"] == "none" and "response_format" in s1
    assert "ANALYSIS (follow it where it is right):\n" + ANALYSIS in s1["messages"][1]["content"]
    # review sees the draft; the revision sees draft and review
    assert "DRAFT SELECTION:\nsummary: sum-platform" in r["messages"][1]["content"]
    assert "REVIEW OF A FIRST DRAFT" in s2["messages"][1]["content"] and "analytics-heavy" in s2["messages"][1]["content"]
    assert res.usage.output_tokens == 4 * 30


def test_no_changes_keeps_draft(monkeypatch, master, selection_dict, job_text):
    http = FakeHTTP(monkeypatch, chat(ANALYSIS), sel_reply(selection_dict), chat("NO CHANGES"))
    res = careful_select(local_llm(), job_text, master, matches={})
    assert not res.revised and len(http.requests) == 3 and res.notes[-1] == "review: no changes"


def test_failed_revision_keeps_valid_draft(monkeypatch, master, selection_dict, job_text):
    bad = {**selection_dict, "summary_id": "nope"}
    FakeHTTP(monkeypatch, chat(ANALYSIS), sel_reply(selection_dict), chat("Change the summary."),
             sel_reply(bad), sel_reply(bad))
    res = careful_select(local_llm(), job_text, master, matches={})
    assert not res.revised and res.selection.summary_id == selection_dict["summary_id"]
    assert "the revision failed" in res.notes[-1]


def test_cut_off_passes_fall_back(monkeypatch, master, selection_dict, job_text):
    """A truncated analysis or review never fails the run: they are evidence only."""
    http = FakeHTTP(monkeypatch, chat("", finish="length"), sel_reply(selection_dict), chat("", finish="length"))
    llm = LLM("m", None, client=LocalClient("http://h/v1"), count_limits=True, max_output=10000)
    res = careful_select(llm, job_text, master, matches={})
    assert res.selection.summary_id == selection_dict["summary_id"] and res.analysis == ""
    assert res.notes == ["analysis: cut off at the token limit, selected without it",
                         "review: cut off at the token limit; kept the first draft"]
    assert http.requests[0]["body"]["max_tokens"] == 10000  # free text gets the full limit
    assert "ANALYSIS" not in http.requests[1]["body"]["messages"][1]["content"]


def test_tag_hints(master, job_text):
    hints = tag_hints(master, match_tags(job_text, master))
    assert hints and all(h.startswith("- ") for h in hints)
    assert any("acme-dbx-pipeline-01" in h and '("' in h for h in hints)


def test_examples_skip_the_same_posting():
    ex = [{"name": "A", "job": "Data Engineer at X", "notes": "n"}, {"name": "B", "job": "Other", "notes": "m"}]
    block = examples_block(ex, "data engineer at x!")
    assert "(A)" not in block and "(B)" in block


def test_embedding_hints(monkeypatch, master):
    class Embed:
        def __call__(self, req, timeout=None):
            body = json.loads(req.data)
            self.url = req.full_url
            # line 0 -> [1, 0]; every item [0, 1] except the first, which matches line 0
            vecs = [[1, 0]] + [[1, 0] if i == 0 else [0, 1] for i in range(len(body["input"]) - 1)]
            return io.BytesIO(json.dumps({"embeddings": vecs}).encode())
    fake = Embed()
    monkeypatch.setattr(careful.urllib.request, "urlopen", fake)
    hints = embedding_hints("Spark pipelines engineer", master, "bge-m3", "http://localhost:11434/v1")
    assert fake.url == "http://localhost:11434/api/embed"
    assert hints[0].endswith('1.00, closest to "Spark pipelines engineer"')

    def down(req, timeout=None):
        raise OSError("refused")
    monkeypatch.setattr(careful.urllib.request, "urlopen", down)
    assert embedding_hints("x y z w", master, "bge-m3", "http://h/v1") == []  # optional: never fails the run


def test_pipeline_careful_dry_run(make_ctx, quiet_io, job_text, selection_dict):
    client = FakeClient(selections=[selection_dict], texts=[ANALYSIS, "NO CHANGES"])
    ctx = make_ctx(client=client)
    run_new(ctx, job_text, NewOptions(dry_run=True, careful=True, yes=True, verbose=True), quiet_io)
    out = "\n".join(quiet_io.lines)
    assert "careful: analysis: 6 words; review: no changes" in out
    assert "--- analysis ---\n" + ANALYSIS in out
    assert len(client.calls) == 4  # meta, analysis, selection, review
