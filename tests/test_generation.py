"""Opt-in text generation: drafts, y/n review, facts vs unsafe items, promotion."""

import copy
import json

import pytest
from typer.testing import CliRunner

from cvtailor import db
from cvtailor import mcp_server as srv
from cvtailor.facts import load_facts, load_unsafe, promote
from cvtailor.generation import (
    GenerationError,
    apply_approved,
    check_proposal,
    create_proposals,
    parse_reply,
    review,
)
from cvtailor.pipeline import NewOptions, run_new
from cvtailor.render import make_render_input, render_html
from cvtailor.schema import Selection
from cvtailor.validate import validate_selection
from tests.conftest import FakeClient

REPHRASE = {
    "kind": "bullet", "job_id": "job-acme", "origin": "rephrase",
    "based_on": ["acme-dbx-pipeline-01"], "replaces": ["acme-dbx-pipeline-01"], "tags": ["databricks", "spark"],
    "en": "Engineered PySpark pipelines on Databricks moving 2 TB/day of payment events into Delta Lake, "
          "cutting batch latency from 6 h to 40 min.",
    "pt": "Desenvolvi pipelines PySpark no Databricks levando 2 TB/dia de eventos de pagamento ao Delta Lake, "
          "reduzindo a latência de 6 h para 40 min.",
    "rationale": "Mirrors the posting's 'design PySpark pipelines'.",
}
NEW = {
    "kind": "bullet", "job_id": "job-acme", "origin": "new", "based_on": [], "replaces": [], "tags": ["dbt"],
    "en": "Built dbt models for payment reporting.", "pt": "Construí modelos dbt para relatórios de pagamentos.",
    "rationale": "Confirm you used dbt at Acme.",
}
MERGE = {
    "kind": "bullet", "job_id": "job-datacorp", "origin": "merge",
    "based_on": ["datacorp-dbt-01", "datacorp-sql-02"], "replaces": ["datacorp-dbt-01", "datacorp-sql-02"],
    "tags": ["dbt", "sql"],
    "en": "Modeled the sales domain in dbt and tuned PostgreSQL, cutting dashboard load times by 70%.",
    "pt": "Modelei o domínio de vendas em dbt e otimizei o PostgreSQL, reduzindo a carga dos dashboards em 70%.",
    "rationale": "Shorter.",
}


def answer(selection_dict):
    d = copy.deepcopy(selection_dict)
    d.update({"unmatched_requirements": ["dbt"], "suggestions": []})
    return d


@pytest.fixture
def gen_ctx(make_ctx, selection_dict, job_text, quiet_io):
    def _make(proposals=(), revisions=(), selections=2):
        client = FakeClient(selections=[answer(selection_dict)] * selections,
                            proposals=list(proposals), revisions=list(revisions))
        ctx = make_ctx(client=client, allow_generation=True)
        run_new(ctx, job_text, NewOptions(), quiet_io)
        return ctx, client
    return _make


# ---------------------------------------------------------------- unit

@pytest.mark.parametrize("text,expected", [
    ("y", (True, None)), ("Y", (True, None)), ("n", (False, None)), ("sim", (True, None)),
    ("y, make sure to place sample text: before that though",
     (True, "make sure to place sample text: before that though")),
    ("n - too vague", (False, "too vague")),
    ("y adapt context from a stored verified/master bullet",
     (True, "adapt context from a stored verified/master bullet")),
])
def test_parse_reply(text, expected):
    assert parse_reply(text) == expected


@pytest.mark.parametrize("bad", ["", "maybe", "yeah sure", "nope"])
def test_parse_reply_rejects(bad):
    with pytest.raises(ValueError):
        parse_reply(bad)


def test_check_rephrase_is_fact(master):
    p, notes = check_proposal(REPHRASE, master)
    assert p["destination"] == "fact" and notes == []


def test_check_new_number_downgrades_to_unsafe(master):
    raw = dict(REPHRASE, en=REPHRASE["en"].replace("40 min", "25 min"))
    p, notes = check_proposal(raw, master)
    assert p["origin"] == "new" and p["destination"] == "unsafe" and p["replaces"] == []
    assert "25" in notes[0]


def test_check_new_is_unsafe(master):
    assert check_proposal(NEW, master)[0]["destination"] == "unsafe"


@pytest.mark.parametrize("change,msg", [
    ({"based_on": ["acme-delta-02", "datacorp-dbt-01"], "origin": "merge"}, "same job"),
    ({"based_on": ["nope"]}, "unknown source"),
    ({"based_on": ["acme-delta-02", "acme-adf-03"]}, "exactly one source"),
    ({"pt": ""}, "Portuguese"),
    ({"kind": "summary", "origin": "merge"}, "not merged"),
    ({"kind": "summary", "job_id": "", "based_on": []}, "exactly one source"),  # used to raise IndexError
])
def test_check_rejects(master, change, msg):
    p, notes = check_proposal({**REPHRASE, **change}, master)
    assert p is None and msg in notes[0]


def test_disabled_by_default(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[answer(selection_dict)]))
    run_new(ctx, job_text, NewOptions(), quiet_io)
    with pytest.raises(GenerationError, match="CV_ALLOW_GENERATION"):
        create_proposals(ctx, 1)


# ---------------------------------------------------------------- review flow

def test_review_flow_facts_unsafe_rejected(gen_ctx, quiet_io):
    ctx, client = gen_ctx(proposals=[[REPHRASE, NEW, MERGE]])
    props = create_proposals(ctx, 1, quiet_io)
    assert [p["destination"] for p in props] == ["fact", "unsafe", "fact"]
    r_id, n_id, m_id = (p["id"] for p in props)

    res = review(ctx, 1, {r_id: "y", n_id: "y", m_id: "n, too vague, keep them separate"})
    assert res.facts == ["gen-acme-01"] and res.unsafe == ["gen-acme-02"] and res.rejected == [m_id]

    path = apply_approved(ctx, 1, quiet_io)
    assert path is not None
    row = db.get_application(ctx.db(), 1)
    assert row["selection"]["bullets"]["job-acme"][0] == "gen-acme-01"         # replaced its source in place
    assert "acme-dbx-pipeline-01" not in row["selection"]["bullets"]["job-acme"]
    assert row["generated_ids"] == ["gen-acme-01"]
    assert row["input_tokens"] > 1050                                            # generation tokens added

    # facts are part of the renderable master; unsafe items are not
    ids = {b.id for j in ctx.master.jobs for b in j.bullets}
    assert "gen-acme-01" in ids and "gen-acme-02" not in ids
    html = render_html(make_render_input(ctx.master, Selection.model_validate(row["selection"]), "en"))
    assert "Engineered PySpark pipelines" in html and "Built dbt models" not in html


def test_merge_removes_every_selected_source(gen_ctx, quiet_io):
    # `replaces` names only one source; the other must still leave the selection,
    # or rule 8 (fact + its source) rejects the whole merge.
    ctx, _ = gen_ctx(proposals=[[dict(MERGE, replaces=["datacorp-dbt-01"])]])
    p = create_proposals(ctx, 1, quiet_io)[0]
    assert review(ctx, 1, {p["id"]: "y"}).facts == ["gen-datacorp-01"]
    assert apply_approved(ctx, 1, quiet_io) is not None
    row = db.get_application(ctx.db(), 1)
    assert row["selection"]["bullets"]["job-datacorp"] == ["datacorp-airflow-03", "gen-datacorp-01"]
    assert row["generated_ids"] == ["gen-datacorp-01"]


PROJECT_REPHRASE = {
    "kind": "bullet", "job_id": "proj-airflow-demo", "origin": "rephrase",
    "based_on": ["lakehouse-tests-02"], "replaces": ["lakehouse-tests-02"], "tags": ["testing"],
    "en": "Added pytest data-quality checks that run in CI before each merge.",
    "pt": "Adicionei verificações de qualidade de dados em pytest que rodam no CI antes de cada merge.",
    "rationale": "Tighter.",
}


def test_project_bullet_proposal_is_checked_like_a_job_bullet(master):
    p, notes = check_proposal(PROJECT_REPHRASE, master)
    assert p["job_id"] == "proj-airflow-demo" and p["destination"] == "fact" and notes == []
    wrong = dict(PROJECT_REPHRASE, based_on=["acme-cost-05"])
    assert check_proposal(wrong, master)[0] is None  # source from another job


def test_approved_project_bullet_lands_in_its_project(gen_ctx, quiet_io):
    ctx, _ = gen_ctx(proposals=[[PROJECT_REPHRASE]])
    p = create_proposals(ctx, 1, quiet_io)[0]
    assert review(ctx, 1, {p["id"]: "y"}).facts == ["gen-airflow-demo-01"]
    assert apply_approved(ctx, 1, quiet_io) is not None
    row = db.get_application(ctx.db(), 1)
    assert row["selection"]["bullets"]["proj-airflow-demo"] == ["gen-airflow-demo-01"]
    assert "proj-airflow-demo" in row["selection"]["project_ids"]


def test_unsafe_items_never_reach_selector_or_schema(gen_ctx, quiet_io, job_text):
    ctx, client = gen_ctx(proposals=[[NEW]], selections=2)
    p = create_proposals(ctx, 1, quiet_io)[0]
    review(ctx, 1, {p["id"]: "y"})
    run_new(ctx, job_text + "\nRemote.", NewOptions(yes=True, force_generate=True), quiet_io)
    sel_call = [c for c in client.calls if "summary_id" in c["output_config"]["format"]["schema"]["properties"]][-1]
    assert "gen-acme" not in json.dumps(sel_call)
    assert "Built dbt models" not in json.dumps(sel_call)


def test_comment_triggers_revision_then_approval(gen_ctx, quiet_io):
    revised = dict(REPHRASE, en="Designed PySpark pipelines on Databricks moving 2 TB/day into Delta Lake, "
                                "cutting batch latency from 6 h to 40 min.")
    ctx, client = gen_ctx(proposals=[[REPHRASE]], revisions=[revised])
    p = create_proposals(ctx, 1, quiet_io)[0]
    res = review(ctx, 1, {p["id"]: "y, start with 'Designed' like the posting"})
    assert res.facts == [] and len(res.revised) == 1
    child = res.revised[0]
    assert child["depth"] == 1 and child["parent_id"] == p["id"] and child["en"].startswith("Designed")
    revise_call = client.calls[-1]
    assert "start with 'Designed'" in revise_call["messages"][0]["content"]
    res2 = review(ctx, 1, {child["id"]: "y"})
    [fact] = load_facts(ctx.db())
    assert res2.facts == [fact["id"]] and fact["en"].startswith("Designed")
    assert fact["comment"] == "start with 'Designed' like the posting"


def test_revision_limit(gen_ctx, quiet_io):
    ctx, _ = gen_ctx(proposals=[[REPHRASE]], revisions=[REPHRASE] * 3)
    p = create_proposals(ctx, 1, quiet_io)[0]
    for _ in range(3):
        p = review(ctx, 1, {p["id"]: "y, again"}).revised[0]
    with pytest.raises(ValueError, match="revision limit"):
        review(ctx, 1, {p["id"]: "y, once more"})


def test_bad_reply_changes_nothing(gen_ctx, quiet_io):
    ctx, _ = gen_ctx(proposals=[[REPHRASE, NEW]])
    a, b = (p["id"] for p in create_proposals(ctx, 1, quiet_io))
    with pytest.raises(ValueError):
        review(ctx, 1, {a: "y", b: "perhaps"})
    assert load_facts(ctx.db()) == [] and load_unsafe(ctx.db()) == []


def test_feedback_reaches_next_proposal_prompt(gen_ctx, quiet_io):
    ctx, client = gen_ctx(proposals=[[MERGE], []])
    p = create_proposals(ctx, 1, quiet_io)[0]
    review(ctx, 1, {p["id"]: "n, never merge the dbt and SQL bullets"})
    create_proposals(ctx, 1, quiet_io)
    assert "never merge the dbt and SQL bullets" in client.calls[-1]["messages"][0]["content"]


def test_promote_moves_unsafe_to_facts(gen_ctx, quiet_io):
    ctx, _ = gen_ctx(proposals=[[NEW]])
    p = create_proposals(ctx, 1, quiet_io)[0]
    uid = review(ctx, 1, {p["id"]: "y"}).unsafe[0]
    conn = ctx.db()
    conn.execute("BEGIN")
    promote(conn, uid, en="Built dbt models for payment reporting (internal tool).")
    conn.execute("COMMIT")
    ctx.invalidate_master()
    assert load_unsafe(conn) == []
    [fact] = load_facts(conn)
    assert fact["approved_via"] == "promote" and fact["en"].endswith("(internal tool).")
    assert uid in {b.id for b in ctx.master.jobs[0].bullets}


def test_fact_and_its_source_cannot_both_be_selected(gen_ctx, quiet_io, selection_dict):
    ctx, _ = gen_ctx(proposals=[[REPHRASE]])
    p = create_proposals(ctx, 1, quiet_io)[0]
    review(ctx, 1, {p["id"]: "y"})
    ctx.invalidate_master()
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-acme"][-1] = "gen-acme-01"   # alongside acme-dbx-pipeline-01
    errors = validate_selection(Selection.model_validate(d), ctx.master)
    assert any("rewording of 'acme-dbx-pipeline-01'" in e for e in errors)


# ---------------------------------------------------------------- MCP + CLI

def test_mcp_flow_and_no_promote_tool(gen_ctx, quiet_io, monkeypatch):
    ctx, _ = gen_ctx(proposals=[[REPHRASE, NEW]])
    monkeypatch.setattr(srv, "make_context", lambda: ctx)
    props = srv.propose_text(1)["proposals"]
    assert props[1]["if_approved"].startswith("UNSAFE")
    out = srv.review_proposals(1, {str(props[0]["proposal_id"]): "y", str(props[1]["proposal_id"]): "y"})
    assert out["facts_added"] == ["gen-acme-01"] and out["unsafe_stored"] == ["gen-acme-02"]
    assert out["pdf_path"] and out["reminder"]
    assert [u["id"] for u in srv.list_unsafe()] == ["gen-acme-02"]
    names = {fn.__name__ for fn, _ in srv.TOOLS}
    assert not any("promote" in n for n in names)


def test_cli_promote_requires_confirmation(gen_ctx, quiet_io, monkeypatch, tmp_path):
    import apply
    ctx, _ = gen_ctx(proposals=[[NEW]])
    p = create_proposals(ctx, 1, quiet_io)[0]
    uid = review(ctx, 1, {p["id"]: "y"}).unsafe[0]
    monkeypatch.setattr(apply, "_ctx", lambda: ctx)
    r = CliRunner().invoke(apply.app, ["unsafe", "promote", uid], input="n\n")
    assert r.exit_code == 0 and "skipped" in r.output
    assert len(load_unsafe(ctx.db())) == 1
    r = CliRunner().invoke(apply.app, ["unsafe", "promote", uid], input="y\n")
    assert r.exit_code == 0, r.output
    assert load_unsafe(ctx.db()) == [] and len(load_facts(ctx.db())) == 1


def test_cli_review_loop(gen_ctx, monkeypatch):
    import apply
    revised = dict(REPHRASE, en=REPHRASE["en"].replace("Engineered", "Designed"))
    ctx, _ = gen_ctx(proposals=[[REPHRASE, NEW, MERGE]], revisions=[revised])
    monkeypatch.setattr(apply, "_ctx", lambda: ctx)
    # 1: revise then approve; 2: garbage then approve (unsafe); 3: reject with feedback
    replies = "y, start with Designed\ny\nmaybe\ny\nn, keep them separate\n"
    r = CliRunner().invoke(apply.app, ["generate", "1"], input=replies)
    assert r.exit_code == 0, r.output
    assert "approve -> UNSAFE" in r.output and "could not read reply" in r.output
    assert "review done: 1 fact(s) added, 1 unsafe item(s) stored." in r.output
    [fact] = load_facts(ctx.db())
    assert fact["en"].startswith("Designed")
    assert db.get_application(ctx.db(), 1)["generated_ids"] == [fact["id"]]
