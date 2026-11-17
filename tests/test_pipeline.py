import copy

import pytest

from cvtailor import db
from cvtailor.editor import EditAborted, parse_annotated, to_annotated_yaml
from cvtailor.pipeline import (
    IO, NewOptions, PipelineError, Skipped, approve, check_edit, job_hash, regen, rerender,
    run_new, save_edit, split_url_line,
)
from cvtailor.schema import Selection
from tests.conftest import FakeClient, FakeRenderer


def llm_answer(selection_dict, **kw):
    d = copy.deepcopy(selection_dict)
    d.update({"unmatched_requirements": ["dbt"], "suggestions": ["add a kafka bullet"]})
    d.update(kw)
    return d


def test_split_url_line():
    text, url = split_url_line("\n  https://x.example/job/1 \nSenior DE\nbody")
    assert url == "https://x.example/job/1" and text == "Senior DE\nbody"
    assert split_url_line("Senior DE\nhttps://x")[1] is None


def test_job_hash_normalizes_whitespace_and_case():
    assert job_hash("Hello   World\n") == job_hash("hello world")


def test_new_generated_end_to_end(make_ctx, selection_dict, job_text, quiet_io):
    client = FakeClient(selections=[llm_answer(selection_dict)])
    ctx = make_ctx(client=client)
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    assert res.app_id == 1 and res.route == "generated"
    assert res.pdf_path.exists()
    assert res.pdf_path.parent.name.startswith("0001_") and res.pdf_path.parent.name.endswith("_acme-pagamentos")
    assert res.pdf_path.name == "Ana_Souza_CV_Senior_Data_Engineer_EN.pdf"
    row = db.get_application(ctx.db(), 1)
    assert row["job_url"] == "https://jobs.example.com/acme/senior-data-engineer"
    assert not row["job_description"].startswith("https://")
    assert row["suggestions"] == ["add a kafka bullet"] and row["unmatched_requirements"] == ["dbt"]
    assert row["input_tokens"] == 1050 and row["cache_read_tokens"] == 800 and row["output_tokens"] == 220
    assert row["selection"] == Selection.model_validate(selection_dict).model_dump()
    assert row["llm_selection"]["suggestions"] == ["add a kafka bullet"]
    assert row["trimmed"] == [] and row["status"] == "applied"
    out = "\n".join(quiet_io.lines)
    assert "Acme Pagamentos | Senior Data Engineer | en" in out
    assert "#1 -> " in out


def test_meta_skipped_when_flags_given(make_ctx, selection_dict, job_text, quiet_io):
    client = FakeClient(selections=[llm_answer(selection_dict)])
    ctx = make_ctx(client=client)
    res = run_new(ctx, job_text, NewOptions(company="X Corp", role="DE", lang="pt"), quiet_io)
    assert len(client.calls) == 1  # only the selection call
    assert res.lang == "pt" and res.pdf_path.name.endswith("_PT.pdf")


def test_meta_failure_prompts_interactively(make_ctx, selection_dict, job_text):
    client = FakeClient(selections=[llm_answer(selection_dict)], meta_error=RuntimeError("boom"))
    answers = {"Company": "Typed Co", "Role title": "Typed Role", "Language (en/pt)": "pt"}
    io = IO(out=lambda s: None, confirm=lambda q: True, prompt=lambda q, d: answers[q])
    res = run_new(make_ctx(client=client), job_text, NewOptions(), io)
    assert (res.company, res.role, res.lang) == ("Typed Co", "Typed Role", "pt")


def test_meta_failure_non_interactive_fails(make_ctx, job_text):
    client = FakeClient(meta_error=RuntimeError("boom"))
    with pytest.raises(PipelineError, match="meta extraction failed"):
        run_new(make_ctx(client=client), job_text, NewOptions(), IO(out=lambda s: None, interactive=False))


def test_dry_run_writes_nothing(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]))
    res = run_new(ctx, job_text, NewOptions(dry_run=True), quiet_io)
    assert res.app_id is None
    assert db.list_applications(ctx.db()) == []
    assert not ctx.cfg.output_dir.exists()
    assert any("acme-dbx-pipeline-01  Built Databricks" in l for l in quiet_io.lines)


def test_failure_leaves_no_row_and_no_files(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]), capacity=1)
    with pytest.raises(PipelineError, match="minimum bullets"):
        run_new(ctx, job_text, NewOptions(), quiet_io)
    assert db.list_applications(ctx.db()) == []
    assert not ctx.cfg.output_dir.exists() or not any(ctx.cfg.output_dir.iterdir())


def test_write_failure_rolls_back(make_ctx, selection_dict, job_text, quiet_io, monkeypatch):
    import cvtailor.pipeline as p
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]))

    def boom(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        raise OSError("disk full")
    monkeypatch.setattr(p, "write_atomic", boom)
    with pytest.raises(OSError):
        run_new(ctx, job_text, NewOptions(), quiet_io)
    assert db.list_applications(ctx.db()) == []
    assert not any(ctx.cfg.output_dir.iterdir())


def test_trim_is_logged(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]), capacity=12)
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    row = db.get_application(ctx.db(), res.app_id)
    assert row["trimmed"] == ["acme-cost-05"]
    assert "acme-cost-05" not in row["selection"]["bullets"]["job-acme"]
    assert "acme-cost-05" in row["llm_selection"]["bullets"]["job-acme"]


def test_no_trim_fails(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]), capacity=12)
    with pytest.raises(PipelineError, match="--no-trim|edit"):
        run_new(ctx, job_text, NewOptions(no_trim=True), quiet_io)


def test_duplicate_hash_blocked_and_company_warns(make_ctx, selection_dict, job_text, quiet_io):
    client = FakeClient(selections=[llm_answer(selection_dict)] * 3)
    ctx = make_ctx(client=client)
    run_new(ctx, job_text, NewOptions(force_generate=True), quiet_io)
    with pytest.raises(PipelineError, match="already used for application #1"):
        run_new(ctx, job_text, NewOptions(force_generate=True), quiet_io)
    # same company+role, different text: warning + confirm
    asked = []
    io = IO(out=quiet_io.out, confirm=lambda q: asked.append(q) or False)
    with pytest.raises(PipelineError, match="aborted"):
        run_new(ctx, job_text + "\nAlso: Terraform.", NewOptions(force_generate=True), io)
    assert asked
    # batch mode skips it
    with pytest.raises(Skipped):
        run_new(ctx, job_text + "\nAlso: Terraform.", NewOptions(force_generate=True),
                IO(out=lambda s: None, interactive=False))
    # --yes continues; --allow-duplicate allows the same text
    run_new(ctx, job_text + "\nAlso: Terraform.", NewOptions(force_generate=True, yes=True), quiet_io)
    run_new(ctx, job_text, NewOptions(force_generate=True, allow_duplicate=True, yes=True), quiet_io)
    assert len(db.list_applications(ctx.db())) == 3


def test_approve_then_reuse_without_selection_call(make_ctx, selection_dict, job_text, quiet_io):
    client = FakeClient(selections=[llm_answer(selection_dict)])
    ctx = make_ctx(client=client)
    run_new(ctx, job_text, NewOptions(), quiet_io)
    approve(ctx, 1, "platform-v1", replace=False, io=quiet_io)
    assert db.get_application(ctx.db(), 1)["variant_name"] == "platform-v1"
    n_calls = len(client.calls)
    res = run_new(ctx, job_text + "\nFull-time.", NewOptions(yes=True), quiet_io)
    assert res.route == "reused" and res.variant_name == "platform-v1"
    assert len(client.calls) == n_calls + 1  # meta only, no selection call
    row = db.get_application(ctx.db(), res.app_id)
    assert row["llm_selection"] is None and row["route_score"] == pytest.approx(1.0)
    assert row["route_candidates"][0]["name"] == "platform-v1"


def test_approve_refuses_overflow(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]))
    run_new(ctx, job_text, NewOptions(), quiet_io)
    ctx.renderer = FakeRenderer(capacity=5)
    with pytest.raises(PipelineError, match="EN is 2 pages .1 over"):
        approve(ctx, 1, "v", replace=False, io=quiet_io)


def test_render_other_language_same_folder(make_ctx, selection_dict, job_text, quiet_io):
    client = FakeClient(selections=[llm_answer(selection_dict)])
    ctx = make_ctx(client=client)
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    n_calls = len(client.calls)
    path = rerender(ctx, res.app_id, "pt", quiet_io)
    assert path.parent == res.pdf_path.parent and path.name.endswith("_PT.pdf")
    assert path.read_bytes().endswith(b"pt")
    assert len(client.calls) == n_calls
    assert len(db.list_applications(ctx.db())) == 1


def test_regen_updates_in_place(make_ctx, selection_dict, job_text, quiet_io):
    second = llm_answer(selection_dict, summary_id="sum-analytics")
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict), second]))
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    db.update_application(ctx.db(), res.app_id, {"edited": 1})
    regen(ctx, res.app_id, quiet_io)
    row = db.get_application(ctx.db(), res.app_id)
    assert row["selection"]["summary_id"] == "sum-analytics" and row["edited"] == 0
    assert len(db.list_applications(ctx.db())) == 1


def test_invalid_edit_rejected_and_stored_unchanged(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]))
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    app = db.get_application(ctx.db(), res.app_id)
    before = app["selection"]
    pdf_before = res.pdf_path.read_bytes()
    text = to_annotated_yaml(Selection.model_validate(before), ctx.master)
    broken = text.replace("    - intern-python-01", "    - acme-streaming-06")  # wrong job
    errors, fit = check_edit(ctx, app, parse_annotated(broken))
    assert errors and fit is None
    assert db.get_application(ctx.db(), res.app_id)["selection"] == before
    assert res.pdf_path.read_bytes() == pdf_before


def test_valid_edit_saved(make_ctx, selection_dict, job_text, quiet_io):
    ctx = make_ctx(client=FakeClient(selections=[llm_answer(selection_dict)]))
    res = run_new(ctx, job_text, NewOptions(), quiet_io)
    app = db.get_application(ctx.db(), res.app_id)
    text = to_annotated_yaml(Selection.model_validate(app["selection"]), ctx.master)
    text = text.replace("    # - acme-streaming-06", "    - acme-streaming-06")
    text = text.replace("    - acme-cost-05", "    # - acme-cost-05")
    sel = parse_annotated(text)
    errors, fit = check_edit(ctx, app, sel)
    assert errors == []
    save_edit(ctx, app, fit)
    row = db.get_application(ctx.db(), res.app_id)
    assert row["edited"] == 1 and "acme-streaming-06" in row["selection"]["bullets"]["job-acme"]


def test_empty_edit_aborts():
    with pytest.raises(EditAborted):
        parse_annotated("# nothing here\n\n")
