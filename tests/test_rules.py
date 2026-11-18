"""Master-driven rules: headline options, projects gated by job tags, required skill
categories and minimum item counts (grammar, validation, index and render)."""

import json

from cvtailor import choices
from cvtailor.llm import build_selection_schema
import pytest

from cvtailor.master import MasterError, build_llm_index, check_master, gate_projects, parse_master
from cvtailor.render import make_render_input, render_html
from cvtailor.schema import Selection
from cvtailor.validate import validate_selection


def _headlines(md):
    md["profile"]["headline_alts"] = [
        {"id": "hl-analytics", "tags": ["analytics"], "en": "Analytics Engineer", "pt": "Engenheira de Analytics"}]
    return md


def test_headline_option_is_selectable_and_rendered(master_dict, selection_dict):
    m = parse_master(_headlines(master_dict))
    assert check_master(m) == []
    sel = Selection.model_validate({**selection_dict, "headline_id": "hl-analytics"})
    assert validate_selection(sel, m) == []
    html = render_html(make_render_input(m, sel, "pt"))
    assert '<p class="headline">Engenheira de Analytics</p>' in html
    # "default" means no option
    assert Selection.model_validate({**selection_dict, "headline_id": "default"}).headline_id is None
    bad = Selection.model_validate({**selection_dict, "headline_id": "hl-nope"})
    assert any("headline_id 'hl-nope'" in e for e in validate_selection(bad, m))


def test_headline_in_index_and_schemas(master_dict):
    m = parse_master(_headlines(master_dict))
    idx = json.loads(build_llm_index(m))
    assert [h["id"] for h in idx["headline_options"]] == ["default", "hl-analytics"]
    plain = build_selection_schema(m, count_limits=True)
    assert plain["properties"]["headline_id"]["enum"] == ["default", "hl-analytics"]
    s = choices.build_schema(m, plain, True)
    assert list(s["properties"])[:3] == ["key_requirements", "summary_id", "headline_id"]
    # without options nothing changes
    master_dict["profile"].pop("headline_alts")
    assert "headline_id" not in build_selection_schema(parse_master(master_dict))["properties"]


def test_headline_alts_need_a_default(master_dict):
    md = _headlines(master_dict)
    md["profile"].pop("headline")
    with pytest.raises(MasterError, match="headline_alts need a default headline"):
        parse_master(md)


def test_gated_project_hidden_unless_job_mentions_tag(master_dict, selection_dict):
    master_dict["projects"][2]["only_for_tags"] = ["kafka"]  # proj-kafka-lab
    m = parse_master(master_dict)
    assert check_master(m) == []
    assert [p.id for p in gate_projects(m, {"kafka", "spark"}).projects] == [p.id for p in m.projects]
    gated = gate_projects(m, {"spark"})
    assert "proj-kafka-lab" not in {p.id for p in gated.projects}
    sel = Selection.model_validate(selection_dict)  # selects proj-kafka-lab
    assert validate_selection(sel, m) == []  # a manual selection may still use it
    assert any("proj-kafka-lab" in e for e in validate_selection(sel, gated))


def test_gating_happens_in_the_selection_step(make_ctx, master_dict, selection_dict, monkeypatch):
    from cvtailor import pipeline
    seen = {}

    class FakeLLM:
        def select(self, text, master, context=""):
            seen["projects"] = [p.id for p in master.projects]
            raise pipeline.LLMError("stop here")

    ctx = make_ctx()
    monkeypatch.setattr(ctx, "get_llm", lambda: FakeLLM())
    m = parse_master({**master_dict, "projects": [{**master_dict["projects"][2], "only_for_tags": ["llm"]}]})
    try:
        pipeline.select_with_llm(ctx, "Spark job", m, {"spark": "Spark"}, False, pipeline.IO())
    except pipeline.LLMError:
        pass
    assert seen["projects"] == []


def _skill_rules(md, required=True, min_items=2):
    md["skills"]["working"][0]["required"] = required  # skills-working-orchestration
    md["skills"]["working"][0]["min_items"] = min_items
    return md


def test_required_category_and_min_items(master_dict, selection_dict):
    m = parse_master(_skill_rules(master_dict))
    assert check_master(m) == []
    sel = Selection.model_validate(selection_dict)
    assert validate_selection(sel, m) == []
    without = json.loads(json.dumps(selection_dict))
    without["skill_category_order"]["working"] = ["skills-working-streaming"]
    errs = validate_selection(Selection.model_validate(without), m)
    assert any("skills-working-orchestration' is required" in e for e in errs)
    thin = {**selection_dict, "skill_items": {"skills-working-orchestration": ["Airflow"]}}
    errs = validate_selection(Selection.model_validate(thin), m)
    assert any("below its min_items (2)" in e for e in errs)


def test_required_is_satisfied_by_an_alternative(master_dict, selection_dict):
    md = _skill_rules(master_dict)
    md["skills"]["working"].append({"id": "skills-working-alt", "alt_of": ["skills-working-orchestration"],
                                    "category": {"en": "Other", "pt": "Outro"}, "items": ["Prefect"]})
    m = parse_master(md)
    sel = json.loads(json.dumps(selection_dict))
    sel["skill_category_order"]["working"] = ["skills-working-alt", "skills-working-streaming"]
    assert validate_selection(Selection.model_validate(sel), m) == []


def test_min_items_out_of_range_is_a_master_error(master_dict):
    with pytest.raises(MasterError, match="min_items must be between 1 and its 2 items"):
        parse_master(_skill_rules(master_dict, min_items=5))


def test_grammar_enforces_min_items_and_required(master_dict):
    md = _skill_rules(master_dict)
    md["skills"]["working"].append({"id": "skills-working-alt", "alt_of": ["skills-working-orchestration"],
                                    "category": {"en": "Other", "pt": "Outro"}, "items": ["Prefect"]})
    m = parse_master(md)
    s = choices.build_schema(m, build_selection_schema(m, count_limits=True), True)
    branches = s["properties"]["skills"]["anyOf"]
    with_req = [b for b in branches if "skills-working-orchestration" in b["properties"]["working"]["items"]["enum"]]
    without = [b for b in branches if b not in with_req]
    assert with_req and without
    assert all(b["properties"]["working"]["minItems"] == 1 for b in with_req)
    assert all(b["properties"]["working"]["minItems"] == 0 for b in without)
    items = with_req[0]["properties"]["items"]["properties"]["skills-working-orchestration"]
    # 2 items, min 2: one shape in which both are ranked 1 or more
    assert all(0 not in p["enum"] for p in items["properties"].values())


def test_ranks_min_count():
    r = choices._ranks(["a", "b", "c"], 3, at_least_one=False, min_count=2)
    assert len(r["anyOf"]) == 3
    for b in r["anyOf"]:
        assert sum(1 for p in b["properties"].values() if 0 not in p["enum"]) == 2
    one = choices._ranks(["a", "b"], 2, at_least_one=True)
    assert len(one["anyOf"]) == 2
    assert choices._ranks(["a", "b"], 2, at_least_one=False)["required"] == ["a", "b"]


def test_headline_options_only_when_the_title_names_the_role(master_dict):
    from cvtailor.master import gate_headlines
    m = parse_master(_headlines(master_dict))
    assert gate_headlines(m, "Data Engineer - Acme\nWe use dbt and analytics.").profile.headline_alts == []
    kept = gate_headlines(m, "Analytics Engineer (pleno) - Lumen\nModels in dbt.").profile.headline_alts
    assert [h.id for h in kept] == ["hl-analytics"]
    pt = gate_headlines(m, "Vaga: Engenheira de Analytics na Lumen").profile.headline_alts
    assert [h.id for h in pt] == ["hl-analytics"]
    late = "Data Engineer\n" + "x " * 300 + "Analytics Engineer"  # far below the title
    assert gate_headlines(m, late).profile.headline_alts == []
