"""Validated alternatives in the master: alt_of, summary headlines, description
alternatives, education detail lines and the languages tier."""

import json
import re

import pytest

from cvtailor.editor import parse_annotated, to_annotated_yaml
from cvtailor.llm import build_selection_schema
from cvtailor.master import MasterError, build_llm_index, check_master, parse_master
from cvtailor.render import make_render_input, render_html
from cvtailor.schema import Selection
from cvtailor.validate import validate_selection

ALT_BULLET = {"id": "acme-dbx-short-01b", "alt_of": "acme-dbx-pipeline-01", "tags": ["databricks"],
              "en": "Built Databricks pipelines for payment events.", "pt": "Construí pipelines no Databricks para eventos de pagamento."}


def with_alternatives(md: dict) -> dict:
    acme = md["jobs"][0]
    acme["bullets"].append(dict(ALT_BULLET))
    acme["bullets"].append(dict(ALT_BULLET, id="acme-dbx-short-01c", en="Other version.", pt="Outra versão."))
    md["summaries"][0]["headline"] = {"en": "Platform Data Engineer", "pt": "Engenheira de Dados de Plataforma"}
    md["projects"][0]["description_alts"] = [{
        "id": "lakehouse-desc-short", "alt_of": "lakehouse-tests-02", "tags": ["airflow"],
        "en": "Lakehouse demo with tested DAGs.", "pt": "Demo de lakehouse com DAGs testadas."}]
    md["education"][0]["bullets"] = [{"id": "bsc-coursework-01", "tags": ["sql"],
                                      "en": "Coursework: databases.", "pt": "Disciplinas: bancos de dados."}]
    md["skills"]["hands_on"].append({  # the same skills in one line, an alternative layout
        "id": "skills-flat", "alt_of": ["skills-processing", "skills-platform"],
        "category": {"en": "Hands-on", "pt": "Prática"},
        "items": ["PySpark", "Databricks", "Azure"]})
    md["skills"]["working"].append({  # the other half of the flat layout
        "id": "skills-working-flat", "alt_of": "skills-processing",
        "category": {"en": "Working", "pt": "Funcional"}, "items": ["pandas"]})
    md["skills"]["languages"] = [
        {"id": "langs", "category": {"en": "Languages", "pt": "Idiomas"}, "items": ["Portuguese", "English"]},
        {"id": "langs-en-first", "alt_of": "langs", "category": {"en": "Languages", "pt": "Idiomas"},
         "items": ["English", "Portuguese"]}]
    md["limits"]["max_total_bullets"] = 14
    return md


def sel_with(selection_dict, **changes) -> Selection:
    d = json.loads(json.dumps(selection_dict))
    for k, v in changes.items():
        if k == "bullets":
            d["bullets"].update(v)
        else:
            d[k] = v
    return Selection.model_validate(d)


def test_master_with_alternatives_is_valid(master_dict):
    m = parse_master(with_alternatives(master_dict))
    assert check_master(m) == []  # skills repeat only between alternatives


def test_alt_of_targets_are_checked(master_dict):
    md = with_alternatives(master_dict)
    md["jobs"][0]["bullets"][-1]["alt_of"] = "nope"
    md["jobs"][1]["bullets"][0]["alt_of"] = "acme-delta-02"          # bullet of another job
    md["skills"]["working"][-1]["alt_of"] = "sum-platform"           # not a skill category
    md["projects"][0]["description_alts"][0]["alt_of"] = "acme-delta-02"
    with pytest.raises(MasterError) as e:
        parse_master(md)
    errors = "\n".join(e.value.errors)
    assert "alt_of 'nope' does not exist" in errors
    assert "alt_of 'acme-delta-02' belongs to job-acme, not job-datacorp" in errors
    assert "alt_of 'sum-platform' is a summary, not a skill category" in errors
    assert "alt_of 'acme-delta-02' must be a bullet of proj-airflow-demo" in errors


def test_duplicate_skill_outside_alternatives_still_rejected(master_dict):
    md = with_alternatives(master_dict)
    md["skills"]["hands_on"][-1]["alt_of"] = ["skills-processing"]  # Databricks/Azure clash with skills-platform
    with pytest.raises(MasterError) as e:
        parse_master(md)
    assert any("'Databricks' appears more than once" in x for x in e.value.errors)


def test_alternatives_never_selected_together(master_dict, selection_dict):
    m = parse_master(with_alternatives(master_dict))
    acme = selection_dict["bullets"]["job-acme"]
    both = sel_with(selection_dict, bullets={"job-acme": [*acme[:4], "acme-dbx-short-01b"]})
    assert any("'acme-dbx-pipeline-01' and 'acme-dbx-short-01b' are versions of the same item" in e
               for e in validate_selection(both, m))
    siblings = sel_with(selection_dict, bullets={"job-acme": ["acme-dbx-short-01b", "acme-dbx-short-01c"]})
    assert any("are versions of the same item" in e for e in validate_selection(siblings, m))
    swapped = sel_with(selection_dict, bullets={"job-acme": ["acme-dbx-short-01b", *acme[1:]]})
    assert validate_selection(swapped, m) == []


def test_skill_layouts(master_dict, selection_dict):
    m = parse_master(with_alternatives(master_dict))
    order = selection_dict["skill_category_order"]
    mixed = sel_with(selection_dict, skill_category_order={**order, "hands_on": ["skills-flat", "skills-platform"]})
    assert any("are versions of the same item" in e for e in validate_selection(mixed, m))
    flat = sel_with(selection_dict, skill_category_order={
        "hands_on": ["skills-flat", "skills-languages", "skills-modeling"],
        "working": ["skills-working-flat"], "familiar": [], "languages": ["langs-en-first"]})
    assert validate_selection(flat, m) == []  # two halves of one layout may go together
    html = render_html(make_render_input(m, flat, "en"))
    labels = re.findall(r"<li><strong>(.*?):</strong>", html)
    assert labels == ["Hands-on", "Languages", "Modeling", "Working knowledge", "Languages"]
    assert "English, Portuguese" in html


def test_description_alternative_and_headline(master_dict, selection_dict):
    m = parse_master(with_alternatives(master_dict))
    sel = sel_with(selection_dict, descriptions={"proj-airflow-demo": "lakehouse-desc-short"},
                   bullets={"proj-airflow-demo": ["lakehouse-dags-01"]})
    assert validate_selection(sel, m) == []
    html = render_html(make_render_input(m, sel, "en"))
    assert "Lakehouse demo with tested DAGs." in html and "End-to-end lakehouse demo" not in html
    assert '<p class="headline">Platform Data Engineer</p>' in html  # from the summary
    pt = render_html(make_render_input(m, sel, "pt"))
    assert "Demo de lakehouse com DAGs testadas." in pt and "Engenheira de Dados de Plataforma" in pt

    covered = sel_with(selection_dict, descriptions={"proj-airflow-demo": "lakehouse-desc-short"},
                       bullets={"proj-airflow-demo": ["lakehouse-tests-02"]})
    assert any("are versions of the same item" in e for e in validate_selection(covered, m))
    wrong = sel_with(selection_dict, descriptions={"proj-airflow-demo": "nope", "proj-kafka-lab": "lakehouse-desc-short"})
    errors = "\n".join(validate_selection(wrong, m))
    assert "'nope' is not a description alternative of proj-airflow-demo" in errors
    assert "'lakehouse-desc-short' is not a description alternative of proj-kafka-lab" in errors
    unselected = sel_with(selection_dict, project_ids=["proj-kafka-lab"], bullets={},
                          descriptions={"proj-airflow-demo": "lakehouse-desc-short"})
    assert any("project 'proj-airflow-demo' is not in project_ids" in e for e in validate_selection(unselected, m))
    assert Selection.model_validate({**selection_dict, "descriptions": {"proj-airflow-demo": "default"}}).descriptions == {}


def test_education_details(master_dict, selection_dict):
    m = parse_master(with_alternatives(master_dict))
    sel = sel_with(selection_dict, bullets={"edu-bsc": ["bsc-coursework-01"]})
    assert validate_selection(sel, m) == []
    html = render_html(make_render_input(m, sel, "en"))
    assert '<p class="edu-detail">Coursework: databases.</p>' in html
    assert "Coursework" not in render_html(make_render_input(m, Selection.model_validate(selection_dict), "en"))


def test_llm_schema_and_index(master_dict):
    m = parse_master(with_alternatives(master_dict))
    s = build_selection_schema(m, count_limits=True)["properties"]
    assert s["descriptions"]["properties"]["proj-airflow-demo"]["enum"] == ["default", "lakehouse-desc-short"]
    assert s["bullets"]["properties"]["edu-bsc"]["items"]["enum"] == ["bsc-coursework-01"]
    assert s["skill_category_order"]["properties"]["languages"]["items"]["enum"] == ["langs", "langs-en-first"]
    m.skills.familiar = []
    s = build_selection_schema(m, count_limits=True)["properties"]
    assert s["skill_category_order"]["properties"]["familiar"]["maxItems"] == 0  # empty tier

    idx = json.loads(build_llm_index(m))
    acme = {b["id"]: b for b in idx["jobs"][0]["bullets"]}
    # the exclusion is visible from both sides, including between siblings
    assert acme["acme-dbx-pipeline-01"]["never_with"] == ["acme-dbx-short-01b", "acme-dbx-short-01c"]
    assert acme["acme-dbx-short-01c"]["never_with"] == ["acme-dbx-pipeline-01", "acme-dbx-short-01b"]
    assert idx["summaries"][0]["headline"] == "Platform Data Engineer"
    cats = {c["id"]: c for c in idx["skill_categories"]}
    assert "skills-working-flat" not in cats["skills-flat"].get("never_with", [])


def test_editor_round_trip_with_new_fields(master_dict, selection_dict):
    m = parse_master(with_alternatives(master_dict))
    sel = sel_with(selection_dict, descriptions={"proj-airflow-demo": "lakehouse-desc-short"},
                   bullets={"proj-airflow-demo": ["lakehouse-dags-01"], "edu-bsc": ["bsc-coursework-01"]})
    text = to_annotated_yaml(sel, m)
    assert "[alt of acme-dbx-pipeline-01]" in text
    assert "# proj-airflow-demo: default" in text
    assert parse_annotated(text) == sel
