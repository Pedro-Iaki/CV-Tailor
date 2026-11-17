"""Choice format for local models: bullet versions and skill layouts as single choices."""

import json
import re

import pytest

from cvtailor import choices
from cvtailor.config import Config
from cvtailor.llm import LLM, SELECTION_SYSTEM_PROMPT, LLMError, build_selection_schema
from cvtailor.local_llm import LocalClient
from cvtailor.master import parse_master
from cvtailor.editor import parse_annotated, to_annotated_yaml
from cvtailor.render import make_render_input, render_html
from cvtailor.schema import LLMSelection, Selection
from cvtailor.validate import validate_selection
from tests.test_alternatives import with_alternatives
from tests.test_local_llm import FakeHTTP, chat

GROUP = ["acme-dbx-pipeline-01", "acme-dbx-short-01b", "acme-dbx-short-01c"]


def alt_master(master_dict):
    return parse_master(with_alternatives(master_dict))


def test_analyse_groups_and_layouts(master_dict):
    ch = choices.analyse(alt_master(master_dict))
    assert ch.groups == {"acme-dbx-pipeline-01": GROUP}
    # the description alternative conflicts with a bullet but is not a bullet version
    assert all("lakehouse-tests-02" not in g for g in ch.groups.values())
    shapes = {(tuple(l["hands_on"]), tuple(l["working"]), tuple(l["languages"])) for l in ch.layouts}
    free = ("skills-languages", "skills-modeling")
    assert shapes == {
        (("skills-processing", "skills-platform", *free), ("skills-working-orchestration", "skills-working-streaming"), (lang,))
        for lang in ("langs", "langs-en-first")
    } | {
        (("skills-platform", *free), ("skills-working-orchestration", "skills-working-streaming", "skills-working-flat"), (lang,))
        for lang in ("langs", "langs-en-first")
    } | {
        (("skills-languages", "skills-modeling", "skills-flat"), ("skills-working-orchestration", "skills-working-streaming", "skills-working-flat"), (lang,))
        for lang in ("langs", "langs-en-first")
    }
    # no layout ever holds two conflicting categories
    for l in ch.layouts:
        ids = [c for t in l.values() for c in t]
        assert not ({"skills-flat", "skills-processing"} <= set(ids))


def test_merge_sources_are_not_grouped(master_dict):
    md = master_dict
    md["jobs"][0]["bullets"].append({"id": "acme-merged-07", "alt_of": ["acme-delta-02", "acme-adf-03"],
                                     "tags": ["delta"], "en": "Merged.", "pt": "Mesclado."})
    ch = choices.analyse(parse_master(md))
    assert ch.groups == {}  # delta-02 and adf-03 may still go together


def test_schema(master_dict):
    m = alt_master(master_dict)
    plain = build_selection_schema(m, count_limits=True)
    s = choices.build_schema(m, plain, count_limits=True)
    props = s["properties"]
    assert list(props)[:2] == ["key_requirements", "summary_id"] and s["required"] == list(props)
    # Jobs: one rank per bullet (keys can't repeat), ranks capped at max_bullets_per_job
    # (5), and one anyOf branch per bullet that requires it, so at least one is picked.
    # Acme has 6 slots and a maximum of 5: every branch forces one slot to 0 and
    # requires another to be ranked, so 1 to 5 bullets are always picked.
    branches = props["bullets"]["properties"]["job-acme"]["anyOf"]
    acme = branches[0]["required"]
    assert "acme-dbx-pipeline-01" in acme and not {"acme-dbx-short-01b", "acme-dbx-short-01c"} & set(acme)
    assert len(acme) == 6 and len(branches) == 6 * 5
    shapes = set()
    for branch in branches:
        assert branch["additionalProperties"] is False and branch["required"] == acme
        enums = {k: tuple(v["enum"]) for k, v in branch["properties"].items()}
        zero = [k for k, e in enums.items() if e == (0,)]
        picked = [k for k, e in enums.items() if e == (1, 2, 3, 4, 5)]
        assert len(zero) == 1 and len(picked) == 1 and zero != picked
        shapes.add((zero[0], picked[0]))
    assert len(shapes) == 30
    # A job with fewer slots than the maximum only needs the "at least one" branches
    intern = props["bullets"]["properties"]["job-intern"]["anyOf"]
    assert len(intern) == 2

    # Projects: rank + bullets + description; hidden means every bullet 0 and default
    assert "project_ids" not in props and "proj-airflow-demo" not in props["bullets"]["properties"]
    off, shown_default, shown_alt = props["projects"]["properties"]["proj-airflow-demo"]["anyOf"]
    assert off["properties"]["rank"]["enum"] == [0] and shown_default["properties"]["rank"]["enum"] == [1, 2]
    assert all(v["enum"] == [0] for v in off["properties"]["bullets"]["properties"].values())
    assert off["properties"]["description"]["enum"] == ["default"]
    # one shown shape per description; the alternative covers lakehouse-tests-02, forced to 0
    assert shown_default["properties"]["description"]["enum"] == ["default"]
    assert shown_default["properties"]["bullets"]["properties"]["lakehouse-tests-02"]["enum"] == [0, 1, 2]
    assert shown_alt["properties"]["description"]["enum"] == ["lakehouse-desc-short"]
    assert shown_alt["properties"]["bullets"]["properties"]["lakehouse-tests-02"]["enum"] == [0]
    assert shown_alt["properties"]["bullets"]["properties"]["lakehouse-dags-01"]["enum"] == [0, 1, 2]
    assert "descriptions" not in props  # only projects had alternatives here
    no_bullets = props["projects"]["properties"]["proj-kafka-lab"]["anyOf"][1]["properties"]
    assert set(no_bullets) == {"rank"}
    assert props["bullet_versions"]["properties"] == {"acme-dbx-pipeline-01": {"type": "string", "enum": GROUP}}
    assert "skill_category_order" not in props
    branches = props["skills"]["anyOf"]
    assert len(branches) == 6
    first = branches[0]["properties"]
    assert first["layout"]["enum"] == ["layout-1"]
    assert first["languages"]["maxItems"] == 1
    assert plain["properties"]["bullets"]["properties"]["job-acme"]["items"]["enum"] != acme  # plain untouched


def test_index_shows_versions_and_layouts(master_dict):
    m = alt_master(master_dict)
    idx = json.loads(choices.build_index(m))
    assert list(idx)[0] == "limits"
    acme = idx["jobs"][0]["bullets"]
    ids = [b["id"] for b in acme]
    assert ids.count("acme-dbx-pipeline-01") == 1 and "acme-dbx-short-01b" not in ids
    entry = next(b for b in acme if b["id"] == "acme-dbx-pipeline-01")
    assert [v["id"] for v in entry["versions"]] == GROUP
    assert all("never_with" not in v for v in entry["versions"])  # only conflicts within the group
    assert [l["id"] for l in idx["skill_layouts"]] == [f"layout-{i}" for i in range(1, 7)]
    assert all("never_with" not in c for c in idx["skill_categories"])
    proj = idx["projects"][0]
    tests02 = next(b for b in proj["bullets"] if b["id"] == "lakehouse-tests-02")
    assert tests02["never_with"] == ["lakehouse-desc-short"]  # other conflicts still shown


def test_decode_maps_slots_and_layout():
    out = choices.decode({
        "key_requirements": ["Databricks"],
        "summary_id": "s",
        "bullet_versions": {"acme-dbx-pipeline-01": "acme-dbx-short-01b"},
        "bullets": {"job-acme": ["acme-delta-02", "acme-dbx-pipeline-01"], "job-x": None},
        "skills": {"layout": "layout-2", "hands_on": ["a"], "working": [], "familiar": [], "languages": ["b"]},
    })
    assert out == {"summary_id": "s",
                   "bullets": {"job-acme": ["acme-delta-02", "acme-dbx-short-01b"], "job-x": []},
                   "skill_category_order": {"hands_on": ["a"], "working": [], "familiar": [], "languages": ["b"]}}


def test_decode_ranks():
    out = choices.decode({
        "summary_id": "s",
        "bullet_versions": {"acme-dbx-pipeline-01": "acme-dbx-short-01c"},
        "bullets": {"job-acme": {"acme-dbx-pipeline-01": 2, "acme-delta-02": 0, "acme-adf-03": 1,
                                 "acme-cicd-04": 3, "acme-cost-05": 3},   # tie: keeps listed order
                    "proj-x": {"a": 0, "b": 0}},
    })
    assert out["bullets"] == {"job-acme": ["acme-adf-03", "acme-dbx-short-01c", "acme-cicd-04", "acme-cost-05"],
                              "proj-x": []}


def test_decode_projects():
    out = choices.decode({
        "summary_id": "s",
        "bullet_versions": {},
        "projects": {"proj-a": {"rank": 2, "bullets": {"a1": 1, "a2": 0}, "description": "a-alt"},
                     "proj-b": {"rank": 0, "bullets": {"b1": 0}, "description": "default"},
                     "proj-c": {"rank": 1}},
        "bullets": {"job-x": {"x1": 1}},
        "descriptions": {"job-x": "x-alt"},
    })
    assert out["project_ids"] == ["proj-c", "proj-a"]
    assert out["bullets"] == {"job-x": ["x1"], "proj-a": ["a1"]}  # hidden project: no key
    assert out["descriptions"] == {"job-x": "x-alt", "proj-a": "a-alt"}


def test_prompt_rewritten():
    p = choices.system_prompt(SELECTION_SYSTEM_PROMPT)
    assert "bullet_versions" in p and "skill_layouts" in p and "key_requirements" in p
    assert "either the categories on one side or the other" not in p
    assert "empty list" not in p and "0 leaves it out" in p


def _choice_answer(selection_dict, **over):
    d = dict(selection_dict)
    d["bullets"] = {o: {b: i + 1 for i, b in enumerate(ids)} for o, ids in d["bullets"].items()}  # ranks
    order = d.pop("skill_category_order")
    d.update(key_requirements=["Spark"], bullet_versions={"acme-dbx-pipeline-01": "acme-dbx-short-01b"},
             skills={"layout": "layout-1", **order, "languages": ["langs"]},
             descriptions={"proj-airflow-demo": "default"}, unmatched_requirements=[], suggestions=[])
    d.update(over)
    return d


def test_select_end_to_end(monkeypatch, master_dict, selection_dict):
    m = alt_master(master_dict)
    http = FakeHTTP(monkeypatch, chat(json.dumps(_choice_answer(selection_dict))))
    llm = LLM("m", None, client=LocalClient("http://h/v1", reasoning="low"), count_limits=True, choices=True)
    sel, _ = llm.select("Spark job", m)
    assert sel.bullets["job-acme"][0] == "acme-dbx-short-01b"  # version picked for the slot
    assert sel.skill_category_order.languages == ["langs"]
    assert validate_selection(sel.core(), m) == []
    body = http.requests[0]["body"]
    assert body["reasoning_effort"] == "low"
    assert "skill_layouts" in body["messages"][0]["content"]
    assert "anyOf" in json.dumps(body["response_format"])


def test_decoded_answer_is_still_validated(monkeypatch, master_dict, selection_dict):
    """Ranks decode to a plain selection that is validated as usual (here: too many bullets)."""
    m = alt_master(master_dict)
    acme = {b: i + 1 for i, b in enumerate(["acme-dbx-pipeline-01", "acme-delta-02", "acme-adf-03",
                                            "acme-cicd-04", "acme-cost-05", "acme-streaming-06"])}
    bad = _choice_answer(selection_dict, bullets={**selection_dict["bullets"], "job-acme": acme})
    good = _choice_answer(selection_dict)
    http = FakeHTTP(monkeypatch, chat(json.dumps(bad)), chat(json.dumps(good)))
    llm = LLM("m", None, client=LocalClient("http://h/v1"), count_limits=True, choices=True)
    sel, _ = llm.select("job", m)
    assert isinstance(sel, LLMSelection) and len(http.requests) == 2
    retry = http.requests[1]["body"]["messages"][1]["content"]
    assert "job-acme: 6 bullets, above max_bullets_per_job (5)" in retry


def test_truncated_answer_is_retried_once(monkeypatch, master_dict, selection_dict):
    m = alt_master(master_dict)
    good = chat(json.dumps(_choice_answer(selection_dict)))
    http = FakeHTTP(monkeypatch, chat("", finish="length", usage=(4000, 4096)), good)
    llm = LLM("m", None, client=LocalClient("http://h/v1"), count_limits=True, choices=True, max_output=4096)
    sel, usage = llm.select("job", m)
    assert usage.output_tokens == 4096 + 30
    assert all(r["body"]["max_tokens"] == 4096 for r in http.requests)  # 16000 capped
    retry = http.requests[1]["body"]["messages"][1]["content"]
    assert "max_tokens before finishing the JSON (4096 output tokens)" in retry
    assert "Keep your reasoning short" in retry

    FakeHTTP(monkeypatch, chat("", finish="length"), chat("", finish="length"))
    with pytest.raises(LLMError, match="max_tokens"):  # twice: the run stops
        llm.select("job", m)


def test_no_alternatives_keeps_plain_shape(master):
    s = choices.build_schema(master, build_selection_schema(master, count_limits=True), count_limits=True)
    assert "bullet_versions" not in s["properties"] and "skill_category_order" in s["properties"]
    assert "skill_layouts" not in json.loads(choices.build_index(master))


# ---------------------------------------------------------------- skill items

def test_skill_items_render_and_validate(master, selection_dict):
    sel = Selection.model_validate({**selection_dict, "skill_items": {
        "skills-modeling": ["Dimensional modeling", "dbt"],          # reordered, PostgreSQL dropped
        "skills-working-streaming": ["Spark Structured Streaming"]}})
    assert validate_selection(sel, master) == []
    en = render_html(make_render_input(master, sel, "en"))
    lines = dict(re.findall(r"<li><strong>(.*?):</strong>\s*(.*?)</li>", en))
    assert lines["Modeling"] == "Dimensional modeling, dbt"
    assert lines["Working knowledge"] == "Airflow, Power BI, Spark Structured Streaming"  # Kafka dropped
    assert lines["Processing"] == "PySpark, Spark SQL, Delta Lake, pandas"  # no skill_items: unchanged
    pt = render_html(make_render_input(master, sel, "pt"))
    assert "Modelagem dimensional, dbt" in pt  # chosen by English text, rendered in PT


def test_skill_items_rules(master, selection_dict):
    sel = Selection.model_validate({**selection_dict, "skill_items": {
        "skills-modeling": ["dbt", "PySpark", "dbt"],     # PySpark belongs to skills-processing
        "skills-platform": [],                            # empty: leave the category out instead
        "skills-working-flat": ["Airflow"],               # no such category in this master
        "sum-platform": ["x"]}})
    sel.skill_category_order.hands_on.remove("skills-platform")
    sel.skill_items["skills-platform"] = ["Azure"]          # category not selected
    errors = "\n".join(validate_selection(sel, master))
    assert "'PySpark' is not an item of skills-modeling" in errors
    assert "'dbt' appears 2 times in skills-modeling" in errors
    assert "category 'skills-platform' is not in skill_category_order" in errors
    assert "'skills-working-flat' is not a skill category" in errors
    assert "'sum-platform' is not a skill category" in errors
    empty = Selection.model_validate({**selection_dict, "skill_items": {"skills-modeling": None}})
    assert any("lists no items" in e for e in validate_selection(empty, master))


def test_skill_items_editor_round_trip(master, selection_dict):
    sel = Selection.model_validate({**selection_dict, "skill_items": {"skills-modeling": ["Dimensional modeling", "dbt"]}})
    text = to_annotated_yaml(sel, master)
    assert '    - "Dimensional modeling"' in text and '    # - "PostgreSQL"' in text
    assert '  # skills-processing:' in text and '  #   - "PySpark"' in text
    assert parse_annotated(text) == sel
    plain = Selection.model_validate(selection_dict)
    assert parse_annotated(to_annotated_yaml(plain, master)) == plain


def test_skill_items_in_schema_and_decode(master_dict):
    m = alt_master(master_dict)
    s = choices.build_schema(m, build_selection_schema(m, count_limits=True), count_limits=True)
    branch = next(b for b in s["properties"]["skills"]["anyOf"]
                  if "skills-flat" in b["properties"]["hands_on"]["items"]["enum"])
    items = branch["properties"]["items"]["properties"]
    flat = items["skills-flat"]
    assert flat["required"] == ["PySpark", "Databricks", "Azure"]  # its own items only, each once
    assert flat["properties"]["Azure"]["enum"] == [0, 1, 2, 3]
    assert "skills-processing" not in items  # not part of this layout

    out = choices.decode({"summary_id": "s", "bullets": {}, "skills": {
        "layout": "layout-5", "hands_on": ["skills-flat"], "working": [], "familiar": [], "languages": [],
        "items": {"skills-flat": {"PySpark": 2, "Databricks": 0, "Azure": 1},   # kept: a real choice
                  "skills-modeling": {"dbt": 1},                                # dropped: category not listed
                  "skills-languages": {"Python": 1, "SQL": 2}}}}, m)            # dropped: same as master
    assert out["skill_items"] == {"skills-flat": ["Azure", "PySpark"]}


def test_config_and_context(monkeypatch, make_ctx):
    monkeypatch.setenv("CV_LLM_REASONING", "Low")
    monkeypatch.setenv("CV_LLM_MAX_TOKENS", "6000")
    cfg = Config.from_env()
    assert (cfg.llm_reasoning, cfg.llm_max_tokens) == ("low", 6000)
    monkeypatch.setenv("CV_LLM_MAX_TOKENS", "lots")
    assert Config.from_env().llm_max_tokens == 4096
    ctx = make_ctx()
    ctx.llm = None
    ctx.cfg.llm_base_url = "http://localhost:11434/v1"
    ctx.cfg.llm_reasoning = "low"
    llm = ctx.get_llm()
    assert llm.choices and llm.client.reasoning == "low" and llm.max_output == 4096
