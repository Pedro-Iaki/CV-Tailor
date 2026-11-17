"""Choice format for local models: bullet versions and skill layouts as single choices."""

import json

from cvtailor import choices
from cvtailor.config import Config
from cvtailor.llm import LLM, SELECTION_SYSTEM_PROMPT, build_selection_schema
from cvtailor.local_llm import LocalClient
from cvtailor.master import parse_master
from cvtailor.schema import LLMSelection
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
    acme = props["bullets"]["properties"]["job-acme"]["items"]["enum"]
    assert "acme-dbx-pipeline-01" in acme and not {"acme-dbx-short-01b", "acme-dbx-short-01c"} & set(acme)
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


def test_prompt_rewritten():
    p = choices.system_prompt(SELECTION_SYSTEM_PROMPT)
    assert "bullet_versions" in p and "skill_layouts" in p and "key_requirements" in p
    assert "either the categories on one side or the other" not in p


def _choice_answer(selection_dict, **over):
    d = dict(selection_dict)
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
    """A slot listed twice decodes to a duplicate id, which the validator rejects."""
    m = alt_master(master_dict)
    acme = ["acme-dbx-pipeline-01", "acme-dbx-pipeline-01"]
    bad = _choice_answer(selection_dict, bullets={**selection_dict["bullets"], "job-acme": acme})
    good = _choice_answer(selection_dict)
    http = FakeHTTP(monkeypatch, chat(json.dumps(bad)), chat(json.dumps(good)))
    llm = LLM("m", None, client=LocalClient("http://h/v1"), count_limits=True, choices=True)
    sel, _ = llm.select("job", m)
    assert isinstance(sel, LLMSelection) and len(http.requests) == 2
    retry = http.requests[1]["body"]["messages"][1]["content"]
    assert "'acme-dbx-short-01b' appears 2 times" in retry


def test_no_alternatives_keeps_plain_shape(master):
    s = choices.build_schema(master, build_selection_schema(master, count_limits=True), count_limits=True)
    assert "bullet_versions" not in s["properties"] and "skill_category_order" in s["properties"]
    assert "skill_layouts" not in json.loads(choices.build_index(master))


def test_config_and_context(monkeypatch, make_ctx):
    monkeypatch.setenv("CV_LLM_REASONING", "Low")
    assert Config.from_env().llm_reasoning == "low"
    ctx = make_ctx()
    ctx.llm = None
    ctx.cfg.llm_base_url = "http://localhost:11434/v1"
    ctx.cfg.llm_reasoning = "low"
    llm = ctx.get_llm()
    assert llm.choices and llm.client.reasoning == "low"
