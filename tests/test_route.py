import copy

import pytest

from cvtailor.route import route, score, selection_tags
from cvtailor.variants import load_variants, save_variant
from cvtailor.schema import Selection


def test_score_formula():
    s, sim, cov = score({"a", "b", "c", "d"}, {"a", "b", "e"}, {"a", "c"})
    assert sim == pytest.approx(2 / 5)
    assert cov == pytest.approx(2 / 4)
    assert s == pytest.approx(0.45)


@pytest.fixture
def variants(tmp_path, master, selection):
    vdir = tmp_path / "variants"
    tags = sorted(selection_tags(selection, master))
    save_variant(vdir, name="platform-v1", selection=selection, job_tags=tags, master=master,
                 source_application_id=1)
    other = copy.deepcopy(selection.model_dump())
    other["summary_id"] = "sum-analytics"
    other["project_ids"] = ["proj-dbt-pkg"]
    save_variant(vdir, name="analytics-v1", selection=Selection.model_validate(other),
                 job_tags=["analytics", "dbt", "sql", "data-modeling"], master=master,
                 source_application_id=2)
    return vdir, tags


def test_exact_match_reuses(master, variants):
    vdir, tags = variants
    rr = route(tags, load_variants(vdir, master), master, threshold=0.6)
    assert rr.decision == "reused" and rr.chosen.name == "platform-v1"
    assert rr.chosen.score == pytest.approx(1.0)
    assert rr.message.startswith("route: reused platform-v1 (1.00) | runner-up: analytics-v1")


def test_partial_overlap(master, variants):
    vdir, _ = variants
    rr = route(["dbt", "sql", "analytics", "kubernetes", "go-lang"], load_variants(vdir, master), master,
               threshold=0.6)
    assert rr.candidates[0].name == "analytics-v1"
    # sim 3/6 = .5 ; cov 3/5 = .6 -> .55 < .6
    assert rr.candidates[0].score == pytest.approx(0.55)
    assert rr.decision == "generated"
    assert "no match >= 0.60 (best analytics-v1 0.55)" in rr.message


def test_no_overlap(master, variants):
    vdir, _ = variants
    rr = route(["cobol", "mainframe", "jcl"], load_variants(vdir, master), master, threshold=0.6)
    assert rr.decision == "generated" and rr.candidates[0].score == 0


def test_too_few_tags(master, variants):
    vdir, _ = variants
    rr = route(["spark", "azure"], load_variants(vdir, master), master, threshold=0.0)
    assert rr.decision == "generated" and "skipping routing" in rr.message


def test_stale_variants_excluded(master_dict, variants):
    from cvtailor.master import parse_master
    vdir, tags = variants
    master_dict["jobs"][0]["bullets"] = [b for b in master_dict["jobs"][0]["bullets"]
                                         if b["id"] != "acme-cost-05"]  # in both variants
    m2 = parse_master(master_dict)
    vs = load_variants(vdir, m2)
    assert all(v.state == "stale" for v in vs)
    rr = route(tags, vs, m2, threshold=0.0)
    assert rr.decision == "generated" and rr.candidates == []
    with pytest.raises(ValueError, match="not eligible"):
        route(tags, vs, m2, threshold=0.6, force_variant="platform-v1")


def test_force_variant(master, variants):
    vdir, _ = variants
    rr = route(["cobol", "jcl", "mainframe"], load_variants(vdir, master), master, threshold=0.6,
               force_variant="analytics-v1")
    assert rr.decision == "reused" and rr.chosen.name == "analytics-v1"
    with pytest.raises(ValueError, match="does not exist"):
        route(["a", "b", "c"], load_variants(vdir, master), master, threshold=0.6, force_variant="nope")


def test_force_generate(master, variants):
    vdir, tags = variants
    rr = route(tags, load_variants(vdir, master), master, threshold=0.6, force_generate=True)
    assert rr.decision == "generated"
