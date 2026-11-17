"""One passing and one failing fixture per master validation rule."""

import copy

import pytest

from cvtailor.master import MasterError, parse_master


def errors_for(d):
    try:
        parse_master(d)
    except MasterError as e:
        return e.errors
    return []


def test_example_master_passes(master_dict):
    assert errors_for(master_dict) == []


# 1. unique ids
def test_rule1_unique_ids_fail(master_dict):
    master_dict["projects"][0]["id"] = "acme-delta-02"  # same as a bullet
    errs = errors_for(master_dict)
    assert any("duplicate id 'acme-delta-02'" in e for e in errs)


def test_rule1_unique_ids_pass_with_distinct_ids(master_dict):
    master_dict["projects"][0]["id"] = "proj-renamed"
    assert errors_for(master_dict) == []


# 2. non-empty en/pt
def test_rule2_empty_text_fail(master_dict):
    master_dict["jobs"][0]["bullets"][0]["pt"] = "  "
    master_dict["jobs"][0]["title"]["en"] = ""
    errs = errors_for(master_dict)
    assert any("bullet acme-dbx-pipeline-01: 'pt' is empty" in e for e in errs)
    assert any("job job-acme title: 'en' is empty" in e for e in errs)


def test_rule2_bilingual_skill_item_pass(master_dict):
    master_dict["skills"]["familiar"][0]["items"].append({"en": "Data mesh", "pt": "Data mesh"})
    assert errors_for(master_dict) == []


# 3. skill tiers and duplicates
def test_rule3_skill_in_two_tiers_fail(master_dict):
    master_dict["skills"]["working"][0]["items"].append("pyspark")  # case-insensitive dup of hands_on
    errs = errors_for(master_dict)
    assert any("more than one tier" in e for e in errs)


def test_rule3_skill_twice_same_tier_fail(master_dict):
    master_dict["skills"]["hands_on"][1]["items"].append("Python")
    errs = errors_for(master_dict)
    assert any("more than once" in e for e in errs)


def test_rule3_unknown_tier_fail(master_dict):
    master_dict["skills"]["expert"] = []
    errs = errors_for(master_dict)
    assert any("expert" in e for e in errs)


def test_rule3_pass(master_dict):
    master_dict["skills"]["working"][0]["items"].append("Spark Streaming UI")
    assert errors_for(master_dict) == []


# 4. tags
@pytest.mark.parametrize("bad", ["Spark", "spark_sql", "-spark", "spark--sql", "spark sql"])
def test_rule4_bad_tag_fail(master_dict, bad):
    master_dict["jobs"][0]["bullets"][0]["tags"] = [bad]
    assert any("invalid tag" in e for e in errors_for(master_dict))


def test_rule4_empty_tags_fail(master_dict):
    master_dict["summaries"][0]["tags"] = []
    assert any("tags list is empty" in e for e in errors_for(master_dict))


def test_rule4_good_tags_pass(master_dict):
    master_dict["jobs"][0]["bullets"][0]["tags"] = ["spark", "azure-data-factory", "k8s"]
    assert errors_for(master_dict) == []


# 5. dates
def test_rule5_bad_format_fail(master_dict):
    master_dict["jobs"][0]["start"] = "01/2023"
    assert any("not YYYY-MM or YYYY" in e for e in errors_for(master_dict))


def test_rule5_end_before_start_fail(master_dict):
    master_dict["jobs"][1]["end"] = "2020-01"
    assert any("earlier than start" in e for e in errors_for(master_dict))


def test_rule5_year_only_and_unquoted_pass(master_dict):
    master_dict["education"][0]["start"] = 2017  # unquoted YAML int
    master_dict["education"][0]["end"] = None    # in progress
    assert errors_for(master_dict) == []


def test_rule5_end_null_not_allowed_on_start(master_dict):
    master_dict["jobs"][0]["start"] = None
    assert errors_for(master_dict)


# 6. tag aliases
def test_rule6_alias_for_unused_tag_fail(master_dict):
    master_dict["tag_aliases"]["golang"] = ["go lang"]
    assert any("'golang' is not a tag used" in e for e in errors_for(master_dict))


def test_rule6_alias_reused_fail(master_dict):
    master_dict["tag_aliases"]["azure"] = ["adf"]  # already an alias of azure-data-factory
    assert any("alias 'adf' is used by both" in e for e in errors_for(master_dict))


def test_rule6_pass(master_dict):
    master_dict["tag_aliases"]["azure"] = ["microsoft azure"]
    assert errors_for(master_dict) == []


# 7. limits
def test_rule7_min_greater_than_max_fail(master_dict):
    master_dict["limits"]["min_bullets_per_job"] = 6
    errs = errors_for(master_dict)
    assert any("min_bullets_per_job is greater than max_bullets_per_job" in e for e in errs)


def test_rule7_total_too_small_fail(master_dict):
    master_dict["limits"]["min_bullets_per_job"] = 2
    master_dict["limits"]["max_total_bullets"] = 5
    assert any("exceeds max_total_bullets" in e for e in errors_for(master_dict))


def test_rule7_job_with_too_few_bullets_fail(master_dict):
    master_dict["limits"]["min_bullets_per_job"] = 3
    assert any("job job-intern has 2 bullets" in e for e in errors_for(master_dict))


def test_rule7_non_positive_fail(master_dict):
    master_dict["limits"]["max_pages"] = 0
    assert any("limits.max_pages" in e for e in errors_for(master_dict))


def test_rule7_pass(master_dict):
    master_dict["limits"]["min_bullets_per_job"] = 2
    assert errors_for(master_dict) == []


def test_all_errors_reported_at_once(master_dict):
    d = copy.deepcopy(master_dict)
    d["jobs"][0]["start"] = "bad"
    d["summaries"][0]["tags"] = ["Bad Tag"]
    d["limits"]["max_pages"] = 0
    assert len(errors_for(d)) >= 3


# new format fields
def test_project_needs_description_or_bullets(master_dict):
    master_dict["projects"][1].pop("description")
    assert any("proj-dbt-pkg: needs a description or bullets" in e for e in errors_for(master_dict))


def test_project_bullet_ids_and_tags_checked(master_dict):
    b = master_dict["projects"][0]["bullets"][0]
    b["id"] = "acme-delta-02"
    b["tags"] = ["Bad Tag"]
    errs = errors_for(master_dict)
    assert any("duplicate id 'acme-delta-02'" in e for e in errs)
    assert any("invalid tag 'Bad Tag'" in e for e in errs)


def test_expected_needs_end_and_layout_unique(master_dict):
    master_dict["education"][0].update({"end": None, "expected": True})
    master_dict["layout"] = {"sections": ["summary", "skills", "skills"]}
    errs = errors_for(master_dict)
    assert any("'expected' needs an end date" in e for e in errs)
    assert any("'skills' is listed more than once" in e for e in errs)


def test_layout_rejects_unknown_section(master_dict):
    master_dict["layout"] = {"sections": ["summary", "hobbies"]}
    assert any(e.startswith("layout.sections") for e in errors_for(master_dict))


def test_empty_text_fields(master_dict):
    master_dict["profile"]["headline"] = {"en": "Data Engineer", "pt": ""}
    master_dict["profile"]["links"][0]["label"] = " "
    errs = errors_for(master_dict)
    assert any("profile headline: 'pt' is empty" in e for e in errs)
    assert any("profile link 1 label: empty" in e for e in errs)
