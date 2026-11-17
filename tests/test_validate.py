import copy

from cvtailor.schema import Selection
from cvtailor.validate import validate_selection


def errs(master, d):
    return validate_selection(Selection.model_validate(d), master)


def test_valid_example(master, selection_dict):
    assert errs(master, selection_dict) == []


def test_unknown_id(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-acme"][0] = "made-up-bullet"
    d["project_ids"] = ["proj-nope"]
    e = errs(master, d)
    assert any("made-up-bullet" in x and "does not exist" in x for x in e)
    assert any("proj-nope" in x for x in e)


def test_bullet_under_wrong_job(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-intern"] = ["acme-cost-05"]
    d["bullets"]["job-acme"].remove("acme-cost-05")
    assert any("belongs to job-acme, not job-intern" in x for x in errs(master, d))


def test_missing_job(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    del d["bullets"]["job-intern"]
    assert any("job 'job-intern' is missing" in x for x in errs(master, d))


def test_below_min_and_above_max(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-intern"] = []
    d["bullets"]["job-acme"].append("acme-streaming-06")  # 6 > max 5
    e = errs(master, d)
    assert any("below min_bullets_per_job" in x for x in e)
    assert any("above max_bullets_per_job" in x for x in e)


def test_tier_change(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["skill_category_order"]["hands_on"].append("skills-familiar-other")
    d["skill_category_order"]["familiar"] = []
    assert any("belongs to tier familiar, not hands_on" in x for x in errs(master, d))


def test_over_limit_totals(master_dict, selection_dict):
    from cvtailor.master import parse_master
    master_dict["limits"]["max_total_bullets"] = 10
    master = parse_master(master_dict)
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-datacorp"].append("datacorp-bi-04")
    d["bullets"]["job-intern"].append("intern-scraping-02")  # 5 + 4 + 2 = 11 > 10
    d["project_ids"] = ["proj-airflow-demo", "proj-kafka-lab", "proj-dbt-pkg"]
    e = errs(master, d)
    assert any("above max_total_bullets" in x for x in e)
    assert any("above max_projects" in x for x in e)


def test_duplicate_ids(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["bullets"]["job-acme"][1] = d["bullets"]["job-acme"][0]
    d["project_ids"] = ["proj-kafka-lab", "proj-kafka-lab"]
    e = errs(master, d)
    assert any("'acme-dbx-pipeline-01' appears 2 times" in x for x in e)
    assert any("'proj-kafka-lab' appears 2 times" in x for x in e)


def test_summary_must_be_summary(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["summary_id"] = "proj-kafka-lab"
    assert any("not a summary" in x for x in errs(master, d))


def test_reports_all_failures(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["summary_id"] = "nope"
    del d["bullets"]["job-intern"]
    d["project_ids"] = ["x"]
    assert len(errs(master, d)) >= 3


def test_project_bullets(master, selection_dict):
    d = copy.deepcopy(selection_dict)
    d["bullets"]["proj-airflow-demo"] = ["lakehouse-tests-02", "lakehouse-dags-01"]
    assert errs(master, d) == []  # proj-airflow-demo is in the example's project_ids
    d["project_ids"] = ["proj-kafka-lab"]
    assert any("has bullets but is not in project_ids" in x for x in errs(master, d))
    d["project_ids"] = ["proj-airflow-demo"]
    d["bullets"]["proj-airflow-demo"] = ["acme-cost-05"]
    d["bullets"]["job-acme"].remove("acme-cost-05")
    assert any("belongs to job-acme, not proj-airflow-demo" in x for x in errs(master, d))
    d["bullets"]["job-acme"].append("acme-cost-05")
    d["bullets"]["proj-airflow-demo"] = []  # a selected project may show no bullets
    assert errs(master, d) == []


def test_project_bullet_limits(master_dict, selection_dict):
    from cvtailor.master import parse_master
    master_dict["limits"]["max_bullets_per_project"] = 1
    m = parse_master(master_dict)
    d = copy.deepcopy(selection_dict)
    d["bullets"]["proj-airflow-demo"] = ["lakehouse-tests-02", "lakehouse-dags-01"]
    e = errs(m, d)
    assert any("above max_bullets_per_project (1)" in x for x in e)
