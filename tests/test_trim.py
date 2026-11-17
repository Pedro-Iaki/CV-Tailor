import pytest

from cvtailor.trim import PageFitError, fit_to_page
from tests.conftest import FakeRenderer

# Example selection: acme 5, datacorp 3, intern 1 bullets (9) + 2 projects -> 13 units.


def test_no_trim_needed(master, selection):
    res = fit_to_page(selection, master, "en", FakeRenderer(13).render)
    assert res.removed == [] and res.selection == selection


def test_trim_order(master, selection):
    # capacity 9: must drop 4 units. acme has most bullets -> drop its last bullets first
    # until it ties datacorp (3); tie -> older job (datacorp) first.
    res = fit_to_page(selection, master, "en", FakeRenderer(9).render)
    assert res.removed == ["acme-cost-05", "acme-cicd-04", "datacorp-sql-02", "acme-adf-03"]
    assert res.rendered.pages == 1


def test_projects_removed_after_jobs_at_min(master, selection):
    res = fit_to_page(selection, master, "en", FakeRenderer(4).render)
    # all jobs to min (1 each = 3 units) then the last project goes (3 + 2 = 5 > 4), then the other
    assert res.removed[-2:] == ["proj-kafka-lab", "proj-airflow-demo"]
    for job_id, ids in res.selection.bullets.items():
        assert len(ids) >= master.limits.min_bullets_per_job


def test_never_below_minimum_fails_cleanly(master, selection):
    with pytest.raises(PageFitError, match="minimum bullets"):
        fit_to_page(selection, master, "en", FakeRenderer(1).render)


def test_no_trim_fails_with_page_count(master, selection):
    with pytest.raises(PageFitError, match="2 pages"):
        fit_to_page(selection, master, "en", FakeRenderer(5).render, trim=False)


def test_trim_reports_text(master, selection):
    seen = []
    fit_to_page(selection, master, "pt", FakeRenderer(12).render, on_remove=lambda i, t: seen.append((i, t)))
    assert seen == [("acme-cost-05", "Reduzi o custo de computação do Databricks em 35% com políti")]


def test_project_bullets_trim_and_project_removal_drops_key(master, selection_dict):
    from cvtailor.schema import Selection
    d = dict(selection_dict, bullets=dict(selection_dict["bullets"]))
    d["bullets"]["proj-airflow-demo"] = ["lakehouse-dags-01", "lakehouse-tests-02"]
    sel = Selection.model_validate(d)  # 11 bullets + 2 projects = 15 units
    res = fit_to_page(sel, master, "en", FakeRenderer(4).render)
    assert "lakehouse-tests-02" in res.removed and "lakehouse-dags-01" in res.removed
    assert res.selection.project_ids == [] and "proj-airflow-demo" not in res.selection.bullets
