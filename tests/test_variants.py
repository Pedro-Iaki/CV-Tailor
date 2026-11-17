import json

import pytest

from cvtailor.master import parse_master
from cvtailor.variants import VariantError, get_variant, load_variants, save_variant


def _save(tmp_path, master, selection, name="v1", **kw):
    return save_variant(tmp_path, name=name, selection=selection, job_tags=["spark", "azure", "delta"],
                        master=master, source_application_id=7, **kw)


def test_ok_state_and_file_format(tmp_path, master, selection):
    path = _save(tmp_path, master, selection)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "v1" and data["source_application_id"] == 7
    assert data["item_hashes"]["sum-platform"].startswith("sha256:")
    assert "skills-processing" in data["item_hashes"]
    [lv] = load_variants(tmp_path, master)
    assert lv.state == "ok" and lv.eligible


def test_drifted_on_text_change(tmp_path, master_dict, master, selection):
    _save(tmp_path, master, selection)
    master_dict["jobs"][0]["bullets"][1]["pt"] += " (editado)"
    master_dict["skills"]["hands_on"][0]["items"].append("Polars")
    lv = get_variant(tmp_path, "v1", parse_master(master_dict))
    assert lv.state == "drifted" and lv.eligible
    assert lv.changed_ids == ["acme-delta-02", "skills-processing"]


def test_stale_on_removed_id(tmp_path, master_dict, master, selection):
    _save(tmp_path, master, selection)
    master_dict["projects"] = [p for p in master_dict["projects"] if p["id"] != "proj-kafka-lab"]
    lv = get_variant(tmp_path, "v1", parse_master(master_dict))
    assert lv.state == "stale" and not lv.eligible
    assert any("proj-kafka-lab" in r for r in lv.reasons)


def test_stale_on_new_job(tmp_path, master_dict, master, selection):
    _save(tmp_path, master, selection)
    master_dict["jobs"].append({
        "id": "job-new", "company": "NewCo", "title": {"en": "DE", "pt": "ED"}, "start": "2018-01",
        "end": "2019-01", "bullets": [{"id": "new-01", "tags": ["python"], "en": "x", "pt": "y"}],
    })
    lv = get_variant(tmp_path, "v1", parse_master(master_dict))
    assert lv.state == "stale"
    assert any("job 'job-new' is missing" in r for r in lv.reasons)


def test_name_exists_requires_replace(tmp_path, master, selection):
    _save(tmp_path, master, selection)
    with pytest.raises(VariantError, match="already exists"):
        _save(tmp_path, master, selection)
    _save(tmp_path, master, selection, replace=True)


def test_bad_name(tmp_path, master, selection):
    with pytest.raises(VariantError):
        _save(tmp_path, master, selection, name="../evil")
