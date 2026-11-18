"""The builder's master editor: comment-preserving edits, validation before writing,
history and undo."""

import shutil
from pathlib import Path

import pytest

from cvtailor.master import load_master
from cvtailor.master_edit import EditError, MasterEditor

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "master.example.yaml"


@pytest.fixture
def ed(tmp_path):
    p = tmp_path / "master.yaml"
    shutil.copy(EXAMPLE, p)
    return MasterEditor(p, tmp_path / "history")


def test_round_trip_is_identical(ed):
    assert ed.dump(ed.load()) == ed.path.read_text(encoding="utf-8")


def test_update_text_keeps_comments_and_history(ed):
    before = ed.path.read_text(encoding="utf-8")
    m = ed.update("acme-delta-02", {"en": "Migrated 40 tables to Delta Lake.", "tags": "delta, data-modeling"})
    b = next(b for j in m.jobs for b in j.bullets if b.id == "acme-delta-02")
    assert b.en == "Migrated 40 tables to Delta Lake." and b.tags == ["delta", "data-modeling"]
    after = ed.path.read_text(encoding="utf-8")
    assert "# Example master file." in after and "# Projects can have a context line" in after
    changed = [l for l in after.splitlines() if l not in before.splitlines()]
    assert len(changed) == 2  # only the edited lines differ
    assert ed.history_count() == 1


def test_invalid_change_is_rejected_and_file_untouched(ed):
    before = ed.path.read_text(encoding="utf-8")
    with pytest.raises(EditError, match="at least one tag"):
        ed.update("acme-delta-02", {"tags": ""})
    with pytest.raises(EditError, match="can't be empty"):
        ed.update("acme-delta-02", {"pt": "  "})
    assert ed.path.read_text(encoding="utf-8") == before and ed.history_count() == 0


def test_add_move_delete_undo(ed):
    m, new_id = ed.add("bullet", "job-acme", {"en": "Built a pipeline monitor.", "pt": "Construí um monitor.",
                                              "tags": ["monitoring"]}, after="acme-delta-02")
    ids = [b.id for b in next(j for j in m.jobs if j.id == "job-acme").bullets]
    assert new_id.startswith("acme-") and ids[ids.index("acme-delta-02") + 1] == new_id
    m = ed.move(new_id, -1)
    ids = [b.id for b in next(j for j in m.jobs if j.id == "job-acme").bullets]
    assert ids.index(new_id) == ids.index("acme-delta-02") - 1
    m = ed.delete(new_id)
    assert new_id not in {b.id for j in m.jobs for b in j.bullets}
    m = ed.undo()  # brings the deleted bullet back
    assert new_id in {b.id for j in m.jobs for b in j.bullets}
    assert load_master(ed.path).jobs[0].bullets  # the file on disk is valid


def test_delete_refuses_when_an_alternative_points_at_it(ed):
    ed.add("bullet", "job-acme", {"id": "acme-delta-02b", "alt_of": "acme-delta-02", "en": "Short.",
                                  "pt": "Curto.", "tags": ["delta"]})
    with pytest.raises(EditError, match="acme-delta-02b"):
        ed.delete("acme-delta-02")


def test_skill_category_items_and_duplicate_id(ed):
    m = ed.update("skills-languages", {"items": ["Python", {"en": "SQL", "pt": "SQL"},
                                                 {"en": "Shell scripting", "pt": "Scripts shell"}]})
    cat = next(c for c in m.skills.hands_on if c.id == "skills-languages")
    assert [getattr(i, "en", i) for i in cat.items] == ["Python", "SQL", "Shell scripting"]
    with pytest.raises(EditError, match="already used"):
        ed.add("summary", None, {"id": "sum-platform", "en": "x", "pt": "y", "tags": ["a"]})
    with pytest.raises(EditError, match="not in the master"):
        ed.update("nope", {"en": "x"})
