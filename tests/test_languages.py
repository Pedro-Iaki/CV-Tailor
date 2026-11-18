"""Configurable CV languages: a primary language (required everywhere) and an optional
secondary one (items without it are left out of CVs in that language)."""

from __future__ import annotations

import copy
import hashlib
import shutil
import sqlite3

import pytest

from cvtailor import db
from cvtailor.langs import LANGUAGES, normalize
from cvtailor.llm import build_meta_schema, build_versions_schema
from cvtailor.master import MasterError, gate_language, missing_texts, parse_master
from cvtailor.master_edit import EditError, MasterEditor, create_master
from cvtailor.pipeline import NewOptions, run_new
from cvtailor.render import MissingTextError, load_i18n, make_render_input, render_html
from cvtailor.schema import Meta, Selection
from cvtailor.variants import item_hash
from tests.conftest import EXAMPLE_MASTER, FakeClient


def rename_lang(node, old: str, new: str):
    """The example master with every `old` text key renamed to `new`."""
    if isinstance(node, dict):
        return {(new if k == old else k): rename_lang(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [rename_lang(v, old, new) for v in node]
    return node


@pytest.fixture
def es_dict(master_dict):
    d = rename_lang(master_dict, "pt", "es")
    d["languages"] = {"primary": "en", "secondary": "es"}
    return d


def test_legacy_master_is_english_and_portuguese(master):
    assert master.langs == ["en", "pt"] and master.primary == "en"


def test_any_language_pair(es_dict, selection):
    m = parse_master(es_dict)
    assert m.langs == ["en", "es"]
    html = render_html(make_render_input(m, selection, "es"))
    assert '<html lang="es">' in html and "Experiencia" in html


def test_secondary_text_may_be_missing_but_not_primary(es_dict):
    del es_dict["jobs"][0]["bullets"][0]["es"]
    m = parse_master(es_dict)  # no error: the item is just left out of Spanish CVs
    assert ("acme-dbx-pipeline-01", "bullet acme-dbx-pipeline-01") in missing_texts(m, "es")
    del es_dict["jobs"][0]["bullets"][1]["en"]
    with pytest.raises(MasterError, match=r"'en' \(English\) is empty"):
        parse_master(es_dict)


def test_unknown_language_is_rejected(master_dict):
    master_dict["jobs"][0]["bullets"][0]["xx"] = "?"
    with pytest.raises(MasterError, match="unknown language 'xx'"):
        parse_master(master_dict)
    d = copy.deepcopy(master_dict)
    del d["jobs"][0]["bullets"][0]["xx"]
    d["languages"] = {"primary": "en", "secondary": "en"}
    with pytest.raises(MasterError, match="must differ"):
        parse_master(d)


def test_one_language_only(master_dict, selection):
    d = master_dict
    d["languages"] = {"primary": "en", "secondary": None}
    m = parse_master(d)
    assert m.langs == ["en"] and missing_texts(m, "en") == []


def test_render_refuses_missing_text(es_dict, selection):
    del es_dict["jobs"][0]["bullets"][0]["es"]
    m = parse_master(es_dict)
    assert "acme-dbx-pipeline-01" in selection.bullets["job-acme"]
    with pytest.raises(MissingTextError, match="no Spanish text yet for: acme-dbx-pipeline-01"):
        render_html(make_render_input(m, selection, "es"))
    render_html(make_render_input(m, selection, "en"))  # English is complete


def test_gate_language_hides_untranslated_items(es_dict):
    del es_dict["jobs"][0]["bullets"][0]["es"]
    del es_dict["summaries"][0]["es"]
    m = parse_master(es_dict)
    g = gate_language(m, "es")
    assert "acme-dbx-pipeline-01" not in {b.id for b in g.jobs[0].bullets}
    assert es_dict["summaries"][0]["id"] not in {s.id for s in g.summaries}
    assert gate_language(m, "en") is m


def test_old_selection_naming_skills_in_another_language_still_renders(master, selection):
    sel = selection.model_copy(deep=True)
    cat = next(c for c in master.skills.hands_on if any(not isinstance(i, str) for i in c.items))
    item = next(i for i in cat.items if not isinstance(i, str))
    sel.skill_category_order.hands_on = [cat.id]
    sel.skill_items = {cat.id: [item.get("pt")]}  # named by its Portuguese text
    from cvtailor.validate import validate_selection
    assert not [e for e in validate_selection(sel, master) if "skill_items" in e]
    assert item.get("en") in render_html(make_render_input(master, sel, "en"))


def test_variant_hashes_unchanged_for_english_and_portuguese(master):
    b = master.jobs[0].bullets[0]
    old = "sha256:" + hashlib.sha256(f"{b.get('en')}\n{b.get('pt')}".encode()).hexdigest()
    assert item_hash(master, b.id) == old


def test_i18n_has_every_language():
    i18n = load_i18n()
    assert set(LANGUAGES) <= set(i18n)
    assert all(set(i18n[c]) == set(i18n["en"]) for c in LANGUAGES)


def test_normalize_and_meta():
    assert normalize("pt-BR") == "pt" and normalize("Spanish") == "es" and normalize("Español") == "es"
    assert Meta(company="a", role_title="b", language="Português").language == "pt"


def test_llm_schemas_use_the_master_languages():
    assert build_meta_schema(["en", "es"])["properties"]["language"]["enum"] == ["en", "es"]
    props = build_versions_schema(2, ["en", "es"])["properties"]["drafts"]["items"]["properties"]
    assert set(props) == {"en", "es", "note"}


def test_posting_in_another_language_gets_a_primary_cv(make_ctx, selection_dict, job_text):
    client = FakeClient(meta={"company": "Acme", "role_title": "Data Engineer", "language": "fr", "job_url": None},
                        selections=[selection_dict])
    ctx = make_ctx(client=client)
    res = run_new(ctx, job_text, NewOptions(), _io())
    assert res.lang == "en"
    meta_call = client.calls[0]
    assert meta_call["output_config"]["format"]["schema"]["properties"]["language"]["enum"] == ["en", "pt"]


def _io():
    from cvtailor.pipeline import IO
    return IO(out=lambda s: None, interactive=False)


def test_db_migration_moves_texts(tmp_path):
    conn = sqlite3.connect(tmp_path / "a.db", isolation_level=None)
    for sql in db.MIGRATIONS[:3]:
        for st in [s for s in sql.split(";") if s.strip()]:
            conn.execute(st)
    conn.execute("PRAGMA user_version = 3")
    conn.execute("INSERT INTO facts (id, kind, origin, based_on, tags, en, pt, created_at, approved_at, approved_via) "
                 "VALUES ('f1', 'bullet', 'rephrase', '[]', '[]', 'Hello', 'Olá', 'x', 'x', 'review')")
    conn.close()
    conn = db.connect(tmp_path / "a.db")
    row = db.item_row_to_dict(conn.execute("SELECT * FROM facts").fetchone())
    assert row["texts"] == {"en": "Hello", "pt": "Olá"} and "en" not in row


# ---------------------------------------------------------------- editing

@pytest.fixture
def ed(tmp_path):
    shutil.copy(EXAMPLE_MASTER, tmp_path / "master.yaml")
    return MasterEditor(tmp_path / "master.yaml", tmp_path / "history")


def test_set_languages(ed):
    m = ed.set_languages("en", "es")
    assert m.langs == ["en", "es"]
    text = ed.path.read_text(encoding="utf-8")
    assert "languages: { primary: en, secondary: es }" in text and "pt:" in text  # Portuguese stays in the file
    assert len(missing_texts(m, "es")) > 10
    with pytest.raises(EditError, match="no Spanish yet"):
        ed.set_languages("es", "en")  # the main language must be on every item
    assert ed.set_languages("pt", "en").langs == ["pt", "en"]  # Portuguese is complete
    assert ed.set_languages("en", None).langs == ["en"]


def test_entry_add_and_edit(ed):
    m, new_id = ed.add("job", None, {"title": {"en": "Staff Engineer", "pt": "Engenheiro Staff"}, "company": "NewCo",
                                     "start": "2025-02", "end": "",
                                     "bullet": {"en": "Led the platform team.", "pt": "Liderei o time.", "tags": "leadership"}})
    job = next(j for j in m.jobs if j.id == new_id)
    assert new_id == "job-newco" and job.end is None and job.bullets[0].get("pt") == "Liderei o time."
    m = ed.update(new_id, {"location": "Remote", "end": "2025-12", "title": {"en": "Principal Engineer"}})
    job = next(j for j in m.jobs if j.id == new_id)
    assert job.location == "Remote" and job.end == "2025-12" and job.title.get("pt") == "Engenheiro Staff"
    m, pid = ed.add("project", None, {"name": "Tool", "tags": "python", "description": {"en": "A CLI.", "pt": "Uma CLI."}})
    assert next(p for p in m.projects if p.id == pid).name == "Tool"
    m, eid = ed.add("education", None, {"degree": {"en": "MSc", "pt": "Mestrado"}, "institution": "Uni", "start": "2026",
                                        "end": "2028", "expected": True})
    assert next(e for e in m.education if e.id == eid).expected
    with pytest.raises(EditError, match="needs: start"):
        ed.add("job", None, {"title": {"en": "X"}, "company": "Y", "bullet": {"en": "Z.", "tags": "a"}})


def test_empty_secondary_text_removes_it(ed):
    m = ed.update("acme-delta-02", {"en": "Moved tables to Delta.", "pt": ""})
    b = next(b for b in m.jobs[0].bullets if b.id == "acme-delta-02")
    assert b.get("pt") == "" and ("acme-delta-02", "bullet acme-delta-02") in missing_texts(m, "pt")


def test_create_master_from_the_wizard(tmp_path):
    data = {"languages": {"primary": "es", "secondary": None},
            "profile": {"name": "Ana Ejemplo", "headline": {"es": "Ingeniera de datos"}, "links": [{"label": "Web", "url": "https://example.com"}]},
            "summary": {"es": "Ingeniera de datos.", "tags": "datos"},
            "jobs": [{"title": {"es": "Ingeniera"}, "company": "Foo", "start": "2021-03", "end": "",
                      "bullets": [{"es": "Construí la ingesta.", "tags": "ingesta"}, {"es": "Bajé costes.", "tags": "costes"}]}],
            "education": [{"degree": {"es": "Grado"}, "institution": "Uni", "start": "2014", "end": "2018"}]}
    m = create_master(tmp_path / "master.yaml", data)
    assert m.langs == ["es"] and len(m.jobs[0].bullets) == 2 and m.education[0].id == "edu-uni"
    sel = Selection(summary_id="sum-main", bullets={m.jobs[0].id: [b.id for b in m.jobs[0].bullets]})
    assert "Experiencia" in render_html(make_render_input(m, sel, "es"))
    with pytest.raises(EditError, match="already exists"):
        create_master(tmp_path / "master.yaml", data)
    with pytest.raises(EditError, match="name is missing"):
        create_master(tmp_path / "other.yaml", {**data, "profile": {"name": ""}})
    assert not (tmp_path / "other.yaml").exists()


def test_desktop_languages_and_wizard(tmp_path):
    import dataclasses
    from cvtailor.config import Config
    from cvtailor.desktop import Api
    from cvtailor.pipeline import Context
    cfg = Config(data_dir=tmp_path, output_dir=tmp_path / "out", master_path=tmp_path / "master.yaml",
                 variants_dir=tmp_path / "variants", db_path=tmp_path / "a.db", model="m", effort=None,
                 route_threshold=0.6, pdf_name="{name}_{LANG}")
    api = Api(lambda: Context(dataclasses.replace(cfg)))
    st = api.get_state()
    assert st["master_missing"] and st["languages"] is None and {"code": "es", "name": "Spanish"} in st["language_choices"]
    bad = api.create_master({"languages": {"primary": "en"}, "profile": {"name": "A"}, "summary": {}})
    assert not bad["ok"] and any("summary" in e for e in bad["errors"])
    shutil.copy(EXAMPLE_MASTER, cfg.master_path)
    r = api.set_languages("en", "fr")
    assert r["ok"] and r["languages"]["list"] == [{"code": "en", "name": "English"}, {"code": "fr", "name": "French"}]
    assert r["missing"] > 0
    bs = api.builder_state()
    assert bs["langs"][1]["code"] == "fr" and "acme-delta-02" in bs["missing"]["fr"]
    assert not api.set_languages("fr", "en")["ok"]
