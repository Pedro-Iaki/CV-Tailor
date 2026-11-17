"""The local rating page: candidates, blind labels, storage, summary and HTTP."""

import json
import threading
import urllib.request

import pytest

from cvtailor import db
from cvtailor.ratings import (Candidate, RatingApp, RatingStore, application_candidates,
                              file_candidates, serve)
from cvtailor.schema import Selection


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "a.db")


@pytest.fixture
def store(tmp_path):
    return RatingStore(tmp_path / "a.db")


def cands(master, selection_dict, n=3):
    sel = Selection.model_validate(selection_dict)
    return [Candidate(key=f"k{i}", title=f"Job {i}", job_text=f"Posting {i}", selection=sel, master=master,
                      label=f"model {i % 2}", group=f"setup {i % 2}") for i in range(n)]


def test_blind_until_rated_and_summary(store, master, selection_dict):
    app = RatingApp(cands(master, selection_dict), store)
    items = app.items()
    assert [i["n"] for i in items] == [1, 2, 3] and all(i["label"] == "" and i["rating"] is None for i in items)
    assert app.rate("k0", 4, " fine ") == {"key": "k0", "rating": 4, "label": "model 0"}
    app.rate("k2", 2)
    app.rate("k0", 5)  # re-rating replaces
    items = {i["key"]: i for i in app.items()}
    assert items["k0"]["rating"] == 5 and items["k0"]["label"] == "model 0" and items["k0"]["note"] == ""
    assert items["k1"]["label"] == ""  # still hidden
    summary = {r["group"]: r for r in app.summary()}
    assert summary["setup 0"] == {"group": "setup 0", "rated": 2, "total": 2, "mean": 3.5, "ratings": [5, 2]}
    assert summary["setup 1"]["mean"] is None
    with pytest.raises(ValueError):
        app.rate("k1", 6)


def test_cv_html_renders_both_languages(store, master, selection_dict):
    app = RatingApp(cands(master, selection_dict, 1), store)
    assert '<html lang="en">' in app.cv_html("k0", "en")
    assert '<html lang="pt-BR">' in app.cv_html("k0", "pt")
    bad = Candidate(key="bad", title="x", job_text="x", master=master,
                    selection=Selection.model_validate({**selection_dict, "summary_id": "nope"}))
    app = RatingApp([bad], store)
    assert "not valid" in app.cv_html("bad", "en")


def test_file_and_application_candidates(tmp_path, conn, master, selection_dict):
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"key": "x1", "job_text": "Senior Data Engineer\nmore", "selection": selection_dict,
                             "label": "qwen", "group": "careful"}) + "\n\n", encoding="utf-8")
    [c] = file_candidates(f, master)
    assert (c.key, c.title, c.group, c.lang) == ("x1", "Senior Data Engineer", "careful", "en")
    conn.execute("BEGIN")
    db.insert_application(conn, {"company": "Acme", "role_title": "DE", "company_key": "acme|de", "lang": "pt",
                                 "job_description": "job", "job_hash": "h", "job_tags": [], "route": "generated",
                                 "selection": selection_dict, "pdf_path": "x.pdf"})
    conn.execute("COMMIT")
    [a] = application_candidates(conn, master)
    assert (a.key, a.title, a.lang, a.group) == ("app:1", "Acme | DE", "pt", "applications")


def test_http(store, master, selection_dict):
    server = serve(RatingApp(cands(master, selection_dict, 2), store), port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        def get(path):
            with urllib.request.urlopen(base + path, timeout=5) as r:
                return r.read().decode()
        assert "Rate CVs" in get("/")
        assert len(json.loads(get("/api/items"))) == 2
        assert "Posting 1" == get("/job/k1")
        assert "<html" in get("/cv/k1?lang=pt")
        req = urllib.request.Request(base + "/api/rate", json.dumps({"key": "k1", "rating": 3}).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read())["label"] == "model 1"
        bad = urllib.request.Request(base + "/api/rate", json.dumps({"key": "k1", "rating": 9}).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad, timeout=5)
        assert e.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
    assert server.server_address[0] == "127.0.0.1"  # local only
