"""The desktop app's API (no window), portable layout, settings and the Ollama checks."""

import io
import json
import time

import pytest

from cvtailor import config, ollama_setup
from cvtailor.desktop import Api, ensure_portable_layout, reload_settings
from tests.conftest import FakeClient, FakeRenderer


def wait(api, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        j = api.job_status(job_id)
        if j["done"]:
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def desk(make_ctx, selection_dict):
    client = FakeClient(selections=[selection_dict] * 3)
    renderer = FakeRenderer()

    def factory():
        return make_ctx(client=client, renderer=renderer)
    return Api(factory)


def test_new_cv_job_and_applications(desk, job_text):
    j = wait(desk, desk.new_cv(job_text, {"lang": "en"}))
    assert j["error"] is None, j
    r = j["result"]
    assert r["app_id"] == 1 and r["company"] == "Acme Pagamentos" and r["pdf"].endswith(".pdf")
    assert "unmatched" in r and any("tags:" in line for line in j["lines"])

    rows = desk.list_apps()
    assert [(x["id"], x["status"], x["lang"]) for x in rows] == [(1, "applied", "en")]
    desk.set_status(1, "interview")
    assert desk.list_apps()[0]["status"] == "interview"
    assert '<html lang="pt-BR">' in desk.preview(1, "pt")

    # same posting again: blocked unless allowed
    dup = wait(desk, desk.new_cv(job_text, {}))
    assert dup["error"] and "duplicate of #1" in dup["error"]
    ok = wait(desk, desk.new_cv(job_text, {"allow_duplicate": True}))
    assert ok["error"] is None and ok["result"]["app_id"] == 2


def test_dry_run_and_errors(desk, job_text):
    j = wait(desk, desk.new_cv(job_text, {"dry_run": True}))
    assert j["result"]["app_id"] is None and any("dry run" in line for line in j["lines"])
    assert desk.list_apps() == []
    bad = wait(desk, desk.new_cv("   ", {}))
    assert "empty" in bad["error"]
    assert desk.job_status("nope")["error"] == "unknown job"


def test_state_and_rating(desk, job_text):
    s = desk.get_state()
    assert s["master_ok"] and s["local"] is False and s["ollama"] is None
    assert len(s["recommended"]) == 3
    wait(desk, desk.new_cv(job_text, {}))
    [item] = desk.rate_items()
    assert item["key"] == "app:1" and item["label"] == ""
    assert desk.rate("app:1", 4)["label"].startswith("application #1")
    assert desk.rate_summary()[0]["mean"] == 4.0


def test_broken_candidate_file_is_reported_not_fatal(desk, job_text):
    wait(desk, desk.new_cv(job_text, {}))
    cand_dir = desk._ctx().cfg.data_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "bad.jsonl").write_text('{"key": "x", "job_text": "j", "selection": {}, '
                                        '"master_path": "missing.yaml"}\n', encoding="utf-8")
    assert [i["key"] for i in desk.rate_items()] == ["app:1"]
    info = desk.rate_info()
    assert info["count"] == 1 and len(info["problems"]) == 1 and "bad.jsonl" in info["problems"][0]


def test_portable_layout_and_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CV_PORTABLE", "1")
    monkeypatch.setenv("CV_APP_HOME", str(tmp_path))
    for k in ("CV_MODEL", "CV_LLM_REASONING", "CV_LLM_BASE_URL", "CV_DATA_DIR", "CV_OUTPUT_DIR"):
        monkeypatch.delenv(k, raising=False)
    assert config.default_dirs() == (tmp_path / "Data", tmp_path / "Output", tmp_path / "Data" / "settings.env")
    ensure_portable_layout()
    text = (tmp_path / "Data" / "settings.env").read_text(encoding="utf-8")
    assert "CV_MODEL=qwen3.5:35b-a3b" in text and (tmp_path / "Output").is_dir()
    reload_settings()
    cfg = config.Config.from_env()
    assert cfg.data_dir == tmp_path / "Data" and cfg.model == "qwen3.5:35b-a3b" and cfg.llm_reasoning == "none"

    api = Api()
    fields = {f["key"]: f for f in api.get_settings()["fields"]}
    assert fields["CV_MODEL"]["value"] == "qwen3.5:35b-a3b"
    monkeypatch.setattr(ollama_setup, "status", lambda base: {"running": False, "models": []})
    api.save_settings({"CV_MODEL": "lfm2.5:8b", "CV_LLM_REASONING": "", "NOT_A_SETTING": "x"})
    saved = (tmp_path / "Data" / "settings.env").read_text(encoding="utf-8")
    assert "CV_MODEL=lfm2.5:8b" in saved and "CV_LLM_REASONING" not in saved and "NOT_A_SETTING" not in saved
    cfg = config.Config.from_env()
    assert cfg.model == "lfm2.5:8b" and cfg.llm_reasoning is None


def test_ollama_status(monkeypatch):
    replies = {"/api/version": {"version": "0.40.16"},
               "/api/tags": {"models": [{"name": "qwen3.5:35b-a3b"}, {"name": "bge-m3:latest"}]}}

    def urlopen(url, timeout=None):
        path = url.split("11434", 1)[1]
        return io.BytesIO(json.dumps(replies[path]).encode())
    monkeypatch.setattr(ollama_setup.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ollama_setup, "find_executable", lambda: None)
    s = ollama_setup.status("http://localhost:11434/v1")
    assert s["running"] and s["installed"] and s["version"] == "0.40.16"
    assert ollama_setup.has_model(s["models"], "bge-m3") and ollama_setup.has_model(s["models"], "qwen3.5:35b-a3b")
    assert not ollama_setup.has_model(s["models"], "lfm2.5:8b")

    def down(url, timeout=None):
        raise OSError("refused")
    monkeypatch.setattr(ollama_setup.urllib.request, "urlopen", down)
    s = ollama_setup.status(None)
    assert not s["running"] and not s["installed"] and s["server"] == "http://localhost:11434"


def test_ollama_pull_progress(monkeypatch):
    lines = [b'{"status":"pulling manifest"}\n', b'{"status":"downloading","completed":5,"total":10}\n',
             b'{"status":"success"}\n']
    monkeypatch.setattr(ollama_setup.urllib.request, "urlopen", lambda req, timeout=None: io.BytesIO(b"".join(lines)))
    seen = []
    ollama_setup.pull(None, "lfm2.5:8b", lambda s, d, t: seen.append((s, d, t)))
    assert seen == [("pulling manifest", 0, 0), ("downloading", 5, 10), ("success", 0, 0)]
    monkeypatch.setattr(ollama_setup.urllib.request, "urlopen",
                        lambda req, timeout=None: io.BytesIO(b'{"error":"model not found"}\n'))
    with pytest.raises(RuntimeError, match="not found"):
        ollama_setup.pull(None, "nope", lambda *a: None)
