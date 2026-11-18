from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvtailor.config import Config  # noqa: E402
from cvtailor.llm import LLM  # noqa: E402
from cvtailor.master import parse_master  # noqa: E402
from cvtailor.pipeline import IO, Context  # noqa: E402
from cvtailor.render import Rendered, render_html  # noqa: E402
from cvtailor.schema import Selection  # noqa: E402

EXAMPLE_MASTER = ROOT / "examples" / "master.example.yaml"
EXAMPLE_SELECTION = ROOT / "examples" / "selection.example.json"
EXAMPLE_JOB = ROOT / "examples" / "job.example.txt"


def weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


needs_weasyprint = pytest.mark.skipif(not weasyprint_available(),
                                      reason="WeasyPrint/Pango not installed")


@pytest.fixture
def master_dict() -> dict:
    return copy.deepcopy(yaml.safe_load(EXAMPLE_MASTER.read_text(encoding="utf-8")))


@pytest.fixture
def master(master_dict):
    return parse_master(master_dict)


@pytest.fixture
def selection_dict() -> dict:
    return json.loads(EXAMPLE_SELECTION.read_text(encoding="utf-8"))


@pytest.fixture
def selection(selection_dict) -> Selection:
    return Selection.model_validate(selection_dict)


@pytest.fixture
def job_text() -> str:
    return EXAMPLE_JOB.read_text(encoding="utf-8")


class FakeRenderer:
    """Renders real HTML; 'pages' grows with content so page-fit logic can be tested."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity  # units: one per bullet, two per project
        self.calls = 0
        self.htmls: list[str] = []

    def render(self, ri) -> Rendered:
        self.calls += 1
        html = render_html(ri)
        self.htmls.append(html)
        sel = ri.selection
        units = sel.total_bullets() + 2 * len(sel.project_ids)
        pages = 1 if units <= self.capacity else 2
        return Rendered(pages, b"%PDF-1.7 fake " + ri.lang.encode(), html)


def _resp(payload, usage=(100, 0, 0, 50), stop="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=usage[0], cache_read_input_tokens=usage[1],
                              cache_creation_input_tokens=usage[2], output_tokens=usage[3]),
        stop_reason=stop,
    )


class FakeClient:
    """Mimics client.messages.create. Queues answers for meta and selection calls."""

    def __init__(self, meta=None, selections=None, meta_error: Exception | None = None,
                 proposals=None, revisions=None, texts=None, versions=None):
        self.meta = meta or {"company": "Acme Pagamentos", "role_title": "Senior Data Engineer",
                             "language": "en", "job_url": None}
        self.selections = list(selections or [])
        self.proposals = list(proposals or [])   # each entry: list of drafts for one propose call
        self.revisions = list(revisions or [])   # each entry: one revised draft
        self.meta_error = meta_error
        self.texts = list(texts or [])  # free-text answers (careful mode analysis / review)
        self.versions = list(versions or [])  # each entry: list of drafted versions for one call
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "format" not in kwargs.get("output_config", {}):
            return _resp(self.texts.pop(0), usage=(400, 0, 0, 80))
        schema = kwargs["output_config"]["format"]["schema"]
        if "role_title" in schema["properties"]:
            if self.meta_error:
                raise self.meta_error
            return _resp(self.meta, usage=(50, 0, 0, 20))
        if "drafts" in schema["properties"]:
            return _resp({"drafts": self.versions.pop(0)}, usage=(300, 0, 0, 60))
        if "proposals" in schema["properties"]:
            return _resp({"proposals": self.proposals.pop(0)}, usage=(500, 0, 0, 100))
        if "origin" in schema["properties"]:
            return _resp(self.revisions.pop(0), usage=(300, 0, 0, 60))
        if not self.selections:
            raise AssertionError("unexpected selection call")
        return _resp(self.selections.pop(0), usage=(1000, 800, 0, 200))


@pytest.fixture
def make_ctx(tmp_path, master):
    def _make(client=None, renderer=None, capacity: int = 100, allow_generation: bool = False) -> Context:
        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        cfg = Config(
            data_dir=data,
            output_dir=tmp_path / "output",
            master_path=EXAMPLE_MASTER,
            variants_dir=data / "variants",
            db_path=data / "applications.db",
            model="test-model",
            effort="low",
            route_threshold=0.6,
            pdf_name="{name}_CV_{role}_{LANG}",
            allow_generation=allow_generation,
        )
        ctx = Context(cfg)
        ctx._base_master = master
        ctx.llm = LLM("test-model", "low", client=client or FakeClient())
        ctx.renderer = renderer or FakeRenderer(capacity)
        return ctx
    return _make


@pytest.fixture
def quiet_io():
    lines: list[str] = []
    io = IO(out=lines.append, confirm=lambda q: True, prompt=lambda q, d: d or "x", interactive=True)
    io.lines = lines  # type: ignore[attr-defined]
    return io
