"""Optional live test against the Anthropic API. Run with CV_INTEGRATION=1."""

import os

import pytest

from cvtailor.config import Config
from cvtailor.llm import LLM
from cvtailor.validate import validate_selection

pytestmark = pytest.mark.skipif(os.environ.get("CV_INTEGRATION") != "1",
                                reason="set CV_INTEGRATION=1 to call the real API")


def test_live_meta_and_selection(master, job_text):
    cfg = Config.from_env()
    llm = LLM(cfg.model, cfg.effort)
    meta, _ = llm.extract_meta(job_text)
    assert meta.company and meta.role_title
    sel, usage = llm.select(job_text, master)
    assert validate_selection(sel.core(), master) == []
    assert usage.input_tokens > 0 and usage.output_tokens > 0
