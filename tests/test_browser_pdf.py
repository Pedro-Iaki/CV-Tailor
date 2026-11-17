"""PDF through a headless browser (Edge/Chrome): engine choice, page counting, errors."""

import subprocess
from pathlib import Path

import pytest

from cvtailor import render
from cvtailor.render import (BrowserRenderer, PdfEngineMissing, WeasyRenderer, count_pdf_pages,
                             find_browser, make_render_input, make_renderer)


def fake_pdf(pages: int) -> bytes:
    kids = b" ".join(b"%d 0 R" % (i + 3) for i in range(pages))
    objs = [b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj",
            b"2 0 obj <</Type /Pages /Kids [" + kids + b"] /Count %d>> endobj" % pages]
    objs += [b"%d 0 obj <</Type /Page /Parent 2 0 R>> endobj" % (i + 3) for i in range(pages)]
    return b"%PDF-1.4\n" + b"\n".join(objs) + b"\n%%EOF"


def test_count_pages():
    assert count_pdf_pages(fake_pdf(1)) == 1
    assert count_pdf_pages(fake_pdf(3)) == 3
    assert count_pdf_pages(b"<</Type/Page>><</Type /Pages>><</Type /PageLabel>>") == 1


def test_renderer_runs_browser_in_a_temp_profile(monkeypatch, master, selection):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        out = next(a.split("=", 1)[1] for a in cmd if a.startswith("--print-to-pdf="))
        Path(out).write_bytes(fake_pdf(2))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(render.subprocess, "run", run)
    r = BrowserRenderer("msedge.exe").render(make_render_input(master, selection, "en"))
    assert r.pages == 2 and r.pdf.startswith(b"%PDF") and "<html" in r.html
    cmd = seen["cmd"]
    assert cmd[0] == "msedge.exe" and "--headless" in cmd and "--no-pdf-header-footer" in cmd
    profile = next(a for a in cmd if a.startswith("--user-data-dir="))
    assert "cvtailor-pdf-" in profile  # a throwaway profile, not the user's
    assert cmd[-1].startswith("file:")


def test_browser_errors(monkeypatch, master, selection):
    monkeypatch.setattr(render.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(PdfEngineMissing, match="did not write a PDF"):
        BrowserRenderer("msedge.exe").render(make_render_input(master, selection, "en"))

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 60)
    monkeypatch.setattr(render.subprocess, "run", slow)
    with pytest.raises(PdfEngineMissing, match="within"):
        BrowserRenderer("msedge.exe").render(make_render_input(master, selection, "en"))

    monkeypatch.setattr(render, "find_browser", lambda: None)
    with pytest.raises(PdfEngineMissing, match="no Edge or Chrome"):
        BrowserRenderer()


def test_make_renderer(monkeypatch):
    monkeypatch.setattr(render, "find_browser", lambda: "edge.exe")
    assert isinstance(make_renderer("auto"), BrowserRenderer)
    assert isinstance(make_renderer("weasyprint"), WeasyRenderer)
    monkeypatch.setattr(render, "find_browser", lambda: None)
    assert isinstance(make_renderer("auto"), WeasyRenderer)  # fallback


@pytest.mark.skipif(find_browser() is None, reason="no Edge or Chrome installed")
def test_real_browser_pdf(master, selection):
    r = BrowserRenderer().render(make_render_input(master, selection, "pt"))
    assert r.pdf.startswith(b"%PDF") and r.pages >= 1
