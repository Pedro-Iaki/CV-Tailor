import copy
import html as htmllib
import io
import re

import pytest

from cvtailor.pipeline import NewOptions, run_new
from cvtailor.render import load_i18n, make_render_input, render_html
from tests.conftest import FakeClient, FakeRenderer, needs_weasyprint

WORD = re.compile(r"\w+", re.UNICODE)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def allowed_words(master) -> set[str]:
    words = set()
    sources = list(_strings(master.model_dump())) + list(_strings(load_i18n()))
    for s in sources:
        words.update(w.lower() for w in WORD.findall(s))
    return words


def html_text_lines(html: str) -> list[str]:
    body = html[html.index("<body>"):]
    body = re.sub(r"<[^>]+>", "\n", body)
    return [l.strip() for l in htmllib.unescape(body).splitlines() if l.strip()]


def untraceable(lines, allowed):
    bad = []
    for line in lines:
        for w in WORD.findall(line):
            if w.lower() not in allowed:
                bad.append((w, line))
    return bad


@pytest.mark.parametrize("lang", ["en", "pt"])
def test_html_traceability(master, selection, lang):
    html = render_html(make_render_input(master, selection, lang))
    assert untraceable(html_text_lines(html), allowed_words(master)) == []


def test_sections_order_and_i18n(master, selection):
    en = render_html(make_render_input(master, selection, "en"))
    pt = render_html(make_render_input(master, selection, "pt"))
    order = ["Summary", "Experience", "Projects", "Skills", "Education"]
    assert [en.index(f"<h2>{h}</h2>") for h in order] == sorted(en.index(f"<h2>{h}</h2>") for h in order)
    assert "Experiência" in pt and "Atual" in pt and "01/2023" in pt
    assert "Present" in en and "01/2023" in en
    # reverse chronological: Acme (current) before DataCorp before the internship
    assert en.index("Acme Pagamentos") < en.index("DataCorp Analytics") < en.index("Research Intern")
    # all jobs are listed and bullet order follows the selection
    assert en.index("Built Databricks pipelines") < en.index("Migrated 40 legacy tables")


def test_skills_blocks(master, selection_dict):
    from cvtailor.schema import Selection
    d = copy.deepcopy(selection_dict)
    d["skill_category_order"]["working"] = ["skills-working-streaming", "skills-working-orchestration"]
    d["skill_category_order"]["familiar"] = []
    html = render_html(make_render_input(master, Selection.model_validate(d), "en"))
    assert "<strong>Processing:</strong> PySpark, Spark SQL, Delta Lake, pandas" in html
    assert "<strong>Working knowledge:</strong> Kafka, Spark Structured Streaming, Airflow, Power BI" in html
    assert "Familiar with" not in html  # empty block not rendered
    assert "Orchestration" not in html  # working/familiar category labels are merged away


def test_header_links_and_entry_details(master, selection_dict):
    from cvtailor.schema import Selection
    d = copy.deepcopy(selection_dict)
    d["project_ids"] = ["proj-airflow-demo"]
    d["bullets"]["proj-airflow-demo"] = ["lakehouse-tests-02"]
    sel = Selection.model_validate(d)
    en = render_html(make_render_input(master, sel, "en"))
    pt = render_html(make_render_input(master, sel, "pt"))
    assert '<p class="headline">Data Engineer</p>' in en and "Engenheira de Dados" in pt
    assert '<a href="https://github.com/example">github.com/example</a>' in en
    assert "Payments company processing card" in en  # job description
    # one project: singular heading, bilingual link label, dates, description, chosen bullets only
    assert "<h2>Project</h2>" in en and "<h2>Projeto</h2>" in pt
    assert '<a href="https://github.com/example/lakehouse-demo">Código</a>' in pt
    assert "06/2022 – 09/2022" in en
    assert "End-to-end lakehouse demo" in en
    assert "Added pytest data-quality checks" in en and "Wrote Airflow DAGs" not in en
    assert en.index("Added pytest data-quality checks") < en.index("<h2>Skills</h2>")
    # education note
    assert "Universidade Federal · Honours" in en and "Láurea acadêmica" in pt


def test_layout_order_hidden_sections_and_expected(master_dict, selection):
    from cvtailor.master import parse_master
    master_dict["layout"] = {"sections": ["summary", "projects", "experience", "education"]}
    master_dict["education"][0].update({"start": "2026-05", "end": "2027-10", "expected": True})
    del master_dict["jobs"][0]["company"]
    m = parse_master(master_dict)
    html = render_html(make_render_input(m, selection, "en"))
    heads = re.findall(r"<h2>(.*?)</h2>", html)
    assert heads == ["Summary", "Projects", "Experience", "Education"]  # skills hidden
    assert "05/2026 – 10/2027 (expected)" in html
    pt = render_html(make_render_input(m, selection, "pt"))
    assert "(previsão)" in pt
    assert "<strong>Data Engineer</strong>" in html  # a job without a company


def test_no_layout_tables_images_or_icons(master, selection):
    html = render_html(make_render_input(master, selection, "en")).lower()
    for tag in ("<table", "<img", "<svg", "<i ", "font-awesome"):
        assert tag not in html


def test_sentinels_never_reach_html(make_ctx, selection_dict, job_text, quiet_io):
    llm_out = copy.deepcopy(selection_dict)
    llm_out["unmatched_requirements"] = ["SENTINELUNMATCHED"]
    llm_out["suggestions"] = ["SENTINELSUGGESTION"]
    client = FakeClient(
        meta={"company": "SENTINELCOMPANY", "role_title": "SENTINELROLE", "language": "en", "job_url": None},
        selections=[llm_out],
    )
    renderer = FakeRenderer()
    ctx = make_ctx(client=client, renderer=renderer)
    res = run_new(ctx, job_text, NewOptions(force_generate=True), quiet_io)
    assert res.app_id == 1
    assert renderer.htmls
    for h in renderer.htmls:
        for s in ("SENTINELUNMATCHED", "SENTINELSUGGESTION", "SENTINELCOMPANY", "SENTINELROLE"):
            assert s not in h
    assert "SENTINELROLE" in res.pdf_path.name  # role is used in the filename only


def test_render_input_rejects_extra_fields(master, selection):
    from pydantic import ValidationError
    from cvtailor.schema import RenderInput
    with pytest.raises(ValidationError):
        RenderInput(master=master, selection=selection, lang="en", i18n={}, suggestions=["x"])


@needs_weasyprint
@pytest.mark.parametrize("lang", ["en", "pt"])
def test_pdf_one_page_and_traceable(master, selection, lang):
    from pypdf import PdfReader
    from cvtailor.render import WeasyRenderer
    r = WeasyRenderer().render(make_render_input(master, selection, lang))
    assert r.pages == 1
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(r.pdf)).pages)
    lines = [l for l in text.splitlines() if l.strip()]
    assert untraceable(lines, allowed_words(master)) == []
