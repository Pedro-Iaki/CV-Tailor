from cvtailor.config import app_folder_name, company_key, filename_part, folder_slug, pdf_filename
from cvtailor.editor import parse_annotated, to_annotated_yaml


def test_filename_parts():
    assert filename_part("João Silva") == "Joao_Silva"
    assert filename_part("Sr. Data Engineer (Remote) / PJ") == "Sr_Data_Engineer_Remote_PJ"
    assert len(filename_part("x" * 200)) == 60


def test_folder_and_pdf_names():
    assert folder_slug("Itaú Unibanco S.A.") == "itau-unibanco-s-a"
    assert app_folder_name(42, "2026-11-16", "Acme") == "0042_2026-11-16_acme"
    assert pdf_filename("{name}_CV_{role}_{LANG}", name="Full Name", role="Senior Data Engineer",
                        lang="pt") == "Full_Name_CV_Senior_Data_Engineer_PT.pdf"


def test_company_key_normalized():
    assert company_key("  ACME Ltda. ", "Engenheiro de Dados") == company_key("acme ltda", "engenheiro  de dados")


def test_annotated_yaml_round_trip(master, selection):
    text = to_annotated_yaml(selection, master)
    assert "summary_id: sum-platform   # Data engineer with 4 years" in text
    assert "    # - acme-streaming-06   # Designed a Spark Structured Streaming" in text
    assert parse_annotated(text) == selection


def test_annotated_yaml_empty_sections(master, selection_dict):
    from cvtailor.schema import Selection
    selection_dict["project_ids"] = []
    selection_dict["skill_category_order"]["familiar"] = []
    sel = Selection.model_validate(selection_dict)
    assert parse_annotated(to_annotated_yaml(sel, master)) == sel
