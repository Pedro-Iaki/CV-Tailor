"""Paths, environment settings and output naming."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = ROOT / "templates"

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_PDF_NAME = "{name}_CV_{role}_{LANG}"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env", override=False)


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    master_path: Path
    variants_dir: Path
    db_path: Path
    model: str
    effort: str | None
    route_threshold: float
    pdf_name: str
    allow_generation: bool = False  # opt-in: LLM-drafted text, approved item by item
    llm_base_url: str | None = None  # set: use a local OpenAI-compatible server instead of Anthropic
    llm_api_key: str | None = None
    llm_timeout: float = 600.0

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv()
        data_dir = Path(os.environ.get("CV_DATA_DIR") or ROOT / "data")
        output_dir = Path(os.environ.get("CV_OUTPUT_DIR") or ROOT / "output")
        master = Path(os.environ.get("CV_MASTER") or data_dir / "master.yaml")
        try:
            threshold = float(os.environ.get("ROUTE_THRESHOLD") or 0.6)
        except ValueError:
            threshold = 0.6
        effort = os.environ.get("CV_EFFORT", "low").strip() or None
        try:
            timeout = float(os.environ.get("CV_LLM_TIMEOUT") or 600)
        except ValueError:
            timeout = 600.0
        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            master_path=master,
            variants_dir=data_dir / "variants",
            db_path=data_dir / "applications.db",
            model=os.environ.get("CV_MODEL") or DEFAULT_MODEL,
            effort=effort,
            route_threshold=threshold,
            pdf_name=os.environ.get("CV_PDF_NAME") or DEFAULT_PDF_NAME,
            allow_generation=os.environ.get("CV_ALLOW_GENERATION", "").strip().lower()
            in ("1", "true", "yes", "on"),
            llm_base_url=os.environ.get("CV_LLM_BASE_URL", "").strip() or None,
            llm_api_key=os.environ.get("CV_LLM_API_KEY", "").strip() or None,
            llm_timeout=timeout,
        )


# ---------------------------------------------------------------- naming

def strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c))


def filename_part(text: str, max_len: int = 60) -> str:
    """'João Silva' -> 'Joao_Silva'. Any non [A-Za-z0-9] run becomes '_'."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", strip_accents(text or "")).strip("_")
    return s[:max_len].rstrip("_") or "x"


def folder_slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", strip_accents(text or "").lower()).strip("-")
    return s[:max_len].rstrip("-") or "x"


def app_folder_name(app_id: int, date: str, company: str) -> str:
    return f"{app_id:04d}_{date}_{folder_slug(company)}"


def pdf_filename(template: str, *, name: str, role: str, lang: str, company: str = "") -> str:
    fields = {
        "name": filename_part(name),
        "role": filename_part(role),
        "company": filename_part(company),
        "LANG": lang.upper(),
        "lang": lang.lower(),
    }
    try:
        base = template.format(**fields)
    except (KeyError, IndexError, ValueError):
        base = DEFAULT_PDF_NAME.format(**fields)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")[:120]
    return f"{base}.pdf"


def normalize_key(text: str) -> str:
    """Lowercase, accent-free, punctuation-free text for comparisons."""
    s = re.sub(r"[^a-z0-9]+", " ", strip_accents(text or "").lower())
    return " ".join(s.split())


def company_key(company: str, role: str) -> str:
    return f"{normalize_key(company)}|{normalize_key(role)}"
