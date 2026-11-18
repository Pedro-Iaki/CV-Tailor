"""The languages a CV can be written in.

The master file names its languages (`languages: {primary: en, secondary: es}`; without
that key, English and Brazilian Portuguese). Every text is keyed by language code
(`en: ...`, `es: ...`). The primary language is required on every item; it is the one
the selector reads and the default for postings in other languages. The secondary
language is optional: items without it are left out of CVs in that language.

The fixed words of the CV (headings, months, "Present") for each language live in
templates/i18n.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str    # key in master.yaml and i18n.yaml
    name: str    # shown in the app
    html: str    # <html lang> of the rendered CV
    prompt: str  # how prompts name it
    style: str = ""  # extra writing rule for drafted text in this language


LANGUAGES: dict[str, Language] = {l.code: l for l in [
    Language("en", "English", "en", "English"),
    Language("pt", "Portuguese", "pt-BR", "Brazilian Portuguese",
             'natural Brazilian Portuguese, with first person verb forms as in the original (e.g. "Construí")'),
    Language("es", "Spanish", "es", "Spanish"),
    Language("fr", "French", "fr", "French"),
    Language("de", "German", "de", "German"),
    Language("it", "Italian", "it", "Italian"),
]}

DEFAULT_PRIMARY = "en"
LEGACY_SECONDARY = "pt"  # a master file without a `languages` key is English + Portuguese

# Other ways a model or a person may name a language.
_ALIASES = {"pt-br": "pt", "pt_br": "pt", "portugues": "pt", "português": "pt", "brazilian portuguese": "pt",
            "en-us": "en", "en-gb": "en", "espanol": "es", "español": "es", "castellano": "es",
            "francais": "fr", "français": "fr", "deutsch": "de", "italiano": "it"}


def normalize(value: object) -> str:
    """A language code from a code, a locale or a name ("pt-BR", "Spanish" -> "pt", "es").
    Unknown values come back lowercased, for the caller to reject."""
    v = str(value or "").strip().lower()
    if v in LANGUAGES:
        return v
    if v in _ALIASES:
        return _ALIASES[v]
    for lang in LANGUAGES.values():
        if v in (lang.name.lower(), lang.prompt.lower(), lang.html.lower()):
            return lang.code
    return v


def name(code: str) -> str:
    lang = LANGUAGES.get(code)
    return lang.name if lang else code.upper()


def prompt_name(code: str) -> str:
    lang = LANGUAGES.get(code)
    return lang.prompt if lang else code


def describe(codes: list[str], conj: str = "and") -> str:
    """'"en" (English) and "pt" (Brazilian Portuguese)' for prompts."""
    parts = [f'"{c}" ({prompt_name(c)})' for c in codes]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" {conj} " + parts[-1]


def choices() -> list[dict[str, str]]:
    return [{"code": l.code, "name": l.name} for l in LANGUAGES.values()]
