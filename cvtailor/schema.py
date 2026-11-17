"""Pydantic models for the master file, selections, variants and job meta.

Structural typing lives here. Cross-field rules (unique ids, tag format, dates,
limits...) live in master.check_master so that every problem is reported at once.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# languages renders last, one line per category, after the working/familiar lines.
TIERS = ("hands_on", "working", "familiar", "languages")
SECTIONS = ("summary", "experience", "projects", "skills", "education")
Lang = Literal["en", "pt"]
Section = Literal["summary", "experience", "projects", "skills", "education"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _as_list(v: Any) -> Any:
    # alt_of accepts one id or a list of ids
    return [v] if isinstance(v, str) else ([] if v is None else v)


def _coerce_date(v: Any) -> Any:
    # YAML turns 2017 into int and 2017-01-01 into a date; keep them as text so the
    # format check in master.py can report them properly.
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, _dt.date)):
        return str(v)
    return v


# ---------------------------------------------------------------- master

class Bilingual(_Strict):
    en: str
    pt: str

    def get(self, lang: str) -> str:
        return self.pt if lang == "pt" else self.en


# A string used as-is in both languages, or an {en, pt} pair.
Text = Union[str, Bilingual]


def text_of(value: Optional[Text], lang: str = "en") -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.get(lang)


class Link(_Strict):
    label: Text  # the visible text; the PDF links it to url
    url: str


class Profile(_Strict):
    name: str
    headline: Optional[Text] = None  # line under the name, e.g. the target role
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[Link] = Field(default_factory=list)


class Summary(_Strict):
    id: str
    tags: list[str]
    en: str
    pt: str
    headline: Optional[Text] = None  # replaces profile.headline when this summary is selected
    # Alternative versions: never selected together with these ids, or with other
    # alternatives of the same ids.
    alt_of: list[str] = Field(default_factory=list)
    # Set only on approved generated items merged in from the facts table.
    origin: Optional[Literal["rephrase", "merge", "new"]] = None
    based_on: list[str] = Field(default_factory=list)

    coerce_alt = field_validator("alt_of", mode="before")(_as_list)


class Bullet(_Strict):
    id: str
    tags: list[str]
    en: str
    pt: str
    alt_of: list[str] = Field(default_factory=list)  # alternative versions of these bullets
    origin: Optional[Literal["rephrase", "merge", "new"]] = None
    based_on: list[str] = Field(default_factory=list)

    coerce_alt = field_validator("alt_of", mode="before")(_as_list)


class DescriptionAlt(_Strict):
    """An alternative to a job's or project's description; the selection picks one."""
    id: str
    tags: list[str]
    en: str
    pt: str
    # Bullets of the same job or project whose content this description already covers;
    # they are never selected together with it.
    alt_of: list[str] = Field(default_factory=list)

    coerce_alt = field_validator("alt_of", mode="before")(_as_list)


class Job(_Strict):
    id: str
    title: Bilingual
    company: Optional[str] = None  # omit for freelance work and similar
    start: str
    end: Optional[str] = None
    location: Optional[str] = None
    description: Optional[Text] = None  # context line shown above the bullets
    description_alts: list[DescriptionAlt] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)

    coerce_dates = field_validator("start", "end", mode="before")(_coerce_date)


class Project(_Strict):
    id: str
    name: Text
    tags: list[str]
    start: Optional[str] = None
    end: Optional[str] = None
    links: list[Link] = Field(default_factory=list)
    description: Optional[Text] = None  # context line shown above the bullets
    description_alts: list[DescriptionAlt] = Field(default_factory=list)
    bullets: list[Bullet] = Field(default_factory=list)  # selectable like job bullets

    coerce_dates = field_validator("start", "end", mode="before")(_coerce_date)


class Education(_Strict):
    id: str
    institution: Text
    degree: Bilingual
    note: Optional[Text] = None  # e.g. honours, shown after the institution
    start: str
    end: Optional[str] = None
    expected: bool = False  # end is a planned date: rendered with "(expected)"
    # Optional detail lines (e.g. coursework), selectable like project bullets and
    # rendered in italics under the entry.
    bullets: list[Bullet] = Field(default_factory=list)

    coerce_dates = field_validator("start", "end", mode="before")(_coerce_date)


SkillItem = Text
skill_item_text = text_of


class SkillCategory(_Strict):
    id: str
    category: Bilingual
    items: list[SkillItem]
    # Alternative groupings: never selected together with these categories. Skills may
    # repeat between a category and its alternatives.
    alt_of: list[str] = Field(default_factory=list)

    coerce_alt = field_validator("alt_of", mode="before")(_as_list)


class Skills(_Strict):
    hands_on: list[SkillCategory] = Field(default_factory=list)
    working: list[SkillCategory] = Field(default_factory=list)
    familiar: list[SkillCategory] = Field(default_factory=list)
    languages: list[SkillCategory] = Field(default_factory=list)  # spoken languages, rendered last

    def tier(self, name: str) -> list[SkillCategory]:
        return getattr(self, name)


class Limits(_Strict):
    min_bullets_per_job: int = 1
    max_bullets_per_job: int = 5
    max_bullets_per_project: int = 5
    max_total_bullets: int = 14  # job and project bullets together
    max_projects: int = 2
    max_pages: int = 1


class Layout(_Strict):
    # Section order on the page. A section left out is not rendered.
    sections: list[Section] = Field(default_factory=lambda: list(SECTIONS))


class Master(_Strict):
    profile: Profile
    summaries: list[Summary]
    jobs: list[Job]
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    tag_aliases: dict[str, list[str]] = Field(default_factory=dict)
    limits: Limits = Field(default_factory=Limits)
    layout: Layout = Field(default_factory=Layout)

    @field_validator("tag_aliases", mode="before")
    @classmethod
    def _none_aliases(cls, v: Any) -> Any:
        return {} if v is None else v


# ---------------------------------------------------------------- selection

class SkillOrder(_Strict):
    hands_on: list[str] = Field(default_factory=list)
    working: list[str] = Field(default_factory=list)
    familiar: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _nulls(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {k: ([] if val is None else val) for k, val in v.items()}
        return v


class Selection(_Strict):
    summary_id: str
    # Keyed by job id (every job), project id (selected projects; may be empty) and
    # education id (detail lines; optional).
    bullets: dict[str, list[str]]
    project_ids: list[str] = Field(default_factory=list)
    skill_category_order: SkillOrder = Field(default_factory=SkillOrder)
    # Job or project id -> id of one of its description_alts. Absent: the default description.
    descriptions: dict[str, str] = Field(default_factory=dict)
    # Skill category id -> the subset of its own items to show, in this order, each
    # given by its exact English text. Absent: every item, in master order.
    skill_items: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _nulls(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            if v.get("project_ids") is None:
                v["project_ids"] = []
            if v.get("skill_category_order") is None:
                v["skill_category_order"] = {}
            if isinstance(v.get("bullets"), dict):
                v["bullets"] = {k: ([] if b is None else b) for k, b in v["bullets"].items()}
            if v.get("skill_items") is None:
                v["skill_items"] = {}
            elif isinstance(v["skill_items"], dict):  # a key with every item deleted
                v["skill_items"] = {k: ([] if i is None else i) for k, i in v["skill_items"].items()}
            if v.get("descriptions") is None:
                v["descriptions"] = {}
            elif isinstance(v["descriptions"], dict):  # "default" means: no alternative
                v["descriptions"] = {k: d for k, d in v["descriptions"].items()
                                     if d not in (None, "", "default")}
        return v

    def total_bullets(self) -> int:
        return sum(len(b) for b in self.bullets.values())


class LLMSelection(Selection):
    unmatched_requirements: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    def core(self) -> Selection:
        return Selection.model_validate(
            self.model_dump(exclude={"unmatched_requirements", "suggestions"})
        )


# ---------------------------------------------------------------- meta / variants / render

class Meta(BaseModel):
    company: str
    role_title: str
    language: Lang = "en"
    job_url: Optional[str] = None

    @field_validator("language", mode="before")
    @classmethod
    def _lang(cls, v: Any) -> str:
        v = str(v or "").strip().lower()
        return "pt" if v in ("pt", "pt-br", "pt_br", "portuguese") else "en"


class Variant(_Strict):
    name: str
    created_at: str
    source_application_id: Optional[int] = None
    job_tags: list[str]
    selection: Selection
    item_hashes: dict[str, str] = Field(default_factory=dict)


class RenderInput(_Strict):
    """Everything the template may see. Meta, suggestions, unmatched requirements and
    job tags are deliberately absent, so they cannot reach the rendered CV."""

    master: Master
    selection: Selection
    lang: Lang
    i18n: dict[str, Any]
