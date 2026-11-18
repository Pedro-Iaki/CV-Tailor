"""LLM calls: job meta extraction and id-only content selection.

Calls go to the Anthropic API by default, or to a local OpenAI-compatible server
(see local_llm.py) when CV_LLM_BASE_URL is set.

The model never writes resume text: the selection schema is built from the master so
every id field is an enum of real ids, and the result is re-validated in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from . import choices
from .master import build_llm_index
from .schema import TIERS, LLMSelection, Master, Meta
from .validate import validate_selection

SELECTION_SYSTEM_PROMPT = """\
You are a resume content selector. You do not write resume text.

You receive (1) a job description and (2) an index of the candidate's master
resume: items with an id, tags, and English text, plus the jobs, projects,
skill categories, and limits. Jobs and projects both have bullets.

Your job is to choose which existing items to include and in what order so the
resume best matches the job. You may ONLY output ids that appear in the index.

Hard rules:
- Never create, edit, or rephrase any text. Output ids only.
- Include EVERY job in the index under "bullets", each with at least the
  minimum number of bullets, so there are no gaps in the work history.
- Projects are optional. Pick them in "project_ids". A picked project's bullets
  go under "bullets" keyed by the project id (from none up to the project
  maximum). A project you do not pick gets an empty list.
- "education_details" lists optional detail lines (such as coursework) of an
  education entry. Use them only when the job asks for those specific skills; the
  degree name already implies its general subjects. Put the ones that help under
  "bullets" keyed by the education id; otherwise give it an empty list.
- Some items are alternative versions of each other: an item's "never_with" lists
  the ids it must never be selected together with (anywhere in the selection). Pick
  the version that fits the job best. For skills, alternatives are whole layouts:
  either the categories on one side or the other, never a mix.
- A summary may carry a "headline" (the line under the name); picking the summary
  picks its headline too. If the index has "headline_options", put one id in
  "headline_id": keep "default" unless the job's own title is the role another
  option names.
- A job or project with "description_options" needs one option id under
  "descriptions" (keyed by the job or project id); "default" keeps the default.
- Never move a skill category between tiers. You may only reorder categories
  within their own tier or omit them. The "languages" tier holds spoken languages.
  A category marked "required" must be listed unless one of its alternatives is;
  one with "min_items" shows at least that many of its items when listed.
- Never include an id that is not in the index, and never list a bullet under
  a job or project it does not belong to.
- Respect the limits provided (min and max bullets per job, max bullets per
  project, max total bullets counting job and project bullets, max projects).
  The total is one budget shared by every entry: decide how many bullets each
  entry gets before ranking them, give more to the entries most relevant to the
  job, and add up the bullets you kept before answering.
- Size each job and project by its relevance to this job: the most relevant
  entries get up to their maximum, the least relevant only the minimum. A project
  whose stack or subject the job asks for should get nearly all of its bullets;
  one unrelated to the job is better left out.
- Prefer bullets that directly show a requirement the job names (for example, a
  CI/CD bullet when it mentions deployment or pipelines) over general ones.
- Order bullets within each job and project, and the projects themselves, by
  relevance to the job description, most relevant first.
- If the job asks for something with no matching item, do not invent coverage.
  List it under "unmatched_requirements".
- Anything you would want to add or change goes in "suggestions" as a short
  plain-text note. Suggestions are for the human and are never rendered.

Return only a single JSON object matching the provided schema.
"""

META_SYSTEM_PROMPT = """\
You extract metadata from a job description. Return only JSON matching the schema.
- company: the hiring company's name as written (use the recruiting agency only if
  the employer is not named).
- role_title: the job title as written in the posting.
- language: the language the job description is written in: "en" or "pt".
  Anything other than Portuguese is "en".
- job_url: a URL only if one literally appears in the text and points to the posting,
  else null.
"""

META_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "role_title": {"type": "string"},
        "language": {"type": "string", "enum": ["en", "pt"]},
        "job_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["company", "role_title", "language", "job_url"],
    "additionalProperties": False,
}


PROPOSAL_SYSTEM_PROMPT = """\
You draft optional wording improvements for a candidate's resume. A human reviews
every draft one by one; nothing you write is used unless they approve it.

You receive the job description, the resume items currently selected for this job,
requirements the selection does not cover, the reviewer's earlier feedback, and an
index of every verified item (id, tags, English text).

Propose at most {max_items} drafts, most valuable first. Each draft is one of:
- "rephrase": rewrite ONE verified item (based_on = [its id]) so it uses the job's
  vocabulary. Keep every fact, number, tool and scope; add nothing.
- "merge": combine 2-3 bullets of the SAME job or project (based_on = their ids) into one
  tighter bullet, keeping only facts present in the sources.
- "new": a bullet for a requirement the resume does not cover. This is an
  UNVERIFIED claim the candidate must confirm. Put in "rationale" exactly what they
  must confirm. based_on may list items that give context, or be empty.

Rules:
- kind is "bullet" (job_id = the job or project it belongs to) or "summary" (job_id = "").
  Summaries can only be rephrased or new, not merged.
- Write every draft in English ("en") and Brazilian Portuguese ("pt") with the same
  meaning. Bullets: one sentence, past tense, starting with a verb, no first person,
  at most about 200 characters.
- In rephrase and merge drafts never introduce a number, tool, technology, company,
  or result that is not in the based_on items. Drafts that do are treated as
  unverified new claims.
- "replaces": ids from the current selection this draft should replace (usually its
  based_on ids that are selected); empty for "new".
- "tags": lowercase-hyphenated tags; reuse tags from the index where they apply.
- Follow the reviewer's earlier feedback. Do not re-propose rejected ideas.
- If nothing is worth proposing, return an empty list.
"""

REVISE_SYSTEM_PROMPT = """\
You revise ONE resume draft according to the reviewer's comment. The same rules
apply as when it was drafted:
- Keep "en" (English) and "pt" (Brazilian Portuguese) equivalent.
- For "rephrase" and "merge", use only facts present in the based_on items; never
  introduce numbers, tools, companies or results that are not in them. If the
  comment points to other verified items, you may add their ids to based_on and use
  their facts.
- If the comment asks for something that would add an unverified claim, set origin
  to "new" and say in "rationale" what the candidate must confirm.
- Keep the same kind and job_id unless the comment clearly asks otherwise.
Return the revised draft only.

Index of verified items follows.
"""


class LLMError(Exception):
    pass


class Truncated(LLMError):
    """The answer was cut off at max_tokens (often a model thinking for too long)."""


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.output_tokens + other.output_tokens,
        )

    @classmethod
    def from_response(cls, resp: Any) -> "Usage":
        u = getattr(resp, "usage", None)
        if u is None:
            return cls()
        return cls(
            int(getattr(u, "input_tokens", 0) or 0),
            int(getattr(u, "cache_read_input_tokens", 0) or 0),
            int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            int(getattr(u, "output_tokens", 0) or 0),
        )

    def line(self) -> str:
        return (f"tokens: input {self.input_tokens}, cache read {self.cache_read_tokens}, "
                f"cache write {self.cache_write_tokens}, output {self.output_tokens}")


# ---------------------------------------------------------------- schema

def _enum_array(ids: list[str]) -> dict[str, Any]:
    if ids:
        return {"type": "array", "items": {"type": "string", "enum": list(ids)}}
    return {"type": "array", "items": {"type": "string"}}


def _sized(schema: dict[str, Any], lo: int, hi: int) -> dict[str, Any]:
    return {**schema, "minItems": lo, "maxItems": hi}


def build_selection_schema(master: Master, count_limits: bool = False) -> dict[str, Any]:
    """JSON schema whose id fields are enums of the master's real ids.

    count_limits adds minItems/maxItems from the master's limits. Local servers enforce
    them while decoding, which keeps small models inside the limits; the Anthropic API
    does not accept them, so they are off by default. The total-bullets limit spans
    several lists and is only checked by validate_selection.
    """
    lim = master.limits
    # Every job, plus every project and education entry that has bullets (an empty
    # list when not picked).
    optional = [o for o in [*master.projects, *master.education] if o.bullets]
    job_props = {j.id: _enum_array([b.id for b in j.bullets]) for j in master.jobs}
    job_props |= {o.id: _enum_array([b.id for b in o.bullets]) for o in optional}
    project_ids = _enum_array([p.id for p in master.projects])
    if count_limits:
        for j in master.jobs:
            job_props[j.id] = _sized(job_props[j.id], min(lim.min_bullets_per_job, len(j.bullets)),
                                     min(lim.max_bullets_per_job, len(j.bullets)))
        for p in master.projects:
            if p.bullets:
                job_props[p.id] = _sized(job_props[p.id], 0, min(lim.max_bullets_per_project, len(p.bullets)))
        for e in master.education:
            if e.bullets:
                job_props[e.id] = _sized(job_props[e.id], 0, len(e.bullets))
        project_ids = _sized(project_ids, 0, min(lim.max_projects, len(master.projects)))
    tier_props = {
        tier: _enum_array([c.id for c in master.skills.tier(tier)]) for tier in TIERS
    }
    if count_limits:  # an empty tier has no enum to constrain it, so cap it at zero
        for tier in TIERS:
            if not master.skills.tier(tier):
                tier_props[tier] = _sized(tier_props[tier], 0, 0)
    # One choice per job or project that has description alternatives.
    desc_props = {o.id: {"type": "string", "enum": ["default", *(d.id for d in o.description_alts)]}
                  for o in [*master.jobs, *master.projects] if o.description_alts}
    extra: dict[str, Any] = {}
    if master.profile.headline_alts:
        extra["headline_id"] = {"type": "string",
                                "enum": ["default", *(h.id for h in master.profile.headline_alts)]}
    if desc_props:
        extra["descriptions"] = {"type": "object", "properties": desc_props,
                                 "required": list(desc_props), "additionalProperties": False}
    return {
        "type": "object",
        "properties": {
            "summary_id": {"type": "string", "enum": [s.id for s in master.summaries]},
            "bullets": {
                "type": "object",
                "properties": job_props,
                "required": list(job_props),
                "additionalProperties": False,
            },
            "project_ids": project_ids,
            "skill_category_order": {
                "type": "object",
                "properties": tier_props,
                "required": list(TIERS),
                "additionalProperties": False,
            },
            **extra,
            "unmatched_requirements": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "summary_id", "bullets", "project_ids", "skill_category_order", *extra,
            "unmatched_requirements", "suggestions",
        ],
        "additionalProperties": False,
    }


def _draft_schema(master: Master) -> dict[str, Any]:
    entries = [*master.jobs, *master.projects]
    item_ids = [s.id for s in master.summaries] + [b.id for e in entries for b in e.bullets]
    ids = _enum_array(item_ids)
    props = {
        "kind": {"type": "string", "enum": ["bullet", "summary"]},
        "job_id": {"type": "string", "enum": [e.id for e in entries] + [""]},
        "origin": {"type": "string", "enum": ["rephrase", "merge", "new"]},
        "based_on": ids,
        "replaces": ids,
        "tags": {"type": "array", "items": {"type": "string"}},
        "en": {"type": "string"},
        "pt": {"type": "string"},
        "rationale": {"type": "string"},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_proposal_schema(master: Master) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"proposals": {"type": "array", "items": _draft_schema(master)}},
        "required": ["proposals"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------- client

def _text_of(resp: Any) -> str:
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        raise LLMError("the model declined the request (stop_reason=refusal)")
    if stop == "max_tokens":
        out = getattr(getattr(resp, "usage", None), "output_tokens", None)
        raise Truncated("the model hit max_tokens before finishing the JSON"
                        + (f" ({out} output tokens)" if out else ""))
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return block.text
        if getattr(block, "type", None) == "tool_use":
            return json.dumps(block.input)
    raise LLMError("the response contained no JSON output")


class LLM:
    """Calls go to the Anthropic API, or to `client` when one is given (for example a
    local_llm.LocalClient). count_limits puts the bullet limits into the schema;
    choices uses the choice format of choices.py for the selection call; max_output caps
    every call's max_tokens (local models: a runaway answer fails fast instead of
    running for many minutes, and stays inside the server's context window)."""

    def __init__(self, model: str, effort: Optional[str] = "low", client: Any = None,
                 count_limits: bool = False, choices: bool = False,
                 max_output: Optional[int] = None):
        self.model = model
        self.effort = effort
        self._client = client
        self.count_limits = count_limits
        self.choices = choices
        self.max_output = max_output

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError("the 'anthropic' package is not installed") from e
            self._client = anthropic.Anthropic(max_retries=2)
        return self._client

    def _call(self, *, system: list[dict], user: str, schema: Optional[dict], max_tokens: int,
              reasoning: Optional[str] = None) -> Any:
        """One call. schema=None asks for free text. reasoning overrides the local
        server's reasoning_effort for this call (ignored by the Anthropic API)."""
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.effort:
            output_config["effort"] = self.effort
        if self.max_output:
            max_tokens = min(max_tokens, self.max_output)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if reasoning is not None and getattr(self.client, "accepts_reasoning", False):
            kwargs["reasoning"] = reasoning
        if output_config:
            kwargs["output_config"] = output_config
        try:
            try:
                return self.client.messages.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                if schema is None or not _is_structured_output_unsupported(e):
                    raise
            kwargs.pop("output_config", None)
            # Fallback for models without native structured outputs: a tool whose input
            # schema is the selection schema.
            return self.client.messages.create(
                tools=[{
                    "name": "submit",
                    "description": "Submit the result as structured JSON.",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": "submit"},
                **kwargs,
            )
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 - API, auth and network errors
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            if "authentication" in msg.lower() or "api_key" in msg.lower():
                msg += " (set ANTHROPIC_API_KEY in .env or the environment)"
            raise LLMError(f"API call failed: {type(e).__name__}: {msg}") from e

    # ------------------------------------------------------------ meta

    def extract_meta(self, job_text: str) -> tuple[Meta, Usage]:
        resp = self._call(
            system=[{"type": "text", "text": META_SYSTEM_PROMPT}],
            user="JOB DESCRIPTION:\n" + job_text,
            schema=META_SCHEMA,
            # Haiku 5.5 thinks by default (adaptive), and thinking counts against max_tokens.
            max_tokens=4096,
        )
        usage = Usage.from_response(resp)
        try:
            meta = Meta.model_validate(json.loads(_text_of(resp)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise LLMError(f"meta response was not valid: {e}") from None
        if not meta.company.strip() or not meta.role_title.strip():
            raise LLMError("meta response had an empty company or role")
        if meta.job_url and meta.job_url not in job_text:
            meta.job_url = None
        return meta, usage

    # ------------------------------------------------------------ selection

    def write(self, system: str, user: str, max_tokens: int = 16000,
              reasoning: Optional[str] = None) -> tuple[str, Usage]:
        """A free-text call (no schema): analysis and review notes for careful mode.
        The text is only ever shown to the model again or to the user, never rendered."""
        resp = self._call(system=[{"type": "text", "text": system}], user=user, schema=None,
                          max_tokens=max_tokens, reasoning=reasoning)
        return _text_of(resp).strip(), Usage.from_response(resp)

    def select(self, job_text: str, master: Master, context: str = "") -> tuple[LLMSelection, Usage]:
        """Select content by id. Validates, retries once with the errors, then aborts.
        `context` (careful mode: hints, analysis, review) follows the job description."""
        schema = build_selection_schema(master, count_limits=self.count_limits)
        prompt, index = SELECTION_SYSTEM_PROMPT, None
        if self.choices:
            ch = choices.analyse(master)
            schema = choices.build_schema(master, schema, self.count_limits, ch)
            prompt, index = choices.system_prompt(prompt), choices.build_index(master, ch)
        system = [
            {"type": "text", "text": prompt},
            {
                "type": "text",
                "text": "MASTER RESUME INDEX (JSON):\n" + (index or build_llm_index(master)),
                "cache_control": {"type": "ephemeral"},
            },
        ]
        base_user = "JOB DESCRIPTION:\n" + job_text + (f"\n\n{context.strip()}" if context.strip() else "")
        usage = Usage()
        errors: list[str] = []
        for attempt in range(2):
            user = base_user
            if errors:
                user += (
                    "\n\nYour previous answer was rejected for these reasons. "
                    "Return a corrected selection:\n- " + "\n- ".join(errors)
                )
            resp = self._call(system=system, user=user, schema=schema, max_tokens=16000)
            usage = usage.add(Usage.from_response(resp))
            try:
                text = _text_of(resp)
            except Truncated as e:  # counts as a failed attempt: one retry, asking for brevity
                errors = [f"{e}. Keep your reasoning short and answer with the JSON."]
                continue
            try:
                data = json.loads(text)
                if self.choices and isinstance(data, dict):
                    data = choices.decode(data, master)
                sel = LLMSelection.model_validate(data)
            except json.JSONDecodeError as e:
                errors = [f"output was not valid JSON: {e}"]
                continue
            except ValidationError as e:
                errors = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
                continue
            errors = validate_selection(sel.core(), master)
            if not errors:
                sel.unmatched_requirements = [s.strip() for s in sel.unmatched_requirements if s.strip()][:5]
                sel.suggestions = [s.strip() for s in sel.suggestions if s.strip()][:3]
                # The schema makes the model send an empty list for every project or
                # education entry it gave no bullets; drop those so stored selections
                # stay canonical.
                jobs = {j.id for j in master.jobs}
                sel.bullets = {k: v for k, v in sel.bullets.items() if v or k in jobs}
                return sel, usage
        raise LLMError(
            "selection failed validation twice:\n  - " + "\n  - ".join(errors)
        )


    # ------------------------------------------------------------ opt-in text drafts

    def _index_block(self, master: Master) -> dict[str, Any]:
        return {"type": "text", "text": "VERIFIED ITEM INDEX (JSON):\n" + build_llm_index(master),
                "cache_control": {"type": "ephemeral"}}

    def propose(self, job_text: str, master: Master, context: str,
                max_items: int = 5) -> tuple[list[dict], Usage]:
        """Draft wording proposals. Returned raw; generation.check_proposal validates them."""
        resp = self._call(
            system=[{"type": "text", "text": PROPOSAL_SYSTEM_PROMPT.format(max_items=max_items)},
                    self._index_block(master)],
            user="JOB DESCRIPTION:\n" + job_text + "\n\n" + context,
            schema=build_proposal_schema(master),
            max_tokens=16000,
        )
        usage = Usage.from_response(resp)
        try:
            data = json.loads(_text_of(resp))
        except json.JSONDecodeError as e:
            raise LLMError(f"proposal response was not valid JSON: {e}") from None
        return list(data.get("proposals") or [])[:max_items], usage

    def revise(self, master: Master, draft: dict, comment: str, sources: str) -> tuple[dict, Usage]:
        resp = self._call(
            system=[{"type": "text", "text": REVISE_SYSTEM_PROMPT}, self._index_block(master)],
            user=("DRAFT (JSON):\n" + json.dumps(draft, ensure_ascii=False)
                  + "\n\nSOURCE ITEMS:\n" + sources
                  + "\n\nREVIEWER COMMENT:\n" + comment),
            schema=_draft_schema(master),
            max_tokens=8000,
        )
        usage = Usage.from_response(resp)
        try:
            return json.loads(_text_of(resp)), usage
        except json.JSONDecodeError as e:
            raise LLMError(f"revision response was not valid JSON: {e}") from None


def _is_structured_output_unsupported(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    msg = str(e).lower()
    return status == 400 and ("output_config" in msg or "output_format" in msg
                              or "structured output" in msg) and "not supported" in msg
