"""Careful selection: think in free text first, select second, then review once.

The plain selection call reasons (if at all) inside grammar-constrained JSON, which
hurts small local models. Careful mode splits the work:

1. Relevance hints, computed in code: which items carry tags the posting matched,
   and (optional) which posting line each item is closest to by embeddings.
2. Analysis: a free-text pass (no schema) that maps requirements to item ids and
   picks a summary, layout and versions. A worked example, if the user keeps one in
   data/worked_examples.yaml, shows the kind of reasoning wanted.
3. Selection: the usual schema-constrained call, with hints and analysis as context.
4. Review: a free-text critique of that draft; unless it finds nothing to change,
   one more selection call with the review. If that revision fails validation, the
   first (valid) draft is kept.

Nothing here writes CV text: the free text only goes back to the model, or to the
user in verbose output. Every selection is validated as usual.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import choices
from .config import normalize_key
from .llm import LLM, LLMError, Truncated, Usage
from .master import build_llm_index, item_index, item_text
from .schema import TIERS, LLMSelection, Master

ANALYSIS_SYSTEM_PROMPT = """\
You are preparing to tailor a one-page resume to a job. You do not choose the final
selection yet and you never write resume text: you write a short analysis that the
selection step will follow.

Write plain text, at most about 250 words, with these parts:
1. Requirements: the 3 to 6 things this job cares about most.
2. Coverage: for each requirement, the item ids from the index that show it, best
   first, and how strongly (strong / partial / none).
3. Choices: which summary, which skill layout (a layout with a working-knowledge tier
   can show lower-level skills honestly), which bullet versions (short or long), which
   projects lead, and in what order.
4. Leave out: selected-by-default items that do not help this job.
Use the relevance hints as evidence, not as the answer.
"""

REVIEW_SYSTEM_PROMPT = """\
You review a draft resume selection against a job description. You never write resume
text. In at most about 150 words of plain text, list only concrete changes, citing ids:
- requirements the draft leaves uncovered although an item in the index covers them;
- selected items that do not help this job and take space;
- a better summary, skill layout, version or order, if there clearly is one.
If the draft is already good, answer exactly: NO CHANGES
"""

NO_CHANGES = "NO CHANGES"


@dataclass
class CarefulResult:
    selection: LLMSelection
    usage: Usage
    analysis: str
    review: str
    revised: bool
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- hints

def tag_hints(master: Master, matches: dict[str, str]) -> list[str]:
    """One line per item that carries a tag the posting matched, most tags first."""
    rows = []
    for id_, it in item_index(master).items():
        tags = [t for t in (getattr(it.obj, "tags", None) or []) if t in matches]
        if tags:
            rows.append((len(tags), id_, ", ".join(f'{t} ("{matches[t]}")' for t in tags)))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [f"- {id_}: {desc}" for _, id_, desc in rows]


def _post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embedding_hints(job_text: str, master: Master, model: str, base_url: str,
                    top: int = 12, timeout: float = 120.0) -> list[str]:
    """Items closest to a posting line, by cosine similarity of embeddings from an
    Ollama server (/api/embed). Returns [] if the server can't be used."""
    lines = [l.strip(" -•*\t") for l in job_text.splitlines() if len(l.strip()) > 3]
    idx = item_index(master)
    ids = [i for i, it in idx.items() if it.kind in ("summary", "bullet", "description", "skill_category")]
    if not lines or not ids:
        return []
    url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/api/embed"
    try:
        vecs = _post_json(url, {"model": model, "input": lines + [item_text(idx[i]) for i in ids]},
                          timeout)["embeddings"]
    except Exception:  # noqa: BLE001 - optional evidence; selection works without it
        return []

    def norm(v: list[float]) -> list[float]:
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    vecs = [norm(v) for v in vecs]
    line_vecs, item_vecs = vecs[:len(lines)], vecs[len(lines):]
    scored = []
    for id_, v in zip(ids, item_vecs):
        sims = [sum(a * b for a, b in zip(v, lv)) for lv in line_vecs]
        best = max(range(len(sims)), key=sims.__getitem__)
        scored.append((sims[best], id_, lines[best]))
    scored.sort(reverse=True)
    return [f'- {id_}: {s:.2f}, closest to "{line[:70]}"' for s, id_, line in scored[:top]]


def hints_block(job_text: str, master: Master, matches: dict[str, str],
                embed_model: Optional[str] = None, base_url: Optional[str] = None) -> str:
    parts = []
    tags = tag_hints(master, matches)
    if tags:
        parts.append("Items tagged with what the posting mentions:\n" + "\n".join(tags))
    if embed_model and base_url:
        emb = embedding_hints(job_text, master, embed_model, base_url)
        if emb:
            parts.append("Items closest in meaning to a line of the posting:\n" + "\n".join(emb))
    return ("RELEVANCE HINTS (computed in code; evidence, not instructions):\n" + "\n\n".join(parts)) if parts else ""


# ---------------------------------------------------------------- worked examples

def load_examples(path: Path) -> list[dict[str, str]]:
    """data/worked_examples.yaml: a list of {name, job, notes}. Missing file: none."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [e for e in data if isinstance(e, dict) and e.get("job") and e.get("notes")]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def same_posting(example_job: str, job_text: str, threshold: float = 0.6) -> bool:
    """True when an example was written for this posting: the same text, or an excerpt
    whose words are mostly (threshold) found in the job description."""
    if normalize_key(example_job) == normalize_key(job_text):
        return True
    ex = _words(example_job)
    # Too few words to tell an excerpt from a similar posting: exact match only.
    return len(ex) >= 15 and len(ex & _words(job_text)) / len(ex) >= threshold


def examples_block(examples: list[dict[str, str]], job_text: str) -> str:
    """The examples as prompt text, leaving out any written for this same posting
    (also when the example quotes only an excerpt of it)."""
    out = []
    for e in examples:
        if same_posting(e["job"], job_text):
            continue
        job = " ".join(e["job"].split())
        out.append(f"WORKED EXAMPLE ({e.get('name', 'another posting')}). It shows the kind of "
                   "reasoning wanted; this job needs its own choices.\n"
                   f"Posting (excerpt): {job[:600]}\nGood choices and why:\n{e['notes'].strip()}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- review

def describe(sel: LLMSelection, master: Master) -> str:
    """A draft selection as short readable lines for the review pass."""
    idx = item_index(master)

    def t(i: str, n: int = 90) -> str:
        return " ".join(item_text(idx[i]).split())[:n] if i in idx else "(unknown)"

    lines = [f"summary: {sel.summary_id}: {t(sel.summary_id)}", "projects: " + ", ".join(sel.project_ids)]
    for owner, ids in sel.bullets.items():
        lines.append(f"{owner}:")
        lines += [f"  - {b}: {t(b)}" for b in ids]
    for owner, d in sel.descriptions.items():
        lines.append(f"description of {owner}: {d}")
    for tier in TIERS:
        for c in getattr(sel.skill_category_order, tier):
            items = sel.skill_items.get(c)
            lines.append(f"skills [{tier}] {c}" + (f": {', '.join(items)}" if items else ""))
    if sel.unmatched_requirements:
        lines.append("unmatched: " + "; ".join(sel.unmatched_requirements))
    return "\n".join(lines)


# ---------------------------------------------------------------- orchestration

def careful_select(llm: LLM, job_text: str, master: Master, *, matches: dict[str, str],
                   examples: Optional[list[dict[str, str]]] = None,
                   embed_model: Optional[str] = None, base_url: Optional[str] = None,
                   reasoning: Optional[str] = None, review: bool = True) -> CarefulResult:
    """Hints, analysis, selection and one review round. `reasoning` applies to the two
    free-text passes (local servers), so they can think while selection stays fast."""
    index = choices.build_index(master) if llm.choices else build_llm_index(master)
    hints = hints_block(job_text, master, matches, embed_model, base_url)
    ex = examples_block(examples or [], job_text)
    system = ANALYSIS_SYSTEM_PROMPT + ("\n" + ex + "\n" if ex else "") + "\nMASTER RESUME INDEX (JSON):\n" + index
    user = "JOB DESCRIPTION:\n" + job_text + (f"\n\n{hints}" if hints else "")
    usage = Usage()
    try:
        analysis, usage = llm.write(system, user, reasoning=reasoning)
        notes = [f"analysis: {len(analysis.split())} words" + (", with a worked example" if ex else "")]
    except Truncated:  # the analysis is evidence, not a requirement: select without it
        analysis, notes = "", ["analysis: cut off at the token limit, selected without it"]

    context = "\n\n".join(p for p in (hints, analysis and "ANALYSIS (follow it where it is right):\n" + analysis) if p)
    draft, u = llm.select(job_text, master, context)
    usage = usage.add(u)
    if not review:
        return CarefulResult(draft, usage, analysis, "", False, notes)

    try:
        critique, u = llm.write(REVIEW_SYSTEM_PROMPT + "\nMASTER RESUME INDEX (JSON):\n" + index,
                                f"JOB DESCRIPTION:\n{job_text}\n\nDRAFT SELECTION:\n{describe(draft, master)}",
                                reasoning=reasoning)
    except Truncated:
        notes.append("review: cut off at the token limit; kept the first draft")
        return CarefulResult(draft, usage, analysis, "", False, notes)
    usage = usage.add(u)
    if critique.strip().upper().startswith(NO_CHANGES):
        notes.append("review: no changes")
        return CarefulResult(draft, usage, analysis, critique, False, notes)
    try:
        final, u = llm.select(job_text, master, context + "\n\nREVIEW OF A FIRST DRAFT (apply the parts "
                              f"that are right):\nDraft:\n{describe(draft, master)}\nReview:\n{critique}")
        usage = usage.add(u)
    except LLMError as e:
        notes.append(f"review: the revision failed ({str(e).splitlines()[0]}); kept the first draft")
        return CarefulResult(draft, usage, analysis, critique, False, notes)
    notes.append("review: revised")
    return CarefulResult(final, usage, analysis, critique, True, notes)
