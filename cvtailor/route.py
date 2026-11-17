"""Score a job's tags against approved variants and decide reuse vs. generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .master import item_index
from .schema import Master, Selection
from .variants import LoadedVariant

MIN_JOB_TAGS = 3


@dataclass
class Candidate:
    name: str
    score: float
    similarity: float
    coverage: float
    variant: LoadedVariant


@dataclass
class RouteResult:
    decision: str  # reused | generated
    chosen: Optional[Candidate] = None
    candidates: list[Candidate] = field(default_factory=list)
    message: str = ""

    @property
    def score(self) -> Optional[float]:
        return self.chosen.score if self.chosen else (self.candidates[0].score if self.candidates else None)

    def candidates_json(self) -> list[dict]:
        return [{"name": c.name, "score": round(c.score, 4)} for c in self.candidates[:3]]


def selection_tags(sel: Selection, master: Master) -> set[str]:
    idx = item_index(master)
    ids = [sel.summary_id, *sel.project_ids]
    for b in sel.bullets.values():
        ids.extend(b)
    tags: set[str] = set()
    for i in ids:
        it = idx.get(i)
        if it is not None:
            tags.update(getattr(it.obj, "tags", []) or [])
    return tags


def score(job_tags: set[str], variant_tags: set[str], content_tags: set[str]) -> tuple[float, float, float]:
    union = job_tags | variant_tags
    similarity = len(job_tags & variant_tags) / len(union) if union else 0.0
    coverage = len(job_tags & content_tags) / len(job_tags) if job_tags else 0.0
    return 0.5 * similarity + 0.5 * coverage, similarity, coverage


def rank(job_tags: list[str], variants: list[LoadedVariant], master: Master) -> list[Candidate]:
    J = set(job_tags)
    out = []
    for lv in variants:
        if not lv.eligible:
            continue
        s, sim, cov = score(J, set(lv.variant.job_tags), selection_tags(lv.variant.selection, master))
        out.append(Candidate(lv.name, s, sim, cov, lv))
    out.sort(key=lambda c: (-c.score, c.name))
    return out


def _fmt(cands: list[Candidate]) -> str:
    return ", ".join(f"{c.name} ({c.score:.2f})" for c in cands)


def route(job_tags: list[str], variants: list[LoadedVariant], master: Master, *,
          threshold: float, force_generate: bool = False,
          force_variant: Optional[str] = None) -> RouteResult:
    cands = rank(job_tags, variants, master)
    top = cands[:3]

    if force_variant:
        lv = next((v for v in variants if v.name == force_variant), None)
        if lv is None:
            raise ValueError(f"variant '{force_variant}' does not exist")
        if not lv.eligible:
            raise ValueError(f"variant '{force_variant}' is not eligible ({lv.state_text()})")
        chosen = next((c for c in cands if c.name == force_variant), None)
        others = [c for c in top if c.name != force_variant]
        msg = f"route: reused {force_variant} (forced, {chosen.score:.2f})"
        if others:
            msg += f" | runner-up: {_fmt(others[:2])}"
        return RouteResult("reused", chosen, top, msg)

    if force_generate:
        return RouteResult("generated", None, top, "route: --force-generate, generating")

    if len(job_tags) < MIN_JOB_TAGS:
        return RouteResult(
            "generated", None, top,
            f"route: only {len(job_tags)} job tag(s) (< {MIN_JOB_TAGS}), skipping routing, generating",
        )

    if not cands:
        return RouteResult("generated", None, [], "route: no eligible variants, generating")

    best = cands[0]
    if best.score >= threshold:
        msg = f"route: reused {best.name} ({best.score:.2f})"
        if len(top) > 1:
            msg += f" | runner-up: {_fmt(top[1:])}"
        return RouteResult("reused", best, top, msg)
    msg = f"route: no match >= {threshold:.2f} (best {best.name} {best.score:.2f}), generating"
    return RouteResult("generated", None, top, msg)
