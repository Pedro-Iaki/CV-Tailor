"""MCP server exposing CV Tailor to Claude (Claude Code, Claude Desktop, any MCP client).

Run over stdio:  python -m cvtailor.mcp_server

The same core rule holds here as in the CLI: tools take and return ids; no tool accepts
resume text. stdout belongs to the MCP protocol, so pipeline output is collected and
returned in each result's "log" instead of printed.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import ValidationError

from . import db
from .config import Config
from .facts import FactError, load_facts, load_unsafe
from .generation import (
    apply_approved,
    create_proposals,
    pending_proposals,
    proposal_view,
    review,
)
from .llm import LLMError
from .master import MasterError, build_llm_index, load_master
from .pipeline import (
    IO,
    Context,
    NewOptions,
    PipelineError,
    Skipped,
    approve,
    check_edit,
    describe_selection,
    get_app,
    regen,
    rerender,
    run_new,
    save_edit,
)
from .render import WeasyPrintMissing
from .schema import Selection
from .stats import compute_stats, variant_usage
from .trim import PageFitError
from .variants import VariantError, load_variants

INSTRUCTIONS = """\
CV Tailor builds one-page CVs (English or Brazilian Portuguese) by SELECTING and
ORDERING items from the user's master resume. It never writes or rephrases resume
text, and neither should you: to change a CV, change which ids are selected
(update_selection), or tell the user what to add to their master.yaml.

Typical flow: tailor_cv(job_description) -> report the PDF path, route, trimmed ids,
unmatched requirements and suggestions -> optionally render_application(lang=...) for
the other language, approve_variant(...) for reusable selections, and
set_status(...) as the application progresses.

Optional text generation (only when CV_ALLOW_GENERATION=true): propose_text drafts
wording; show every draft to the user EXACTLY as returned (EN and PT) and pass their
replies VERBATIM to review_proposals. Never answer for the user. Approved rephrasings
become reusable facts; approved new claims are stored as UNSAFE and can only be
promoted by the user in a terminal (`apply.py unsafe promote <id>`); no tool can do it.
"""

KNOWN_ERRORS = (PipelineError, MasterError, LLMError, WeasyPrintMissing, PageFitError,
                VariantError, FactError, ValidationError, ValueError, KeyError)

# Overridable in tests.
make_context: Callable[[], Context] = lambda: Context(Config.from_env())  # noqa: E731


def _collector() -> tuple[IO, list[str]]:
    lines: list[str] = []
    return IO(out=lines.append, interactive=False), lines


def _tool_errors(fn: Callable) -> Callable:
    """Turn expected failures into ToolError so the model sees the message."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except KNOWN_ERRORS as e:
            raise ToolError(str(e)) from e
    return wrapper


def _app_summary(r: dict) -> dict:
    return {
        "id": r["id"],
        "date": r["created_at"][:10],
        "company": r["company"],
        "role": r["role_title"],
        "lang": r["lang"],
        "route": r["route"],
        "variant": r["variant_name"],
        "status": r["status"],
        "age_days": db.age_days(r["created_at"]),
        "pdf_path": r["pdf_path"],
    }


# ---------------------------------------------------------------- tools

@_tool_errors
def tailor_cv(
    job_description: str,
    lang: Optional[str] = None,
    company: Optional[str] = None,
    role: Optional[str] = None,
    job_url: Optional[str] = None,
    force_generate: bool = False,
    variant: Optional[str] = None,
    dry_run: bool = False,
    confirm_same_company: bool = False,
    allow_duplicate: bool = False,
) -> dict:
    """Create a tailored one-page CV PDF for a job description and log the application.

    Pass the full job description text. lang ('en' or 'pt') defaults to the language
    the posting is written in. company/role/lang override detected values (if all three
    are given, no metadata call is made). dry_run returns the selection without writing
    a PDF or a database row. If the user already applied to the same company and role
    in the last 60 days the call fails; ask the user, then retry with
    confirm_same_company=true. An identical posting needs allow_duplicate=true.
    """
    ctx = make_context()
    io, log = _collector()
    opts = NewOptions(lang=lang, company=company, role=role, url=job_url, force_generate=force_generate,
                      variant=variant, dry_run=dry_run, yes=confirm_same_company,
                      allow_duplicate=allow_duplicate)
    try:
        res = run_new(ctx, job_description, opts, io)
    except Skipped as e:
        raise ToolError(
            f"{e}. Ask the user whether to continue; if yes, call again with "
            + ("allow_duplicate=true." if "duplicate" in str(e) else "confirm_same_company=true.")
        ) from e
    except PipelineError as e:
        msg = str(e)
        if "meta extraction failed" in msg:
            msg += " Pass company, role and lang explicitly to skip metadata extraction."
        raise ToolError(msg) from e
    out: dict[str, Any] = {
        "app_id": res.app_id,
        "pdf_path": str(res.pdf_path) if res.pdf_path else None,
        "company": res.company,
        "role": res.role,
        "lang": res.lang,
        "route": res.route,
        "variant": res.variant_name,
        "trimmed": res.trimmed,
        "tokens": vars(res.usage),
        "log": log,
    }
    if res.app_id:
        row = db.get_application(ctx.db(), res.app_id)
        out["job_tags"] = row["job_tags"]
        out["unmatched_requirements"] = row["unmatched_requirements"] or []
        out["suggestions"] = row["suggestions"] or []
    return out


@_tool_errors
def list_applications(status: Optional[str] = None, since: Optional[str] = None, limit: int = 30) -> list[dict]:
    """List logged applications, newest first. status filters by current status;
    since is a YYYY-MM-DD date."""
    ctx = make_context()
    return [_app_summary(r) for r in db.list_applications(ctx.db(), status=status, since=since, limit=limit)]


@_tool_errors
def get_application(app_id: int) -> dict:
    """Full details of one application: the selected items with their English text,
    trimmed ids, unmatched requirements, suggestions, tokens and status history."""
    ctx = make_context()
    r = get_app(ctx, app_id)
    sel = Selection.model_validate(r["selection"])
    try:
        selection_text = describe_selection(sel, ctx.master, "en")
    except KeyError:
        selection_text = ["(selection references ids no longer in the master; use update_selection or regen)"]
    return {
        **_app_summary(r),
        "job_url": r["job_url"],
        "job_tags": r["job_tags"],
        "route_score": r["route_score"],
        "route_candidates": r["route_candidates"],
        "selection": r["selection"],
        "selection_text": selection_text,
        "trimmed": r["trimmed"],
        "edited": bool(r["edited"]),
        "unmatched_requirements": r["unmatched_requirements"],
        "suggestions": r["suggestions"],
        "tokens": {k: r[k] for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")},
        "generated_ids": r.get("generated_ids") or [],
        "pending_proposals": [proposal_view(p, ctx.master) for p in pending_proposals(ctx.db(), app_id)],
        "status_history": db.status_history(ctx.db(), app_id),
    }


@_tool_errors
def set_status(app_ids: list[int], status: str, note: Optional[str] = None) -> dict:
    """Set the status of one or more applications: applied, screen, interview, offer,
    rejected, ghosted or discarded (discarded = generated but never sent)."""
    db.set_status_many(make_context().db(), app_ids, status, note)
    return {"updated": app_ids, "status": status}


@_tool_errors
def get_stats(min_age: int = 14, since: Optional[str] = None) -> dict:
    """Callback rates (screen+ and interview+) overall and by variant, language and
    route. Discarded applications are excluded; 'applied' ones younger than min_age
    days are counted as pending. Groups with n < 10 are flagged low_sample."""
    ctx = make_context()
    s = compute_stats(db.list_applications(ctx.db(), since=since), min_age)

    def g(x) -> dict:
        return {"group": x.key, "n": x.n, "screen_plus": x.screen, "interview_plus": x.interview,
                "screen_rate": round(x.screen_rate, 3), "interview_rate": round(x.interview_rate, 3),
                "low_sample": x.low_sample}

    return {
        "pending": s.pending,
        "discarded_excluded": s.discarded,
        "overall": g(s.overall),
        "by_variant": [g(x) for x in s.by_variant],
        "by_lang": [g(x) for x in s.by_lang],
        "by_route": [g(x) for x in s.by_route],
    }


@_tool_errors
def list_variants(min_age: int = 14) -> list[dict]:
    """Approved variants with state (ok / drifted / stale and why), tag count,
    applications using each and their screen+ rate."""
    ctx = make_context()
    rows = db.list_applications(ctx.db())
    out = []
    for v in load_variants(ctx.cfg.variants_dir, ctx.master):
        u = variant_usage(rows, v.name, min_age)
        out.append({
            "name": v.name,
            "state": v.state,
            "reasons": v.reasons,
            "changed_ids": v.changed_ids,
            "job_tags": v.variant.job_tags if v.variant else [],
            "applications": u.applications,
            "screen_plus": u.screen_plus,
            "n": u.n,
        })
    return out


@_tool_errors
def approve_variant(app_id: int, name: str, replace: bool = False) -> dict:
    """Save an application's selection as a reusable variant. It must fit on one page
    in both English and Portuguese. Use replace=true to overwrite an existing name."""
    ctx = make_context()
    io, log = _collector()
    path = approve(ctx, app_id, name, replace, io)
    return {"variant": name, "path": str(path), "log": log}


@_tool_errors
def render_application(app_id: int, lang: Optional[str] = None) -> dict:
    """Re-render an application's stored selection to PDF without calling the LLM,
    e.g. in the other language ('en' or 'pt') or after fixing a typo in the master."""
    ctx = make_context()
    io, log = _collector()
    path = rerender(ctx, app_id, lang, io)
    return {"pdf_path": str(path), "log": log}


@_tool_errors
def regen_application(app_id: int) -> dict:
    """Re-run the LLM selection for an application (ignores variants) and update its
    PDF and row in place. Use when the first selection was poor."""
    ctx = make_context()
    io, log = _collector()
    path = regen(ctx, app_id, io)
    row = db.get_application(ctx.db(), app_id)
    return {"pdf_path": str(path), "unmatched_requirements": row["unmatched_requirements"],
            "suggestions": row["suggestions"], "log": log}


@_tool_errors
def update_selection(app_id: int, selection: dict) -> dict:
    """Replace an application's selection with a hand-picked one (ids only) and
    re-render without trimming. Shape: {summary_id, bullets: {job, project or education id:
    [bullet ids]}, project_ids: [...], descriptions: {job or project id: description id},
    skill_category_order: {hands_on: [...], working: [...], familiar: [...], languages: [...]}}.
    Every job must be present; items listed in each other's never_with can't go together.
    Use get_master_index for the ids.
    Rejected (and nothing changes) if invalid or longer than one page."""
    ctx = make_context()
    app = get_app(ctx, app_id)
    sel = Selection.model_validate(selection)
    errors, fit = check_edit(ctx, app, sel)
    if errors:
        raise ToolError("selection rejected, nothing changed:\n- " + "\n- ".join(errors))
    path = save_edit(ctx, app, fit)
    return {"pdf_path": str(path), "edited": True}


@_tool_errors
def get_master_index() -> dict:
    """Every selectable id in the master resume with its tags and English text, plus
    jobs, skill categories and limits. Contact details are not included."""
    import json
    ctx = make_context()
    return json.loads(build_llm_index(ctx.master))


@_tool_errors
def validate_master() -> dict:
    """Validate master.yaml (all errors at once) and report each saved variant's state."""
    cfg = Config.from_env()
    try:
        master = load_master(cfg.master_path)
    except MasterError as e:
        return {"ok": False, "errors": e.errors, "variants": []}
    return {
        "ok": True,
        "errors": [],
        "variants": [{"name": v.name, "state": v.state_text()} for v in load_variants(cfg.variants_dir, master)],
    }


@_tool_errors
def propose_text(app_id: int, max_items: int = 5) -> dict:
    """Draft wording improvements for an application's CV (needs CV_ALLOW_GENERATION=true).

    Returns pending proposals. Show each one to the user verbatim (EN and PT, sources,
    rationale, and whether approval makes it a FACT or UNSAFE), then collect the user's
    reply for each: "y", "n", "y, <comment>" (revise and show again) or "n, <comment>".
    Calling this again supersedes proposals still pending for the application."""
    ctx = make_context()
    io, log = _collector()
    props = create_proposals(ctx, app_id, io, max_items=max(1, min(max_items, 5)))
    return {"proposals": [proposal_view(p, ctx.master) for p in props], "log": log}


@_tool_errors
def review_proposals(app_id: int, replies: dict[str, str]) -> dict:
    """Record the USER'S replies to pending proposals, verbatim, keyed by proposal_id.

    Each reply is "y", "n", "y, <comment>" or "n, <comment>". Never invent or infer
    replies. "y, <comment>" returns a revised proposal that must be shown to the user
    and reviewed again. Approved facts are put into this application's CV and the PDF
    is re-rendered; approved new claims are stored as unsafe and are not used."""
    ctx = make_context()
    io, log = _collector()
    res = review(ctx, app_id, {int(k): v for k, v in replies.items()})
    pdf = apply_approved(ctx, app_id, io) if res.facts else None
    return {
        "facts_added": res.facts,
        "unsafe_stored": res.unsafe,
        "rejected": res.rejected,
        "revised": [proposal_view(p, ctx.master) for p in res.revised],
        "still_pending": [p["id"] for p in pending_proposals(ctx.db(), app_id)],
        "pdf_path": str(pdf) if pdf else None,
        "notes": res.notes,
        "log": log,
        "reminder": ("unsafe items are not used in any CV until the user runs "
                     "`apply.py unsafe promote <id>` in a terminal") if res.unsafe else None,
    }


@_tool_errors
def list_facts() -> list[dict]:
    """Approved generated items (reusable; selectable like master items)."""
    ctx = make_context()
    return [{k: f[k] for k in ("id", "kind", "job_id", "origin", "based_on", "en", "pt", "approved_via")}
            for f in load_facts(ctx.db())]


@_tool_errors
def list_unsafe() -> list[dict]:
    """Approved NEW claims awaiting manual promotion by the user (never rendered).
    There is intentionally no tool to promote them."""
    ctx = make_context()
    return [{k: u[k] for k in ("id", "kind", "job_id", "en", "pt", "rationale", "comment")}
            for u in load_unsafe(ctx.db())]


# ---------------------------------------------------------------- server

READ_ONLY = ToolAnnotations(readOnlyHint=True)

TOOLS: list[tuple[Callable, ToolAnnotations | None]] = [
    (tailor_cv, ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)),
    (list_applications, READ_ONLY),
    (get_application, READ_ONLY),
    (set_status, ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)),
    (get_stats, READ_ONLY),
    (list_variants, READ_ONLY),
    (approve_variant, ToolAnnotations(readOnlyHint=False, destructiveHint=False)),
    (render_application, ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)),
    (regen_application, ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)),
    (update_selection, ToolAnnotations(readOnlyHint=False, destructiveHint=True)),
    (get_master_index, READ_ONLY),
    (validate_master, READ_ONLY),
    (propose_text, ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)),
    (review_proposals, ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)),
    (list_facts, READ_ONLY),
    (list_unsafe, READ_ONLY),
]


def build_server() -> MCPServer:
    server = MCPServer("cv-tailor", instructions=INSTRUCTIONS)
    for fn, ann in TOOLS:
        server.add_tool(fn, annotations=ann)
    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
