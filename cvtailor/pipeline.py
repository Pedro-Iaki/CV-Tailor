"""The `new` pipeline and the operations on existing applications.

Kept free of CLI code so it can be driven by tests with a fake LLM and renderer.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import db
from .config import Config, app_folder_name, company_key, pdf_filename
from .llm import LLM, LLMError, Usage
from .master import item_index, item_text, load_master
from .render import Renderer, WeasyRenderer, make_render_input
from .route import RouteResult, route
from .schema import TIERS, LLMSelection, Master, Meta, Selection
from .tags import match_tags
from .trim import FitResult, PageFitError, fit_to_page
from .validate import validate_selection
from .variants import LoadedVariant, load_variants, save_variant

URL_RE = re.compile(r"^\s*(https?://\S+)\s*$", re.IGNORECASE)
DUPLICATE_WINDOW_DAYS = 60


class PipelineError(Exception):
    pass


class Skipped(PipelineError):
    """Raised in non-interactive mode when a file should be skipped (e.g. duplicate warning)."""


# ---------------------------------------------------------------- plumbing

@dataclass
class IO:
    out: Callable[[str], None] = print
    confirm: Optional[Callable[[str], bool]] = None
    prompt: Optional[Callable[[str, Optional[str]], str]] = None
    interactive: bool = True


@dataclass
class Context:
    cfg: Config
    conn: Any = None
    llm: Optional[LLM] = None
    renderer: Optional[Renderer] = None
    _base_master: Optional[Master] = None  # master.yaml only
    _master: Optional[Master] = None       # master.yaml + approved facts
    fact_warnings: list[str] = field(default_factory=list)

    @property
    def base_master(self) -> Master:
        if self._base_master is None:
            self._base_master = load_master(self.cfg.master_path)
        return self._base_master

    @property
    def master(self) -> Master:
        """The selectable/renderable content: master.yaml plus approved facts.
        Unsafe items are never part of it."""
        if self._master is None:
            from .facts import load_facts, merge_facts
            self._master, self.fact_warnings = merge_facts(self.base_master, load_facts(self.db()))
        return self._master

    def invalidate_master(self) -> None:
        self._master = None

    def db(self):
        if self.conn is None:
            self.conn = db.connect(self.cfg.db_path)
        return self.conn

    def get_llm(self) -> LLM:
        if self.llm is None:
            if self.cfg.llm_base_url:
                from .local_llm import LocalClient
                client = LocalClient(self.cfg.llm_base_url, self.cfg.llm_api_key, self.cfg.llm_timeout,
                                     self.cfg.llm_reasoning)
                self.llm = LLM(self.cfg.model, None, client=client, count_limits=True, choices=True,
                               max_output=self.cfg.llm_max_tokens)
            else:
                self.llm = LLM(self.cfg.model, self.cfg.effort)
        return self.llm

    def get_renderer(self) -> Renderer:
        if self.renderer is None:
            self.renderer = WeasyRenderer()
        return self.renderer


@dataclass
class NewOptions:
    lang: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    url: Optional[str] = None
    force_generate: bool = False
    variant: Optional[str] = None
    no_trim: bool = False
    allow_duplicate: bool = False
    yes: bool = False
    dry_run: bool = False
    verbose: bool = False
    careful: bool = False  # or CV_LLM_MODE=careful: analysis, selection, one review round


@dataclass
class NewResult:
    app_id: Optional[int]
    pdf_path: Optional[Path]
    company: str
    role: str
    lang: str
    route: str
    variant_name: Optional[str] = None
    trimmed: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def split_url_line(text: str) -> tuple[str, Optional[str]]:
    """If the first non-empty line is a URL, return (rest, url)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        m = URL_RE.match(line)
        if m:
            return "\n".join(lines[i + 1:]).strip(), m.group(1)
        break
    return text.strip(), None


def job_hash(text: str) -> str:
    norm = " ".join(text.split()).lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def display_path(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def open_file(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def select_with_llm(ctx: "Context", text: str, master: Master, matches: dict[str, str],
                    careful: bool, io: IO, verbose: bool = False) -> tuple[LLMSelection, Usage]:
    """The LLM selection step: one call, or careful mode (careful.py) when asked for
    by the option or by CV_LLM_MODE=careful."""
    llm = ctx.get_llm()
    if not (careful or ctx.cfg.careful):
        return llm.select(text, master)
    from .careful import careful_select, load_examples
    res = careful_select(
        llm, text, master, matches=matches,
        examples=load_examples(ctx.cfg.worked_examples_path),
        embed_model=ctx.cfg.embed_model, base_url=ctx.cfg.llm_base_url,
        reasoning=ctx.cfg.careful_reasoning)
    io.out("careful: " + "; ".join(res.notes))
    if verbose:
        io.out("--- analysis ---\n" + res.analysis)
        if res.review:
            io.out("--- review ---\n" + res.review)
    return res.selection, res.usage


def describe_selection(sel: Selection, master: Master, lang: str = "en") -> list[str]:
    idx = item_index(master)
    out = [f"summary: {sel.summary_id}  {item_text(idx[sel.summary_id], lang)}"]
    out.append("bullets:")
    for job_id, ids in sel.bullets.items():
        out.append(f"  {job_id} ({item_text(idx[job_id], lang)})")
        out += [f"    - {b}  {item_text(idx[b], lang)}" for b in ids]
    if sel.project_ids:
        out.append("projects:")
        out += [f"  - {p}  {item_text(idx[p], lang)}" for p in sel.project_ids]
    out.append("skills:")
    for tier in TIERS:
        for c in getattr(sel.skill_category_order, tier):
            if c in sel.skill_items:
                out.append(f"  [{tier}] {c}  {', '.join(sel.skill_items[c])}")
            else:
                out.append(f"  [{tier}] {c}  {item_text(idx[c], lang)}")
    return out


def _fit(ctx: Context, sel: Selection, lang: str, *, trim: bool, io: IO) -> FitResult:
    lines: list[str] = []

    def on_remove(id_: str, text: str) -> None:
        lines.append(f"  trimmed {id_}: {text}")

    try:
        res = fit_to_page(sel, ctx.master, lang, ctx.get_renderer().render, trim=trim, on_remove=on_remove)
    except PageFitError as e:
        raise PipelineError(str(e)) from None
    finally:
        for l in lines:
            io.out(l)
    return res


# ---------------------------------------------------------------- new

def _resolve_meta(ctx: Context, text: str, opts: NewOptions, first_line_url: Optional[str],
                  io: IO, usage: list[Usage]) -> Meta:
    if opts.company and opts.role and opts.lang:
        return Meta(company=opts.company, role_title=opts.role, language=opts.lang,
                    job_url=opts.url or first_line_url)
    meta: Optional[Meta] = None
    try:
        meta, u = ctx.get_llm().extract_meta(text)
        usage.append(u)
    except Exception as e:  # noqa: BLE001 - any API/parse failure falls back to asking
        if not io.interactive or io.prompt is None:
            raise PipelineError(f"meta extraction failed: {e}") from None
        io.out(f"meta extraction failed ({str(e).splitlines()[0]}); please enter the details.")
        company = opts.company or io.prompt("Company", None)
        role = opts.role or io.prompt("Role title", None)
        lang = opts.lang or io.prompt("Language (en/pt)", "en")
        meta = Meta(company=company, role_title=role, language=lang, job_url=None)
    data = meta.model_dump()
    if opts.company:
        data["company"] = opts.company
    if opts.role:
        data["role_title"] = opts.role
    if opts.lang:
        data["language"] = opts.lang
    data["job_url"] = opts.url or first_line_url or data.get("job_url")
    return Meta.model_validate(data)


def _duplicate_check(ctx: Context, jhash: str, ckey: str, opts: NewOptions, io: IO) -> None:
    conn = ctx.db()
    existing = db.find_by_hash(conn, jhash)
    if existing and not opts.allow_duplicate:
        msg = (f"this job description was already used for application #{existing['id']} "
               f"({existing['company']} | {existing['role_title']}, {existing['created_at'][:10]}). "
               "Use --allow-duplicate to create another one.")
        if opts.dry_run:
            io.out("warning: " + msg)
        elif not io.interactive:
            raise Skipped(f"duplicate of #{existing['id']}")
        else:
            raise PipelineError(msg)
    recent = db.find_recent_company(conn, ckey, DUPLICATE_WINDOW_DAYS)
    if recent and not existing:
        r = recent[-1]
        msg = (f"warning: you applied to the same company and role {db.age_days(r['created_at'])} days ago "
               f"(#{r['id']}, {r['company']} | {r['role_title']}).")
        io.out(msg)
        if opts.yes or opts.dry_run:
            return
        if not io.interactive or io.confirm is None:
            raise Skipped(f"same company+role as #{r['id']} within {DUPLICATE_WINDOW_DAYS} days")
        if not io.confirm("Continue anyway?"):
            raise PipelineError("aborted")


def run_new(ctx: Context, raw_text: str, opts: NewOptions, io: IO = IO()) -> NewResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        timings[name] = now - t0
        t0 = now

    text, first_url = split_url_line(raw_text or "")
    if not text.strip():
        raise PipelineError("the job description is empty")
    master = ctx.master
    lap("load")

    # 0. meta
    usages: list[Usage] = []
    meta = _resolve_meta(ctx, text, opts, first_url, io, usages)
    lang = meta.language
    io.out(f"{meta.company} | {meta.role_title} | {lang}")
    lap("meta")

    # 1. tags
    matches = match_tags(text, master)
    job_tags = sorted(matches)
    io.out(f"tags: {', '.join(job_tags) or '(none)'} ({len(job_tags)})")
    if opts.verbose:
        for tag, phrase in matches.items():
            io.out(f"  {tag} <- \"{phrase}\"")
    lap("tags")

    # 2. duplicates
    jhash = job_hash(text)
    ckey = company_key(meta.company, meta.role_title)
    _duplicate_check(ctx, jhash, ckey, opts, io)
    lap("duplicates")

    # 3. route
    variants = load_variants(ctx.cfg.variants_dir, master)
    try:
        rr: RouteResult = route(job_tags, variants, master, threshold=ctx.cfg.route_threshold,
                                force_generate=opts.force_generate, force_variant=opts.variant)
    except ValueError as e:
        raise PipelineError(str(e)) from None
    io.out(rr.message)
    lap("route")

    # 4-5. selection
    llm_sel: Optional[LLMSelection] = None
    variant_name: Optional[str] = None
    if rr.decision == "reused":
        lv: LoadedVariant = rr.chosen.variant
        variant_name = lv.name
        if lv.state == "drifted":
            io.out(f"warning: variant {lv.name} has drifted, changed since approval: {', '.join(lv.changed_ids)}")
        selection = lv.variant.selection
        errors = validate_selection(selection, master)
        if errors:  # cannot happen for eligible variants, but never render unvalidated data
            raise PipelineError("variant selection is invalid:\n  - " + "\n  - ".join(errors))
    else:
        try:
            llm_sel, u = select_with_llm(ctx, text, master, matches, opts.careful, io, opts.verbose)
        except LLMError as e:
            raise PipelineError(str(e)) from None
        usages.append(u)
        selection = llm_sel.core()
        if llm_sel.unmatched_requirements:
            io.out("unmatched: " + ", ".join(llm_sel.unmatched_requirements))
        if llm_sel.suggestions:
            io.out("suggestions: " + " | ".join(llm_sel.suggestions))
    usage = Usage()
    for u in usages:
        usage = usage.add(u)
    lap("select")

    if opts.dry_run:
        io.out("--- selection (dry run, nothing written) ---")
        for line in describe_selection(selection, master, "en"):
            io.out(line)
        if opts.verbose:
            io.out(usage.line())
            io.out("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
        return NewResult(None, None, meta.company, meta.role_title, lang, rr.decision, variant_name, [], usage)

    # 6. render + fit
    fit = _fit(ctx, selection, lang, trim=not opts.no_trim, io=io)
    io.out("trimmed: " + (", ".join(fit.removed) if fit.removed else "none"))
    lap("render")

    # 7. log + write (one transaction; no row and no files on failure)
    fields = {
        "company": meta.company,
        "role_title": meta.role_title,
        "company_key": ckey,
        "job_url": meta.job_url,
        "lang": lang,
        "job_description": text,
        "job_hash": jhash,
        "job_tags": job_tags,
        "route": rr.decision,
        "route_score": rr.score,
        "route_candidates": rr.candidates_json(),
        "variant_name": variant_name,
        "selection": fit.selection.model_dump(),
        "llm_selection": llm_sel.model_dump() if llm_sel else None,
        "trimmed": fit.removed,
        "unmatched_requirements": llm_sel.unmatched_requirements if llm_sel else None,
        "suggestions": llm_sel.suggestions if llm_sel else None,
        "model": ctx.cfg.model if usages else None,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
    }
    app_id, pdf_path = _log_and_write(ctx, fields, fit.rendered.pdf, meta, lang)
    lap("log")
    io.out(f"#{app_id} -> {display_path(pdf_path)}")
    if opts.verbose:
        io.out(usage.line())
        io.out("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in timings.items()))
    return NewResult(app_id, pdf_path, meta.company, meta.role_title, lang, rr.decision,
                     variant_name, fit.removed, usage)


def _log_and_write(ctx: Context, fields: dict, pdf: bytes, meta: Meta, lang: str) -> tuple[int, Path]:
    conn = ctx.db()
    created_at = db.now_iso()
    folder: Optional[Path] = None
    created_folder = False
    conn.execute("BEGIN IMMEDIATE")
    try:
        app_id = db.insert_application(conn, {**fields, "created_at": created_at, "pdf_path": ""})
        folder = ctx.cfg.output_dir / app_folder_name(app_id, created_at[:10], meta.company)
        created_folder = not folder.exists()
        folder.mkdir(parents=True, exist_ok=True)
        pdf_path = folder / pdf_filename(ctx.cfg.pdf_name, name=ctx.master.profile.name,
                                         role=meta.role_title, lang=lang, company=meta.company)
        write_atomic(pdf_path, pdf)
        db.update_application(conn, app_id, {"pdf_path": str(pdf_path)})
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        if folder is not None and created_folder:
            shutil.rmtree(folder, ignore_errors=True)
        raise
    return app_id, pdf_path


# ---------------------------------------------------------------- existing applications

def get_app(ctx: Context, app_id: int) -> dict:
    app = db.get_application(ctx.db(), app_id)
    if app is None:
        raise PipelineError(f"application #{app_id} not found")
    return app


def stored_selection(ctx: Context, app: dict) -> Selection:
    sel = Selection.model_validate(app["selection"])
    errors = validate_selection(sel, ctx.master)
    if errors:
        raise PipelineError(
            f"the stored selection of #{app['id']} is no longer valid against the master:\n  - "
            + "\n  - ".join(errors) + f"\nUse `apply.py edit {app['id']}` or `apply.py regen {app['id']}`."
        )
    return sel


def pdf_path_for(ctx: Context, app: dict, lang: str) -> Path:
    base = Path(app["pdf_path"])
    if lang == app["lang"]:
        return base
    return base.parent / pdf_filename(ctx.cfg.pdf_name, name=ctx.master.profile.name,
                                      role=app["role_title"], lang=lang, company=app["company"])


def rerender(ctx: Context, app_id: int, lang: Optional[str], io: IO = IO()) -> Path:
    app = get_app(ctx, app_id)
    lang = lang or app["lang"]
    sel = stored_selection(ctx, app)
    fit = _fit(ctx, sel, lang, trim=True, io=io)
    if fit.removed:
        io.out("trimmed for this render (stored selection unchanged): " + ", ".join(fit.removed))
    path = pdf_path_for(ctx, app, lang)
    write_atomic(path, fit.rendered.pdf)
    io.out(f"#{app_id} -> {display_path(path)}")
    return path


def regen(ctx: Context, app_id: int, io: IO = IO()) -> Path:
    app = get_app(ctx, app_id)
    try:
        text = app["job_description"]
        llm_sel, usage = select_with_llm(ctx, text, ctx.master, match_tags(text, ctx.master), False, io)
    except LLMError as e:
        raise PipelineError(str(e)) from None
    if llm_sel.unmatched_requirements:
        io.out("unmatched: " + ", ".join(llm_sel.unmatched_requirements))
    if llm_sel.suggestions:
        io.out("suggestions: " + " | ".join(llm_sel.suggestions))
    fit = _fit(ctx, llm_sel.core(), app["lang"], trim=True, io=io)
    io.out("trimmed: " + (", ".join(fit.removed) if fit.removed else "none"))
    path = Path(app["pdf_path"])
    updates = {
        "route": "generated",
        "route_score": None,
        # the rendered selection no longer matches any variant
        "variant_name": None,
        "llm_selection": llm_sel.model_dump(),
        "selection": fit.selection.model_dump(),
        "trimmed": fit.removed,
        "edited": 0,
        "unmatched_requirements": llm_sel.unmatched_requirements,
        "suggestions": llm_sel.suggestions,
        "model": ctx.cfg.model,
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "output_tokens": usage.output_tokens,
    }
    conn = ctx.db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        db.update_application(conn, app_id, updates)
        write_atomic(path, fit.rendered.pdf)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    io.out(f"#{app_id} -> {display_path(path)}")
    return path


def check_edit(ctx: Context, app: dict, sel: Selection) -> tuple[list[str], Optional[FitResult]]:
    """Validate and render an edited selection without trimming."""
    errors = validate_selection(sel, ctx.master)
    if errors:
        return errors, None
    try:
        fit = fit_to_page(sel, ctx.master, app["lang"], ctx.get_renderer().render, trim=False)
    except PageFitError as e:
        return [str(e)], None
    return [], fit


def save_edit(ctx: Context, app: dict, fit: FitResult) -> Path:
    path = Path(app["pdf_path"])
    conn = ctx.db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        db.update_application(conn, app["id"], {"selection": fit.selection.model_dump(), "edited": 1})
        write_atomic(path, fit.rendered.pdf)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return path


def approve(ctx: Context, app_id: int, name: str, replace: bool, io: IO = IO()) -> Path:
    app = get_app(ctx, app_id)
    sel = stored_selection(ctx, app)
    max_pages = ctx.master.limits.max_pages
    problems = []
    for lang in ("en", "pt"):
        r = ctx.get_renderer().render(make_render_input(ctx.master, sel, lang))
        if r.pages > max_pages:
            over = r.pages - max_pages
            problems.append(f"{lang.upper()} is {r.pages} pages ({over} over the limit of {max_pages})")
    if problems:
        raise PipelineError("cannot approve: " + "; ".join(problems)
                            + f". Use `apply.py edit {app_id}` to drop content first.")
    try:
        path = save_variant(ctx.cfg.variants_dir, name=name, selection=sel, job_tags=app["job_tags"],
                            master=ctx.master, source_application_id=app_id, replace=replace)
    except Exception as e:  # VariantError
        raise PipelineError(str(e)) from None
    conn = ctx.db()
    db.update_application(conn, app_id, {"variant_name": name})
    io.out(f"approved #{app_id} as variant '{name}' -> {display_path(path)}")
    return path
