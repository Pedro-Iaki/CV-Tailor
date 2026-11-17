"""CV Tailor CLI.

    python apply.py new                      # paste a job description, get a PDF
    python apply.py --help                   # all commands
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

import typer
import yaml
from pydantic import ValidationError

from cvtailor import db
from cvtailor.config import Config
from cvtailor.editor import (
    EditAborted,
    open_in_editor,
    parse_annotated,
    strip_error_header,
    to_annotated_yaml,
)
from cvtailor.facts import (
    FactError,
    drop_unsafe,
    get_unsafe,
    load_facts,
    load_unsafe,
    promote,
    remove_fact,
)
from cvtailor.generation import apply_approved, create_proposals, ensure_enabled, format_proposal, review
from cvtailor.llm import LLMError
from cvtailor.master import MasterError
from cvtailor.pipeline import (
    IO,
    Context,
    NewOptions,
    PipelineError,
    Skipped,
    approve as approve_app,
    check_edit,
    display_path,
    get_app,
    open_file,
    regen as regen_app,
    rerender,
    run_new,
    save_edit,
)
from cvtailor.render import PdfEngineMissing
from cvtailor.schema import Selection
from cvtailor.stats import compute_stats, fmt_rate, variant_usage
from cvtailor.trim import PageFitError, fit_to_page
from cvtailor.validate import validate_selection
from cvtailor.variants import load_variants

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Tailor a one-page bilingual CV to a job description by selecting verified content.")

# ValueError covers bad JSON in render-file; ValidationError covers a selection file of the wrong shape.
KNOWN_ERRORS = (PipelineError, MasterError, PdfEngineMissing, LLMError, PageFitError, FactError,
                ValidationError, ValueError)


def _ctx() -> Context:
    return Context(Config.from_env())


def _fail(msg: str, code: int = 1) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _run(fn: Callable[[], None]) -> None:
    try:
        fn()
    except KNOWN_ERRORS as e:
        _fail(str(e))
    except KeyboardInterrupt:
        _fail("interrupted", 130)


def _lang_cb(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.lower()
    if v not in ("en", "pt"):
        raise typer.BadParameter("must be 'en' or 'pt'")
    return v


def _interactive_io() -> IO:
    return IO(
        out=typer.echo,
        confirm=lambda q: typer.confirm(q, default=False),
        prompt=lambda q, d: typer.prompt(q, default=d) if d is not None else typer.prompt(q),
        interactive=True,
    )


def print_table(headers: list[str], rows: list[list], out: Callable[[str], None] = typer.echo) -> None:
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for n, r in enumerate(cells):
        out("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if n == 0:
            out("  ".join("-" * w for w in widths))


# ---------------------------------------------------------------- master / dev

@app.command("validate-master")
def validate_master() -> None:
    """Check master.yaml against every schema rule and report the state of every saved variant."""
    ctx = _ctx()
    cfg = ctx.cfg
    try:
        base = ctx.base_master
        master = ctx.master  # + approved facts
    except MasterError as e:
        typer.secho(f"{cfg.master_path}: {len(e.errors)} problem(s)", fg=typer.colors.RED, err=True)
        for err in e.errors:
            typer.echo(f"  - {err}", err=True)
        raise typer.Exit(1)
    n_bullets = sum(len(e.bullets) for e in [*base.jobs, *base.projects, *base.education])
    typer.echo(f"master ok: {len(base.summaries)} summaries, {len(base.jobs)} jobs, {n_bullets} bullets, "
               f"{len(base.projects)} projects, {len(base.education)} education entries")
    n_facts = len(load_facts(ctx.db()))
    n_unsafe = len(load_unsafe(ctx.db()))
    typer.echo(f"generated text: {n_facts} approved fact(s), {n_unsafe} unsafe item(s) awaiting promotion"
               + ("" if cfg.allow_generation else " (generation disabled)"))
    for w in ctx.fact_warnings:
        typer.secho(f"  warning: {w}", fg=typer.colors.YELLOW)
    variants = load_variants(cfg.variants_dir, master)
    if not variants:
        typer.echo("variants: none")
        return
    typer.echo("variants:")
    bad = False
    for v in variants:
        typer.echo(f"  {v.name}: {v.state_text()}")
        if v.state == "stale":
            bad = True
    if bad:
        typer.echo("  stale variants are not used for routing. Fix with: apply.py regen <id> && "
                   "apply.py approve <id> --name <name> --replace")


@app.command("render-file")
def render_file(
    selection_file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Selection JSON."),
    lang: Optional[str] = typer.Option(None, "--lang", callback=_lang_cb, help="en or pt (default: both)."),
    no_trim: bool = typer.Option(False, "--no-trim"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Output folder (default output/dev)."),
) -> None:
    """[dev] Render a hand-written selection JSON to PDF (EN and PT), with page fit."""
    def go() -> None:
        ctx = _ctx()
        sel = Selection.model_validate(json.loads(selection_file.read_text(encoding="utf-8")))
        errors = validate_selection(sel, ctx.master)
        if errors:
            raise PipelineError("invalid selection:\n  - " + "\n  - ".join(errors))
        target = out_dir or ctx.cfg.output_dir / "dev"
        target.mkdir(parents=True, exist_ok=True)
        renderer = ctx.get_renderer()
        for lg in ([lang] if lang else ["en", "pt"]):
            fit = fit_to_page(sel, ctx.master, lg, renderer.render, trim=not no_trim,
                              on_remove=lambda i, t: typer.echo(f"  trimmed {i}: {t}"))
            path = target / f"{selection_file.stem}_{lg.upper()}.pdf"
            path.write_bytes(fit.rendered.pdf)
            (target / f"{selection_file.stem}_{lg.upper()}.html").write_text(fit.rendered.html, encoding="utf-8")
            typer.echo(f"{lg}: {fit.rendered.pages} page(s), trimmed: {', '.join(fit.removed) or 'none'} "
                       f"-> {display_path(path)}")
    _run(go)


# ---------------------------------------------------------------- new / batch

def _read_job_text(file: Optional[str], clip: bool) -> str:
    if clip:
        try:
            import pyperclip
        except ImportError:
            _fail("--clip needs the optional 'pyperclip' package (pip install pyperclip)")
        try:
            return pyperclip.paste() or ""
        except Exception as e:  # noqa: BLE001
            _fail(f"could not read the clipboard: {e}")
    if file == "-":
        return sys.stdin.read()
    if file:
        p = Path(file)
        if not p.exists():
            _fail(f"file not found: {file}")
        return p.read_text(encoding="utf-8-sig")
    if sys.stdin.isatty():
        eof = "Ctrl-Z then Enter" if sys.platform.startswith("win") else "Ctrl-D"
        typer.echo(f"Paste the job description, then press {eof}:")
    return sys.stdin.read()


@app.command()
def new(
    file: Optional[str] = typer.Argument(None, help="Job description file, or '-' for stdin. Omit to paste."),
    clip: bool = typer.Option(False, "--clip", help="Read the job description from the clipboard."),
    lang: Optional[str] = typer.Option(None, "--lang", callback=_lang_cb, help="CV language (default: the job's)."),
    company: Optional[str] = typer.Option(None, "--company"),
    role: Optional[str] = typer.Option(None, "--role"),
    url: Optional[str] = typer.Option(None, "--url"),
    force_generate: bool = typer.Option(False, "--force-generate", help="Skip routing, always call the LLM."),
    variant: Optional[str] = typer.Option(None, "--variant", help="Reuse this approved variant."),
    no_trim: bool = typer.Option(False, "--no-trim", help="Fail instead of trimming to fit the page."),
    allow_duplicate: bool = typer.Option(False, "--allow-duplicate"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-continue on duplicate warnings."),
    open_pdf: bool = typer.Option(False, "--open", help="Open the PDF when done."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Stop after selection; write nothing."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    careful: bool = typer.Option(False, "--careful",
                                 help="Analyse first, then select, then review once (slower, better)."),
    generate: bool = typer.Option(False, "--generate",
                                  help="Then draft wording improvements for y/n review (needs CV_ALLOW_GENERATION)."),
) -> None:
    """Tailor a CV for one job description (the main command; no flags needed)."""
    ctx = _ctx()
    if generate and not ctx.cfg.allow_generation:
        _fail("--generate needs CV_ALLOW_GENERATION=true in .env")
    text = _read_job_text(file, clip)
    opts = NewOptions(lang=lang, company=company, role=role, url=url, force_generate=force_generate,
                      variant=variant, no_trim=no_trim, allow_duplicate=allow_duplicate, yes=yes,
                      dry_run=dry_run, verbose=verbose, careful=careful)

    def go() -> None:
        res = run_new(ctx, text, opts, _interactive_io())
        if generate and res.app_id:
            _review_loop(ctx, res.app_id)
        if open_pdf and res.pdf_path:
            open_file(res.pdf_path)
    _run(go)


# ---------------------------------------------------------------- opt-in text generation

REPLY_HELP = "y = approve, n = reject, 'y, <comment>' = revise and show again, 'n, <comment>' = reject with feedback"


def _review_loop(ctx: Context, app_id: int) -> None:
    io = _interactive_io()
    typer.echo("")
    typer.echo("drafting wording proposals (nothing is used without your approval)...")
    queue = create_proposals(ctx, app_id, io)
    if not queue:
        typer.echo("no proposals.")
        return
    typer.echo(REPLY_HELP)
    stored_facts, stored_unsafe = [], []
    n = 0
    while queue:
        p = queue.pop(0)
        n += 1
        typer.echo("")
        for line in format_proposal(p, ctx.master, n):
            typer.echo(line)
        while True:
            reply = typer.prompt("    reply")
            try:
                res = review(ctx, app_id, {p["id"]: reply})
                break
            except ValueError as e:
                typer.secho(f"    {e}", fg=typer.colors.YELLOW)
        for note in res.notes:
            typer.secho(f"    {note}", fg=typer.colors.YELLOW)
        if res.notes and not (res.facts or res.unsafe or res.rejected or res.revised):
            queue.insert(0, p)  # failed revision: ask again
            n -= 1
        queue[0:0] = res.revised
        stored_facts += res.facts
        stored_unsafe += res.unsafe
        for f in res.facts:
            typer.echo(f"    saved as fact {f}")
        for u in res.unsafe:
            typer.echo(f"    saved as UNSAFE {u} (not used; promote with `apply.py unsafe promote {u}`)")
    typer.echo("")
    apply_approved(ctx, app_id, io)
    typer.echo(f"review done: {len(stored_facts)} fact(s) added, {len(stored_unsafe)} unsafe item(s) stored.")


@app.command()
def generate(app_id: int = typer.Argument(...)) -> None:
    """Draft wording improvements for an application and review them one by one."""
    def go() -> None:
        ctx = _ctx()
        ensure_enabled(ctx)
        _review_loop(ctx, app_id)
    _run(go)


facts_app = typer.Typer(help="Approved generated text (renderable, reusable).", no_args_is_help=True)
unsafe_app = typer.Typer(help="Approved NEW claims, never rendered until you promote them.", no_args_is_help=True)
app.add_typer(facts_app, name="facts")
app.add_typer(unsafe_app, name="unsafe")


def _item_rows(items: list[dict]) -> list[list]:
    return [[i["id"], i["job_id"] or "summary", i["origin"], i.get("source_application_id") or "",
             i["en"][:70] + ("..." if len(i["en"]) > 70 else "")] for i in items]


@facts_app.command("list")
def facts_list(full: bool = typer.Option(False, "--full", help="Show EN and PT text in full.")) -> None:
    """List approved facts."""
    items = load_facts(_ctx().db())
    if full:
        for i in items:
            typer.echo(f"{i['id']} ({i['origin']}, from {', '.join(i['based_on']) or '-'}, via {i['approved_via']})")
            typer.echo(f"  EN: {i['en']}\n  PT: {i['pt']}")
        return
    print_table(["id", "job", "origin", "app", "en"], _item_rows(items))


@facts_app.command("remove")
def facts_remove(ids: List[str] = typer.Argument(...), yes: bool = typer.Option(False, "--yes", "-y")) -> None:
    """Delete facts. Variants that use them become stale."""
    if not yes and not typer.confirm(f"Remove {len(ids)} fact(s)?", default=False):
        raise typer.Exit(1)
    conn = _ctx().db()
    conn.execute("BEGIN")
    try:
        for i in ids:
            remove_fact(conn, i)
        conn.execute("COMMIT")
    except FactError as e:
        conn.execute("ROLLBACK")
        _fail(str(e))
    typer.echo(f"removed: {', '.join(ids)}")


@unsafe_app.command("list")
def unsafe_list() -> None:
    """List unsafe items (approved new claims awaiting manual promotion)."""
    for i in load_unsafe(_ctx().db()):
        typer.echo(f"{i['id']}  ({i['job_id'] or 'summary'}, from application #{i['source_application_id']})")
        typer.echo(f"  EN: {i['en']}\n  PT: {i['pt']}")
        if i.get("rationale"):
            typer.echo(f"  confirm: {i['rationale']}")
        if i.get("comment"):
            typer.echo(f"  your comments: {i['comment']}")


@unsafe_app.command("promote")
def unsafe_promote(
    ids: List[str] = typer.Argument(...),
    edit: bool = typer.Option(False, "--edit", help="Edit the EN/PT text before promoting."),
) -> None:
    """Move unsafe items to the facts table after you confirm each one is true.

    This is the only way out of the unsafe table, and it is CLI-only on purpose."""
    def go() -> None:
        ctx = _ctx()
        conn = ctx.db()
        for item_id in ids:
            item = get_unsafe(conn, item_id)
            if item is None:
                raise FactError(f"unsafe item '{item_id}' not found")
            en, pt, tags = item["en"], item["pt"], item["tags"]
            if edit:
                edited = open_in_editor(yaml.safe_dump({"en": en, "pt": pt, "tags": tags},
                                                       allow_unicode=True, sort_keys=False, width=1000))
                data = yaml.safe_load(edited) or {}
                if not isinstance(data, dict):
                    raise FactError("the edited file must be a YAML mapping with en, pt and tags")
                en, pt, tags = str(data.get("en", "")), str(data.get("pt", "")), list(data.get("tags") or tags)
            typer.echo(f"{item_id}\n  EN: {en}\n  PT: {pt}")
            if item.get("rationale"):
                typer.echo(f"  confirm: {item['rationale']}")
            if not typer.confirm("Is this accurate and true for you? Promote to facts", default=False):
                typer.echo("  skipped")
                continue
            conn.execute("BEGIN")
            try:
                promote(conn, item_id, en=en, pt=pt, tags=tags)
                ctx.invalidate_master()
                ctx.master  # re-validate master + facts before committing
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                ctx.invalidate_master()
                raise
            typer.echo(f"  promoted {item_id}: it can now be selected for new CVs")
    _run(go)


@unsafe_app.command("drop")
def unsafe_drop(ids: List[str] = typer.Argument(...)) -> None:
    """Delete unsafe items."""
    conn = _ctx().db()
    conn.execute("BEGIN")
    try:
        for i in ids:
            drop_unsafe(conn, i)
        conn.execute("COMMIT")
    except FactError as e:
        conn.execute("ROLLBACK")
        _fail(str(e))
    typer.echo(f"dropped: {', '.join(ids)}")


@app.command()
def batch(
    files: List[Path] = typer.Argument(..., help="Job description files."),
    lang: Optional[str] = typer.Option(None, "--lang", callback=_lang_cb),
    force_generate: bool = typer.Option(False, "--force-generate"),
    no_trim: bool = typer.Option(False, "--no-trim"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run `new` on several files non-interactively and print a summary table."""
    try:
        ctx = _ctx()
        ctx.master  # fail fast on a broken master
    except MasterError as e:
        _fail(str(e))
    rows = []
    io = IO(out=lambda s: typer.echo("  " + s), interactive=False)
    for f in files:
        typer.echo(f"== {f}")
        opts = NewOptions(lang=lang, force_generate=force_generate, no_trim=no_trim, verbose=verbose)
        try:
            if not f.exists():
                raise PipelineError("file not found")
            res = run_new(ctx, f.read_text(encoding="utf-8-sig"), opts, io)
            rows.append([f.name, res.app_id, res.company, res.role, res.route, "ok"])
        except Skipped as e:
            rows.append([f.name, "", "", "", "", f"skipped: {e}"])
        except Exception as e:  # noqa: BLE001 - continue on any error
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            rows.append([f.name, "", "", "", "", f"failed: {reason}"])
        typer.echo("")
    print_table(["file", "id", "company", "role", "route", "status"], rows)


# ---------------------------------------------------------------- existing applications

@app.command()
def render(
    app_id: int = typer.Argument(...),
    lang: Optional[str] = typer.Option(None, "--lang", callback=_lang_cb),
) -> None:
    """Re-render an application's stored selection (no LLM call, no new row)."""
    _run(lambda: rerender(_ctx(), app_id, lang, _interactive_io()))


@app.command()
def edit(app_id: int = typer.Argument(...)) -> None:
    """Edit an application's selection in $EDITOR (annotated YAML)."""
    def go() -> None:
        ctx = _ctx()
        appl = get_app(ctx, app_id)
        sel = Selection.model_validate(appl["selection"])
        text = to_annotated_yaml(sel, ctx.master)
        while True:
            edited = open_in_editor(text)
            try:
                new_sel = parse_annotated(edited)
            except EditAborted:
                typer.echo("empty file: edit aborted, nothing changed.")
                return
            except ValueError as e:
                errors = [str(e)]
            else:
                errors, fit = check_edit(ctx, appl, new_sel)
                if not errors:
                    path = save_edit(ctx, appl, fit)
                    typer.echo(f"#{app_id} updated -> {display_path(path)}")
                    return
            typer.secho("edit rejected:", fg=typer.colors.YELLOW)
            for e in errors:
                typer.echo(f"  - {e}")
            text = "\n".join(f"# !! {l}" for l in
                             ["The edit was rejected. Fix these problems and save again:",
                              *[f"  - {x}" for e in errors for x in e.splitlines()]]) \
                + "\n\n" + strip_error_header(edited)
    _run(go)


@app.command()
def regen(app_id: int = typer.Argument(...)) -> None:
    """Re-run selection (always generates) and update the application in place."""
    _run(lambda: regen_app(_ctx(), app_id, _interactive_io()))


@app.command("open")
def open_cmd(app_id: int = typer.Argument(...)) -> None:
    """Open an application's PDF."""
    def go() -> None:
        appl = get_app(_ctx(), app_id)
        p = Path(appl["pdf_path"])
        if not p.exists():
            raise PipelineError(f"PDF not found: {p} (re-create it with `apply.py render {app_id}`)")
        open_file(p)
    _run(go)


@app.command()
def approve(
    app_id: int = typer.Argument(...),
    name: str = typer.Option(..., "--name", help="Variant name, e.g. platform-v1."),
    replace: bool = typer.Option(False, "--replace"),
) -> None:
    """Save an application's selection as a reusable variant (must fit in EN and PT)."""
    _run(lambda: approve_app(_ctx(), app_id, name, replace, _interactive_io()))


@app.command()
def variants(min_age: int = typer.Option(14, "--min-age")) -> None:
    """List approved variants with state and callback rates."""
    def go() -> None:
        ctx = _ctx()
        vs = load_variants(ctx.cfg.variants_dir, ctx.master)
        if not vs:
            typer.echo("no variants yet (create one with `apply.py approve <id> --name NAME`)")
            return
        rows_all = db.list_applications(ctx.db())
        rows = []
        for v in vs:
            u = variant_usage(rows_all, v.name, min_age)
            tags = len(v.variant.job_tags) if v.variant else "-"
            rows.append([v.name, v.state_text(), tags, u.applications, fmt_rate(u.screen_plus, u.n)])
        print_table(["name", "state", "tags", "apps", "screen+"], rows)
    _run(go)


@app.command()
def status(
    args: List[str] = typer.Argument(..., metavar="ID... STATUS"),
    note: Optional[str] = typer.Option(None, "--note"),
) -> None:
    """Set the status of one or more applications."""
    if len(args) < 2:
        _fail("usage: apply.py status <id>... <status>")
    new_status = args[-1].lower()
    if new_status not in db.STATUSES:
        _fail(f"unknown status '{new_status}' (use one of: {', '.join(db.STATUSES)})")
    try:
        ids = [int(a) for a in args[:-1]]
    except ValueError:
        _fail("ids must be integers")
    _set_status(ids, new_status, note)


def _set_status(ids: list[int], new_status: str, note: Optional[str]) -> None:
    try:
        db.set_status_many(_ctx().db(), ids, new_status, note)
    except ValueError as e:
        _fail(str(e))
    typer.echo(f"{', '.join(f'#{i}' for i in ids)} -> {new_status}")


@app.command()
def discard(ids: List[int] = typer.Argument(...), note: Optional[str] = typer.Option(None, "--note")) -> None:
    """Mark applications as discarded (generated but never sent)."""
    _set_status(ids, "discarded", note)


@app.command("mark-ghosted")
def mark_ghosted(
    older_than: int = typer.Option(..., "--older-than", help="Days since applying."),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Set 'ghosted' on every application still 'applied' and older than N days."""
    conn = _ctx().db()
    rows = [r for r in db.list_applications(conn, status="applied") if db.age_days(r["created_at"]) > older_than]
    if not rows:
        typer.echo(f"nothing to mark: no 'applied' applications older than {older_than} days")
        return
    print_table(["id", "date", "company", "role", "age"],
                [[r["id"], r["created_at"][:10], r["company"], r["role_title"], db.age_days(r["created_at"])]
                 for r in rows])
    if not yes and not typer.confirm(f"Mark these {len(rows)} as ghosted?", default=False):
        typer.echo("aborted")
        raise typer.Exit(1)
    _set_status([r["id"] for r in rows], "ghosted", f"no reply after {older_than} days")


@app.command("list")
def list_cmd(
    status_f: Optional[str] = typer.Option(None, "--status"),
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD"),
    limit: int = typer.Option(30, "--limit"),
) -> None:
    """List applications, newest first."""
    conn = _ctx().db()
    rows = db.list_applications(conn, status=status_f, since=since, limit=limit)
    print_table(
        ["id", "date", "company", "role", "lang", "route", "variant", "status", "age"],
        [[r["id"], r["created_at"][:10], r["company"][:28], r["role_title"][:36], r["lang"], r["route"],
          r["variant_name"] or "", r["status"], db.age_days(r["created_at"])] for r in rows],
    )


@app.command()
def stats(
    min_age: int = typer.Option(14, "--min-age", help="'applied' rows younger than this are pending."),
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD"),
) -> None:
    """Callback rates by variant, language and route."""
    conn = _ctx().db()
    s = compute_stats(db.list_applications(conn, since=since), min_age)
    typer.echo(f"counted: {s.overall.n} | pending (applied < {min_age} days): {s.pending} | "
               f"discarded (excluded): {s.discarded}")

    def table(title: str, groups) -> None:
        typer.echo("")
        typer.echo(title)
        print_table(["group", "n", "screen+", "interview+", ""],
                    [[g.key, g.n, fmt_rate(g.screen, g.n), fmt_rate(g.interview, g.n),
                      "low sample" if g.low_sample else ""] for g in groups])

    table("overall", [s.overall])
    table("by variant", s.by_variant)
    table("by language", s.by_lang)
    table("by route", s.by_route)


@app.command()
def rate(
    candidates: Optional[List[Path]] = typer.Option(
        None, "--candidates", "-c", help="JSONL file of extra CVs to rate (repeatable), e.g. benchmark output."),
    apps: bool = typer.Option(True, "--apps/--no-apps", help="Include saved applications."),
    port: int = typer.Option(8765, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open the page automatically."),
) -> None:
    """Rate CVs 1-5 on a local page (blind: the setup shows after rating). Ctrl+C to stop."""
    import webbrowser

    from cvtailor.ratings import RatingApp, RatingStore, application_candidates, file_candidates, serve
    ctx = _ctx()
    cands = application_candidates(ctx.db(), ctx.master) if apps else []
    for path in candidates or []:
        if not path.exists():
            _fail(f"no such file: {path}")
        cands += file_candidates(path, ctx.master)
    if not cands:
        _fail("nothing to rate: no applications, and no --candidates file given")
    server = serve(RatingApp(cands, RatingStore(ctx.cfg.db_path)), port)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"{len(cands)} CV(s) to rate at {url} (local only). Ctrl+C to stop.")
    if not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("stopped; ratings are saved in applications.db (table cv_ratings)")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()
