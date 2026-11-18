"""The desktop app: a pywebview window (WebView2 on Windows) over the same pipeline.

Run from source with `python -m cvtailor.desktop`, or as the packaged portable app
(CV Tailor.exe). The page (cvtailor/ui/index.html) calls the methods of `Api` through
window.pywebview.api. Long operations (tailoring, rendering, model downloads) run in
background threads as jobs; the page polls job_status for progress.

Every call builds its own Context, because the page calls in from several threads and
SQLite connections are per-thread.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from . import db, langs, ollama_setup
from .config import Config, app_home, default_dirs, is_portable, settings_path
from .master import MasterError
from .pipeline import (IO, Context, NewOptions, PipelineError, get_app, pdf_path_for, regen,
                       rerender, run_new)
from .render import MissingTextError, PdfEngineMissing, find_browser, make_render_input, render_html

UI_DIR = Path(__file__).resolve().parent / "ui"

# Settings: key -> (label, kind, help, level). kind: text | secret | int | choice:a,b,c |
# bool. level: "simple" is always shown; "advanced" (model names and exact tuning) sits
# in a collapsed section. The simple Quality and Engine choices below set several
# advanced keys at once. Anything else in settings.env is kept as is.
SETTINGS: dict[str, tuple[str, str, str, str]] = {
    "ANTHROPIC_API_KEY": ("Anthropic API key", "secret", "Only for the Anthropic API engine.", "simple"),
    "CV_PDF_NAME": ("PDF file name", "text", "{name}, {role}, {company}, {LANG}, {lang}", "simple"),
    "CV_ALLOW_GENERATION": ("Allow drafted wording", "bool", "Off by default; drafts need your approval one by one.", "simple"),
    "CV_LLM_BASE_URL": ("Local model server", "text", "Ollama: http://localhost:11434/v1. Empty: use the Anthropic API.", "advanced"),
    "CV_MODEL": ("Model", "text", "Name on the server, e.g. qwen3.5:35b-a3b.", "advanced"),
    "CV_LLM_REASONING": ("Reasoning (selection)", "choice:,none,low,medium,high",
                         "Qwen: none (fast). Empty: the model's default.", "advanced"),
    "CV_LLM_MODE": ("Default mode", "choice:,careful", "careful: analyse, select, review (slower, better).", "advanced"),
    "CV_LLM_CAREFUL_REASONING": ("Reasoning (careful analysis)", "choice:,none,low,medium,high,default",
                                 "Empty: same as selection. default: the model's own (Qwen thinks).", "advanced"),
    "CV_LLM_EMBED_MODEL": ("Embedding model (hints)", "text", "Optional, e.g. bge-m3.", "advanced"),
    "CV_LLM_MAX_TOKENS": ("Max output tokens", "int", "Thinking included; at most the context window minus about 4.5k.", "advanced"),
    "CV_LLM_TIMEOUT": ("Timeout per call (s)", "int", "", "advanced"),
    "CV_PDF_ENGINE": ("PDF engine", "choice:auto,browser,weasyprint", "auto: Edge/Chrome if found.", "advanced"),
}

LOCAL_URL = "http://localhost:11434/v1"

# The simple layer: each option sets these advanced keys ("" removes the key).
PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "quality": {
        "fast": {"CV_LLM_MODE": "", "CV_MODEL": ollama_setup.MAIN_MODEL, "CV_LLM_REASONING": "none",
                 "CV_LLM_CAREFUL_REASONING": "", "CV_LLM_MAX_TOKENS": "4096"},
        "careful": {"CV_LLM_MODE": "careful", "CV_MODEL": ollama_setup.MAIN_MODEL, "CV_LLM_REASONING": "none",
                    "CV_LLM_CAREFUL_REASONING": "none", "CV_LLM_MAX_TOKENS": "4096",
                    "CV_LLM_EMBED_MODEL": "bge-m3"},
        "thorough": {"CV_LLM_MODE": "careful", "CV_MODEL": ollama_setup.LONG_CONTEXT_MODEL,
                     "CV_LLM_REASONING": "none", "CV_LLM_CAREFUL_REASONING": "default",
                     "CV_LLM_MAX_TOKENS": "10000", "CV_LLM_EMBED_MODEL": "bge-m3"},
    },
    "engine": {
        "local": {"CV_LLM_BASE_URL": LOCAL_URL},
        "anthropic": {"CV_LLM_BASE_URL": ""},
    },
}
PRESET_INFO = {
    "fast": "Fast: one selection call. About 1-3 minutes per CV.",
    "careful": "Careful: analyse the posting, select, then review once. About 6-10 minutes.",
    "thorough": "Thorough: careful mode with the model thinking in its analysis and review. "
                "About 15 minutes; rated best. Needs the 16k-context model (Setup tab, one click).",
    "local": "On this PC with Ollama (private, free).",
    "anthropic": "Anthropic API (needs an API key; the job description and master index are sent to it).",
}

FIRST_RUN_SETTINGS = {"CV_LLM_BASE_URL": LOCAL_URL, **PRESETS["quality"]["fast"], "CV_PDF_ENGINE": "auto"}
FIRST_RUN_SETTINGS = {k: v for k, v in FIRST_RUN_SETTINGS.items() if v}

IMPORTABLE = ["master.yaml", "worked_examples.yaml", "applications.db", "variants"]


def current_quality(values: dict) -> str:
    """The quality preset the settings match ("custom" if none)."""
    for name, keys in PRESETS["quality"].items():
        if all((values.get(k) or "") == v for k, v in keys.items()):
            return name
    return "custom"


def with_quality(cfg: Config, quality: Optional[str]) -> Config:
    """cfg with one run's quality preset applied (local engine only), as if the
    preset's settings were in settings.env."""
    keys = PRESETS["quality"].get(quality or "")
    if not keys or not cfg.llm_base_url:
        return cfg
    import dataclasses
    from .config import _careful_reasoning
    fields = {"CV_LLM_MODE": ("careful", lambda v: v == "careful"),
              "CV_MODEL": ("model", lambda v: v or cfg.model),
              "CV_LLM_REASONING": ("llm_reasoning", lambda v: v or None),
              "CV_LLM_CAREFUL_REASONING": ("careful_reasoning", lambda v: _careful_reasoning(v) if v else None),
              "CV_LLM_MAX_TOKENS": ("llm_max_tokens", lambda v: int(v) if v else cfg.llm_max_tokens),
              "CV_LLM_EMBED_MODEL": ("embed_model", lambda v: v or None)}
    return dataclasses.replace(cfg, **{fields[k][0]: fields[k][1](v) for k, v in keys.items() if k in fields})


def _message_page(text: str) -> str:
    """A small page in place of a CV preview that can't be shown."""
    import html
    return ("<body style=\"margin:0;height:100vh;display:grid;place-items:center;font:14px 'Segoe UI',"
            "sans-serif;color:#6d6c66;background:#fff;text-align:center;padding:0 32px\">"
            f"<div>{html.escape(text)}</div></body>")


def _langs_view(codes: list[str]) -> list[dict]:
    return [{"code": c, "name": langs.name(c)} for c in codes]


def _settings_values() -> dict:
    try:
        from dotenv import dotenv_values
        return dict(dotenv_values(settings_path())) if settings_path().exists() else {}
    except ImportError:  # pragma: no cover
        return {}


def ensure_portable_layout() -> None:
    """Create Data/, Output/ and a first settings.env in the portable folder."""
    data, output, settings = default_dirs()
    data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not settings.exists():
        settings.write_text("# CV Tailor settings (edit here or on the Settings tab)\n" + "".join(
            f"{k}={v}\n" for k, v in FIRST_RUN_SETTINGS.items()), encoding="utf-8")


def reload_settings() -> None:
    """Re-read settings.env into the environment. For the settings the app manages,
    the file is the source of truth: a setting removed from it is removed here too."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover
        return
    values = dotenv_values(settings_path()) if settings_path().exists() else {}
    for k in set(values) | set(SETTINGS):
        v = values.get(k)
        if v is None or v == "":
            if k in SETTINGS or k in values:
                os.environ.pop(k, None)
        else:
            os.environ[k] = v


class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.lines: list[str] = []
        self.done = False
        self.result: Any = None
        self.error: Optional[str] = None
        self.progress: Optional[dict] = None

    def view(self) -> dict:
        return {"id": self.id, "done": self.done, "lines": self.lines[-200:], "result": self.result,
                "error": self.error, "progress": self.progress}


class Api:
    def __init__(self, ctx_factory: Optional[Callable[[], Context]] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._rate_problems: list[str] = []
        # Private on purpose: pywebview exposes every public attribute of js_api to the
        # page and walks it on the UI thread; walking the Window object hangs the app.
        self._window: Any = None  # set by main(); used for file dialogs
        self._factory = ctx_factory or (lambda: Context(Config.from_env()))

    # ------------------------------------------------------------ helpers

    def _ctx(self) -> Context:
        """A fresh Context per call or job (SQLite connections are per-thread)."""
        return self._factory()

    def _run(self, fn: Callable[[Job], Any]) -> str:
        job = Job()
        self._jobs[job.id] = job

        def go() -> None:
            try:
                job.result = fn(job)
            except (PipelineError, MasterError, PdfEngineMissing, MissingTextError) as e:
                job.error = str(e)
            except Exception as e:  # noqa: BLE001 - shown to the user, not swallowed
                job.error = f"{type(e).__name__}: {e}"
                job.lines.append(traceback.format_exc())
            finally:
                job.done = True
        threading.Thread(target=go, daemon=True).start()
        return job.id

    def job_status(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        return job.view() if job else {"id": job_id, "done": True, "error": "unknown job"}

    # ------------------------------------------------------------ overview

    def get_state(self) -> dict:
        ctx = self._ctx()
        cfg = ctx.cfg
        settings = settings_path()
        master_ok, master_msg, languages = True, "", None
        try:
            m = ctx.master
            master_msg = f"{len(m.summaries)} summaries, {len(m.jobs)} jobs, {len(m.projects)} projects"
            languages = {"primary": m.primary, "secondary": m.languages.secondary, "list": _langs_view(m.langs)}
        except MasterError as e:
            master_ok, master_msg = False, "\n".join(e.errors) if getattr(e, "errors", None) else str(e)
        except FileNotFoundError:
            master_ok, master_msg = False, f"no master.yaml in {cfg.data_dir}"
        return {
            "portable": is_portable(), "home": str(app_home()), "data_dir": str(cfg.data_dir),
            "output_dir": str(cfg.output_dir), "settings_file": str(settings),
            "local": bool(cfg.llm_base_url), "model": cfg.model, "careful_default": cfg.careful,
            "pdf_engine": cfg.pdf_engine, "browser": find_browser(),
            "master_ok": master_ok, "master": master_msg, "master_missing": not cfg.master_path.exists(),
            "languages": languages, "language_choices": langs.choices(),
            "ollama": ollama_setup.status(cfg.llm_base_url) if cfg.llm_base_url else None,
            "recommended": [{"name": n, "size": s, "why": w} for n, s, w in ollama_setup.RECOMMENDED],
            "main_model": ollama_setup.MAIN_MODEL, "long_context_model": ollama_setup.LONG_CONTEXT_MODEL,
            "quality": current_quality(_settings_values()) if cfg.llm_base_url else None,
            "preset_info": PRESET_INFO,
        }

    # ------------------------------------------------------------ new CV

    def new_cv(self, text: str, options: dict) -> str:
        def work(job: Job) -> dict:
            ctx = self._ctx()
            ctx.cfg = with_quality(ctx.cfg, options.get("quality"))  # this run only
            if options.get("quality") and ctx.cfg.llm_base_url:
                job.lines.append(f"quality: {options['quality']} ({ctx.cfg.model})")
            io = IO(out=job.lines.append, confirm=lambda q: bool(options.get("allow_duplicate")),
                    prompt=lambda q, d: d or "", interactive=False)
            opts = NewOptions(lang=options.get("lang") or None, force_generate=bool(options.get("force_generate")),
                              allow_duplicate=bool(options.get("allow_duplicate")),
                              yes=bool(options.get("allow_duplicate")), dry_run=bool(options.get("dry_run")),
                              careful=bool(options.get("careful")), verbose=True)
            res = run_new(ctx, text, opts, io)
            out = {"app_id": res.app_id, "pdf": str(res.pdf_path) if res.pdf_path else None,
                   "company": res.company, "role": res.role, "lang": res.lang, "route": res.route,
                   "trimmed": res.trimmed}
            if res.app_id:
                app = get_app(ctx, res.app_id)
                out["unmatched"] = app.get("unmatched_requirements") or []
                out["suggestions"] = app.get("suggestions") or []
            return out
        return self._run(work)

    def read_clipboard(self) -> str:
        return read_clipboard_text()

    # ------------------------------------------------------------ applications

    def list_apps(self) -> list[dict]:
        ctx = self._ctx()
        rows = db.list_applications(ctx.db())
        return [{"id": r["id"], "date": r["created_at"][:10], "company": r["company"], "role": r["role_title"],
                 "lang": r["lang"], "route": r["route"], "status": r["status"],
                 "age": db.age_days(r["created_at"])} for r in rows]

    def set_status(self, app_id: int, status: str) -> dict:
        ctx = self._ctx()
        db.set_status_many(ctx.db(), [int(app_id)], status)
        return {"ok": True}

    def preview(self, app_id: int, lang: str) -> str:
        ctx = self._ctx()
        from .pipeline import stored_selection
        app = get_app(ctx, int(app_id))
        try:
            return render_html(make_render_input(ctx.master, stored_selection(ctx, app), lang or app["lang"]))
        except (MissingTextError, PipelineError) as e:
            return _message_page(str(e))

    def open_pdf(self, app_id: int, lang: str) -> str:
        """Open the PDF in the default viewer, rendering it first if it doesn't exist."""
        def work(job: Job) -> str:
            ctx = self._ctx()
            app = get_app(ctx, int(app_id))
            path = pdf_path_for(ctx, app, lang or app["lang"])
            if not path.exists():
                path = rerender(ctx, int(app_id), lang, IO(out=job.lines.append))
            _open(path)
            return str(path)
        return self._run(work)

    def regen_app(self, app_id: int) -> str:
        return self._run(lambda job: str(regen(self._ctx(), int(app_id), IO(out=job.lines.append))))

    def open_folder(self, which: str) -> None:
        cfg = self._ctx().cfg
        path = {"data": cfg.data_dir, "output": cfg.output_dir}.get(which, app_home())
        Path(path).mkdir(parents=True, exist_ok=True)
        _open(Path(path))

    def open_master(self) -> None:
        _open(self._ctx().cfg.master_path)

    # ------------------------------------------------------------ rating

    def _rating_app(self):
        from .ratings import RatingApp, RatingStore, application_candidates, file_candidates
        ctx = self._ctx()
        cands = application_candidates(ctx.db(), ctx.master)
        cand_dir = ctx.cfg.data_dir / "candidates"
        self._rate_problems = []
        for f in sorted(cand_dir.glob("*.jsonl")) if cand_dir.exists() else []:
            try:  # one broken file must not hide everything else
                cands += file_candidates(f, ctx.master)
            except Exception as e:  # noqa: BLE001 - reported on the Rate tab
                self._rate_problems.append(f"{f.name}: {e}")
        return RatingApp(cands, RatingStore(ctx.cfg.db_path))

    def rate_items(self) -> list[dict]:
        return self._rating_app().items()

    def rate_info(self) -> dict:
        """Where rateable CVs come from, and files that could not be read."""
        ctx = self._ctx()
        items = self._rating_app().items()
        return {"count": len(items), "candidates_dir": str(ctx.cfg.data_dir / "candidates"),
                "problems": list(self._rate_problems)}

    def rate_cv(self, key: str, lang: str) -> str:
        return self._rating_app().cv_html(key, lang)

    def rate_job(self, key: str) -> str:
        return self._rating_app().job_text(key)

    def rate(self, key: str, rating: int, note: str = "") -> dict:
        return self._rating_app().rate(key, int(rating), note or "")

    def rate_summary(self) -> list[dict]:
        return self._rating_app().summary()

    # ------------------------------------------------------------ builder
    # The Master tab: the master as an editable list (edits go through master_edit,
    # validated before each write, with undo) and a manual CV: tick items, see the
    # preview, check the page fit, save a PDF or the selection of an application.

    def _editor(self):
        from .master_edit import MasterEditor
        cfg = self._ctx().cfg
        return MasterEditor(cfg.master_path, cfg.master_history_dir)

    def builder_state(self) -> dict:
        from .master import alt_conflicts
        ctx = self._ctx()
        try:
            m = ctx.master
        except (MasterError, FileNotFoundError) as e:
            return {"error": "\n".join(getattr(e, "errors", None) or [str(e)])}
        from .master import missing_texts
        return {"master": m.model_dump(mode="json"),
                "conflicts": {k: sorted(v) for k, v in alt_conflicts(m).items()},
                "apps": self.list_apps(), "undo": self._editor().history_count(),
                "allow_generation": ctx.cfg.allow_generation, "langs": _langs_view(m.langs),
                # per language other than the primary: ids of items (or entries) with no text in it
                "missing": {c: sorted({i for i, _ in missing_texts(m, c)}) for c in m.langs[1:]}}

    def _selection(self, data: dict):
        from .schema import Selection
        return Selection.model_validate(data)

    def builder_preview(self, selection: dict, lang: str) -> dict:
        from .validate import validate_selection
        ctx = self._ctx()
        try:
            sel = self._selection(selection)
        except Exception as e:  # noqa: BLE001 - a malformed selection, shown to the user
            return {"html": "", "errors": [str(e)]}
        errors = validate_selection(sel, ctx.master)
        try:
            html = render_html(make_render_input(ctx.master, sel, lang))
        except Exception as e:  # noqa: BLE001 - e.g. an id the master no longer has
            html, errors = "", [*errors, str(e) if isinstance(e, MissingTextError) else f"preview failed: {e}"]
        return {"html": html, "errors": errors}

    def set_languages(self, primary: str, secondary: Optional[str]) -> dict:
        """Change the CV's languages in master.yaml (Settings tab)."""
        from .master_edit import EditError
        try:
            m = self._editor().set_languages(primary, secondary or None)
        except (EditError, MasterError) as e:
            return {"ok": False, "errors": e.errors}
        from .master import missing_texts
        missing = len(missing_texts(m, m.languages.secondary)) if m.languages.secondary else 0
        return {"ok": True, "missing": missing, **self.get_state()}

    def create_master(self, data: dict) -> dict:
        """The setup wizard: write a first master.yaml from the user's answers."""
        from .master_edit import EditError, create_master
        try:
            create_master(self._ctx().cfg.master_path, data)
        except EditError as e:
            return {"ok": False, "errors": e.errors}
        return {"ok": True, **self.get_state()}

    def builder_fit(self, selection: dict, lang: str) -> str:
        """Render the real PDF to count its pages (a job: Edge takes a second or two)."""
        def work(job: Job) -> dict:
            ctx = self._ctx()
            r = ctx.get_renderer().render(make_render_input(ctx.master, self._selection(selection), lang))
            return {"pages": r.pages, "max": ctx.master.limits.max_pages}
        return self._run(work)

    def builder_save_pdf(self, selection: dict, lang: str, label: str = "") -> str:
        """Write the manual CV as a PDF in Output/manual/ and open it."""
        from .config import pdf_filename
        from .validate import validate_selection

        def work(job: Job) -> str:
            ctx = self._ctx()
            sel = self._selection(selection)
            errors = validate_selection(sel, ctx.master)
            if errors:
                raise PipelineError("fix these first:\n  - " + "\n  - ".join(errors))
            r = ctx.get_renderer().render(make_render_input(ctx.master, sel, lang))
            name = pdf_filename(ctx.cfg.pdf_name, name=ctx.master.profile.name, role=label or "manual",
                                lang=lang, company="")
            path = ctx.cfg.output_dir / "manual" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.pdf)
            _open(path)
            return str(path)
        return self._run(work)

    def builder_from_app(self, app_id: int) -> dict:
        app = get_app(self._ctx(), int(app_id))
        return {"id": app["id"], "company": app["company"], "role": app["role_title"], "lang": app["lang"],
                "selection": app["selection"]}

    def builder_save_to_app(self, app_id: int, selection: dict) -> str:
        """Store the selection in the application and re-render its PDF (no trimming:
        an over-long CV is reported instead)."""
        from .pipeline import check_edit, save_edit

        def work(job: Job) -> str:
            ctx = self._ctx()
            app = get_app(ctx, int(app_id))
            errors, fit = check_edit(ctx, app, self._selection(selection))
            if errors:
                raise PipelineError("not saved:\n  - " + "\n  - ".join(errors))
            return str(save_edit(ctx, app, fit))
        return self._run(work)

    def _apps_using(self, id_: str) -> list[int]:
        rows = db.list_applications(self._ctx().db())
        return [r["id"] for r in rows if id_ in json.dumps(r.get("selection") or {})]

    def builder_draft_versions(self, item_id: str, note: str = "", previous: Optional[list] = None) -> str:
        """Draft alternative wordings of a bullet or summary (a job). The drafts only
        fill the page's edit form; saving one is a normal "add" edit by the user."""
        from .generation import draft_versions

        def work(job: Job) -> list[dict]:
            return draft_versions(self._ctx(), item_id, note or "", previous or [], io=IO(out=job.lines.append),
                                  on_text=lambda text: setattr(job, "progress", {"partial": text}))
        return self._run(work)

    def builder_edit(self, op: str, args: dict) -> dict:
        """One change to master.yaml: update, add, delete, move or undo. Returns the new
        builder state, or the errors that stopped the change."""
        from .master_edit import EditError
        ed = self._editor()
        new_id = None
        try:
            if op == "update":
                ed.update(args["id"], args.get("changes") or {})
            elif op == "add":
                _, new_id = ed.add(args["kind"], args.get("parent"), args.get("data") or {}, args.get("after"))
            elif op == "delete":
                used = self._apps_using(args["id"])
                if used and not args.get("force"):
                    return {"ok": False, "confirm": used}
                ed.delete(args["id"])
            elif op == "move":
                ed.move(args["id"], int(args.get("delta") or 0))
            elif op == "undo":
                ed.undo()
            else:
                return {"ok": False, "errors": [f"unknown operation '{op}'"]}
        except EditError as e:
            return {"ok": False, "errors": e.errors}
        except MasterError as e:
            return {"ok": False, "errors": e.errors}
        return {"ok": True, "new_id": new_id, **self.builder_state()}

    # ------------------------------------------------------------ settings

    def get_settings(self) -> dict:
        values = _settings_values()
        return {"file": str(settings_path()), "presets": PRESETS, "preset_info": PRESET_INFO,
                "fields": [{"key": k, "label": lab, "kind": kind, "help": hlp, "level": level,
                            "value": values.get(k) or ""}
                           for k, (lab, kind, hlp, level) in SETTINGS.items()]}

    def save_settings(self, values: dict) -> dict:
        from dotenv import set_key, unset_key
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        for k, v in values.items():
            if k not in SETTINGS:
                continue
            v = str(v).strip()
            if v:
                set_key(str(path), k, v, quote_mode="never")
            else:
                unset_key(str(path), k, quote_mode="never")
        reload_settings()
        return self.get_state()

    # ------------------------------------------------------------ setup

    def ollama_start(self) -> dict:
        ok = ollama_setup.start()
        return {"started": ok}

    def ollama_install(self, how: str) -> dict:
        if how == "winget" and ollama_setup.install_with_winget():
            return {"opened": "winget"}
        import webbrowser
        webbrowser.open(ollama_setup.DOWNLOAD_URL)
        return {"opened": "browser"}

    def ollama_long_context(self) -> str:
        """Create the 16k-context name of the main model that thorough mode uses."""
        base = self._ctx().cfg.llm_base_url
        return self._run(lambda job: ollama_setup.create_long_context(base))

    def ollama_pull(self, model: str) -> str:
        base = self._ctx().cfg.llm_base_url

        def work(job: Job) -> str:
            def on(status: str, done: int, total: int) -> None:
                job.progress = {"status": status, "done": done, "total": total}
            ollama_setup.pull(base, model, on)
            return model
        return self._run(work)

    def import_data(self) -> dict:
        """Copy master.yaml, worked examples, the database and variants from a folder the
        user picks (e.g. the old data/ folder). Existing files are kept as .bak."""
        import webview
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER) if self._window else None
        if not picked:
            return {"copied": []}
        src = Path(picked[0])
        dst = self._ctx().cfg.data_dir
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in IMPORTABLE:
            s, d = src / name, dst / name
            if not s.exists():
                continue
            if d.exists():
                bak = d.with_name(d.name + ".bak")
                if bak.exists():
                    shutil.rmtree(bak) if bak.is_dir() else bak.unlink()
                d.rename(bak)
            shutil.copytree(s, d) if s.is_dir() else shutil.copy2(s, d)
            copied.append(name)
        return {"copied": copied, "from": str(src)}


def read_clipboard_text() -> str:
    """The clipboard's text on Windows (Win32 API through ctypes, so tkinter isn't
    needed in the package). Empty string elsewhere or when there's no text."""
    if not sys.platform.startswith("win"):
        return ""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32  # type: ignore[attr-defined]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    CF_UNICODETEXT = 13
    if not user32.OpenClipboard(None):
        return ""
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return ""
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr) if ptr else ""
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _open(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def main() -> None:
    import webview
    if is_portable():
        ensure_portable_layout()
    reload_settings()
    api = Api()
    window = webview.create_window("CV Tailor", url=str(UI_DIR / "index.html"), js_api=api,
                                   width=1280, height=860, min_size=(900, 600))
    api._window = window
    # WebView2 keeps its cache in Data/webview, not in the user profile.
    data, _, _ = default_dirs()
    webview.start(private_mode=False, storage_path=str(data / "webview"))


if __name__ == "__main__":
    main()
