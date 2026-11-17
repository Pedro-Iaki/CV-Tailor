# CV Tailor

A local Python CLI and MCP server that builds one-page EN/PT-BR CV PDFs by selecting and ordering items from `data/master.yaml`. The design and its known limitations are in [docs/DESIGN.md](docs/DESIGN.md).

## The core rule (do not break it)

CV text comes only from **human-verified** sources. The selector LLM, and you when you use this project, only **selects and orders ids**.

Trust levels, highest first:
1. `data/master.yaml`: written by the user.
2. The `facts` table: LLM drafts the user approved with an explicit `y`. These can only reword or merge existing items, and a number guard downgrades drafts that add numbers. Unsafe items the user promoted by hand also land here.
3. The `unsafe_items` table: approved drafts that make **new claims**. They are never merged into the master, so they can't be selected or rendered. The only way out is `apply.py unsafe promote` in a terminal. There is deliberately **no MCP tool** for promotion; do not add one.

Implementation rules:
- Generation is opt-in (`CV_ALLOW_GENERATION`, off by default) and lives only in `cvtailor/generation.py`. Every draft is reviewed one by one. `y, <comment>` produces a revision that is shown again; it is never stored directly.
- The template receives only a `RenderInput` (master + facts, validated selection, lang, i18n). Never add meta, suggestions, job tags, proposals or unsafe items to it.
- To change a CV, change the selection (ids), go through the proposal review, or suggest that the user edit their master file. Never write experience the user didn't state.

## Layout

- `apply.py`: Typer CLI. Keep it thin; logic belongs in `cvtailor/`.
- `cvtailor/pipeline.py`: the `new`, `render`, `regen`, `edit` and `approve` operations. These are UI-free and take an `IO` object for output and prompts.
- `cvtailor/mcp_server.py`: MCP tools over the same pipeline. stdout is the protocol channel, so never `print` there.
- `cvtailor/llm.py`: Anthropic SDK calls. Structured output uses a schema built from the master (enums of real ids). The system prompt plus the master index form the cached prefix.
- `cvtailor/choices.py`: the choice format used with local models. Bullets are ranked per owner (`{id: rank}`, so no duplicates), bullet versions become one slot plus a version pick, and conflicting skill categories become whole layouts (`anyOf`) with per-category item lists. `decode()` maps the answer back to plain ids before validation. It never repairs.
- `cvtailor/careful.py`: careful mode (`new --careful`). Hints from tags and embeddings, a free-text analysis pass (with `data/worked_examples.yaml`), the constrained selection, then one review and revision. Free text only goes back to the model or to verbose output, never into the CV.
- `cvtailor/ratings.py`: `apply.py rate`, a local page (127.0.0.1) for blind 1-5 ratings of CVs (applications plus JSONL candidates), stored in the `cv_ratings` table.
- `cvtailor/local_llm.py`: `LocalClient`, used when `CV_LLM_BASE_URL` is set. It mimics `client.messages.create` over an OpenAI-compatible server (Ollama, llama.cpp, LM Studio), so `llm.py` prompts, schemas and validation are shared. Stdlib only.
- Other modules each own one pipeline concern (see the module table in docs/DESIGN.md): `schema`, `master`, `tags`, `validate`, `route`, `variants`, `trim`, `render`, `editor`, `db`, `stats`.

## Commands

```bash
.venv/Scripts/python -m pytest              # all tests (LLM mocked; PDF tests skip without Pango)
.venv/Scripts/python apply.py --help
.venv/Scripts/python apply.py validate-master
.venv/Scripts/python -m cvtailor.mcp_server # MCP server over stdio (normally started by .mcp.json)
```

On macOS/Linux, use `.venv/bin/python`. WeasyPrint needs Pango from the operating system (see the README). Without it, everything except PDF writing works.

## Conventions

- Validation reports **all** errors at once and never auto-repairs. Selection failures are rejected, not fixed.
- Every write to an application (row + PDF) is one transaction. On failure, no row is written and no partial files are left.
- Tests: use the fakes in `tests/conftest.py` (`FakeClient`, `FakeRenderer`, `make_ctx`). Never call the real API in tests except `tests/test_integration.py` (gated by `CV_INTEGRATION=1`).
- Anthropic SDK 2.x: don't send `temperature`. The current models reject sampling parameters. `CV_EFFORT` controls effort.
- MCP SDK 2.x: the server class is `MCPServer` (not `FastMCP`). Raise `ToolError` for failures the model should see.

## Privacy

`data/` (master, variants, applications.db), `output/` and `.env` hold personal data and are gitignored. Never commit them, paste their contents into external services, or put real personal data in tests or examples. Use `examples/` (fictional data) for demos.
