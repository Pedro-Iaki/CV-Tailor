# CV Tailor

Turns a job description into a tailored, one-page CV PDF in English or Brazilian Portuguese, built from a single master YAML file. It runs as a Python CLI and as an **MCP server**, so you can drive it from Claude Code ("tailor my CV for this posting", "render #42 in Portuguese") or from a terminal.

**Core rule:** the model only selects and orders existing content by id. Every word in the PDF comes from text a human verified: `data/master.yaml`, wording you approved one draft at a time (an optional, opt-in feature), and `templates/i18n.yaml`.

## Highlights

- **MCP server with 16 tools** (MCP Python SDK 2.x) over the same pipeline as the CLI, with tool annotations, errors the model can act on, and confirmations as explicit parameters. The repository also ships the Claude Code side: `.mcp.json`, a `CLAUDE.md` with the project rules, and a `/tailor` skill. See [Using it from Claude Code](#using-it-from-claude-code-mcp).
- **Can't invent experience, by construction.** The selection schema is generated from the master, so every id field is an enum of ids that exist. The template only ever receives validated master content, and tests check that every line of the PDF traces back to it.
- **Human-in-the-loop drafting (opt-in).** A model can draft better wording, but nothing reaches a CV until you approve that exact text. New claims are quarantined, and no MCP tool can release them.
- **Deterministic wherever possible.** Tag matching, routing to approved variants, validation and one-page trimming are plain code. The model is used for two small calls, with prompt caching.
- **Anthropic API or a local model** (Ollama, llama.cpp, LM Studio) behind the same interface.
- **Outcome tracking** in SQLite: callback rates by variant, language and route.
- **186 tests**, with the model mocked.

How it's built and why: [docs/DESIGN.md](docs/DESIGN.md).

```
job description -> meta (LLM, tiny) -> tags (code) -> duplicate check (code)
               -> route to an approved variant (code) or select ids (LLM)
               -> validate (code) -> render + fit to one page (code) -> log (SQLite)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # then set ANTHROPIC_API_KEY
mkdir data
cp examples/master.example.yaml data/master.yaml   # then replace with your content
python apply.py validate-master
```

`data/`, `output/` and `.env` hold personal data and are gitignored, so your CV content never ends up in the repository.

**Try it without an API key** on the fictional example data. This validates the example master and renders a hand-written selection to `output/dev/` in both languages (PDF writing needs Pango, see below):

```bash
# PowerShell: $env:CV_MASTER = "examples/master.example.yaml"
export CV_MASTER=examples/master.example.yaml
python apply.py validate-master
python apply.py render-file examples/selection.example.json
```

### Installing WeasyPrint (Pango)

WeasyPrint needs the Pango text library from the operating system:

- **Linux (Debian/Ubuntu):** `sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0`
- **macOS:** `brew install pango`
- **Windows:**
  1. Install MSYS2 from https://www.msys2.org (or `winget install MSYS2.MSYS2`).
  2. In the MSYS2 shell, run `pacman -S mingw-w64-x86_64-pango`.
  3. Point WeasyPrint at the DLLs before you run the tool. In PowerShell: `$env:WEASYPRINT_DLL_DIRECTORIES = "C:\msys64\mingw64\bin"`. To make it permanent, set it as a user environment variable.

To check the install, run `python -c "import weasyprint"`. Everything except PDF rendering works without Pango, and tests that need it are skipped.

## Using it from Claude Code (MCP)

`cvtailor/mcp_server.py` exposes the same pipeline as MCP tools, so you can work in plain language inside Claude Code or Claude Desktop. For example: "tailor my CV for this posting", "render #42 in Portuguese", "mark 41 and 43 as rejected", "which variant gets the most callbacks?".

| Tool | What it does |
|---|---|
| `tailor_cv` | Full `new` pipeline (supports `dry_run`, `lang`, overrides, and the duplicate/same-company confirmations) |
| `list_applications`, `get_application` | Browse logged applications; the details include the selected items' text, suggestions and status history |
| `set_status` | Update the status of one or more applications |
| `get_stats`, `list_variants` | Callback rates with `n` and low-sample flags; variant states |
| `render_application`, `regen_application` | Re-render (e.g. the other language) or re-select |
| `update_selection` | Replace a selection by ids, validated and with no trimming. The id-only equivalent of `edit` |
| `approve_variant` | Save a selection as a reusable variant |
| `get_master_index`, `validate_master` | Selectable ids with their text (no contact data); master validation |
| `propose_text`, `review_proposals` | Opt-in drafted wording. Claude shows each draft and passes your `y` / `n` / `y, comment` replies through verbatim |
| `list_facts`, `list_unsafe` | Approved facts; unsafe items awaiting promotion. There is no promote tool |

Apart from the opt-in draft review, the tools take ids, never resume text, so the core rule still holds when Claude drives the tool. The design decisions behind the server are in [docs/DESIGN.md](docs/DESIGN.md#mcp-server-and-claude-code-integration).

**Claude Code:** the repository ships with
- `.mcp.json`, which registers the server for this project (Claude Code asks you to approve it on first use)
- [CLAUDE.md](CLAUDE.md), the project rules for Claude
- a `/tailor` skill ([.claude/skills/tailor/SKILL.md](.claude/skills/tailor/SKILL.md)): run `/tailor` and paste a posting, or `/tailor path/to/job.txt`

`.mcp.json` points at the Windows venv path. On macOS/Linux, change `command` to `.venv/bin/python`.

**Claude Desktop:** add this to `claude_desktop_config.json`, using absolute paths:

```json
{
  "mcpServers": {
    "cv-tailor": {
      "command": "C:\\path\\to\\CV-Tailor\\.venv\\Scripts\\python.exe",
      "args": ["-m", "cvtailor.mcp_server"],
      "cwd": "C:\\path\\to\\CV-Tailor"
    }
  }
}
```

If your client doesn't support `cwd`, add `"env": {"PYTHONPATH": "C:\\path\\to\\CV-Tailor"}` instead. Data paths always resolve relative to the project folder.

## Daily use

```bash
python apply.py new                 # paste the job description, then Ctrl-D (Ctrl-Z Enter on Windows)
python apply.py new job.txt --open  # from a file, then open the PDF
python apply.py new --clip          # from the clipboard (needs pyperclip)
```

`new` needs no flags. If the first non-empty line of the text is a URL, it becomes the job URL. Example output:

```
Acme | Senior Data Engineer | pt
tags: azure, databricks, delta, spark (4)
route: reused platform-v1 (0.74) | runner-up: analytics-v2 (0.58)
trimmed: none
#42 -> output/0042_2026-11-16_acme/Full_Name_CV_Senior_Data_Engineer_PT.pdf
```

Useful flags:

| Flag | Effect |
|---|---|
| `--lang en\|pt` | CV language. The default is the language the posting is written in. |
| `--company`, `--role`, `--url` | Override the detected values. With all three of company, role and lang, the meta call is skipped. |
| `--force-generate` / `--variant NAME` | Skip routing and generate / force a specific approved variant. |
| `--dry-run` | Run up to selection and print it with each item's text. Writes nothing. |
| `--no-trim` | Fail instead of dropping content when the CV overflows one page. |
| `--verbose` | Show which phrase matched each tag, token usage (including cache reads and writes) and timings per step. |
| `--yes`, `--allow-duplicate` | Continue past the same-company warning / allow the exact same posting twice. |

### Workflow

1. `new` a few postings. If a selection is poor, run `regen <id>`, or fix it by hand with `edit <id>` (the selection opens as annotated YAML in `$EDITOR`, Notepad on Windows, nano elsewhere).
2. When a selection is good and reusable, run `approve <id> --name platform-v1`. It must fit on one page in both EN and PT.
3. Later postings with similar tags reuse that variant with no selection call (`ROUTE_THRESHOLD`, default 0.6).
4. Record outcomes with `status <id>... screen|interview|offer|rejected|ghosted`, `discard <id>` (generated but never sent), or `mark-ghosted --older-than 30`.
5. Use `stats` and `variants` to see callback rates by variant, language and route, with `n` and low-sample flags.

### Optional: drafted wording with item-by-item approval

This is off by default. Set `CV_ALLOW_GENERATION=true` in `.env`, then run `apply.py new --generate` or `apply.py generate <id>`. The model drafts up to 5 items, and you answer each one:

```
[1] #12 rephrase | job-acme | approve -> FACT (reusable)
    from acme-dbx-pipeline-01: Built Databricks pipelines in PySpark processing 2 TB/day...
    EN: Engineered PySpark pipelines on Databricks moving 2 TB/day of payment events...
    PT: Desenvolvi pipelines PySpark no Databricks levando 2 TB/dia...
    why: Mirrors the posting's "design PySpark pipelines".
    reply: y, start with "Designed" like the posting
```

| Reply | Effect |
|---|---|
| `y` | Approve this exact text |
| `y, <comment>` | Revise using your comment and show it again. Nothing is stored until you answer `y` (at most 3 revisions) |
| `n` | Reject |
| `n, <comment>` | Reject; the comment is saved and passed as feedback the next time drafts are made |

Where approved text goes depends on what kind of draft it is:

| Draft | On `y` it becomes | Used in CVs? |
|---|---|---|
| **rephrase** of one item, or **merge** of 2-3 bullets of the same job | a **fact** (`facts` table): reusable, selectable like any master item. It replaces its sources in the current CV, and the PDF is re-rendered | yes |
| **new** claim (e.g. for an unmatched requirement) | an **unsafe item** (`unsafe_items` table) | **no**, not until you promote it |

- **Promoting unsafe items:** use `apply.py unsafe list`, then `apply.py unsafe promote <id> [--edit]`. You confirm each item is true, and can edit the EN/PT text first. Promotion works only in the terminal: there is deliberately no MCP tool for it, so Claude can never promote a claim on its own.
- **Number guard:** a rephrase or merge that contains a number missing from its sources is automatically downgraded to *new*, which makes it unsafe.
- **Reworded items:** a fact is never selected together with the item it rewords.
- **Managing facts:** `apply.py facts list [--full]` and `apply.py facts remove <id>`. Removing a fact makes variants that use it stale.

### All commands

```
validate-master                 check master + every saved variant (ok / drifted / stale)
new [FILE|-] [flags]            main pipeline
batch FILE... [--lang] [--force-generate] [--no-trim] [--verbose]
render <id> [--lang en|pt]      re-render the stored selection, no LLM call (e.g. the other language)
edit <id>                       edit the selection, re-render, mark as edited
regen <id>                      re-run selection and update the row in place
open <id>                       open the PDF
approve <id> --name N [--replace]
variants                        name, state, tag count, applications, screen+ rate
status <id>... <status> [--note]
discard <id>...
mark-ghosted --older-than N [--yes]
list [--status S] [--since YYYY-MM-DD] [--limit 30]
stats [--min-age 14] [--since YYYY-MM-DD]
render-file selection.json [--lang]   (dev) render a hand-written selection
generate <id>                   draft wording, then review each draft (CV_ALLOW_GENERATION=true)
facts list [--full] | facts remove <id>...
unsafe list | unsafe promote <id>... [--edit] | unsafe drop <id>...
```

## Master file

See [examples/master.example.yaml](examples/master.example.yaml). Besides summaries, jobs, skills and education, the format holds everything a designed one-page CV shows:

- `profile.headline`: the line under your name. `profile.links` have a visible `label` that the PDF links to `url`.
- Jobs: `company` is optional (freelance work), and `description` is a context line shown above the bullets.
- Projects: `name`, optional `start`/`end`, `links`, a `description` and their own `bullets`, which the selector picks like job bullets. A project needs a description, bullets or both.
- Education: an optional `note` (honours), and `expected: true` when `end` is a planned date.
- Education: `bullets` are optional detail lines (coursework), selectable like project bullets and shown in italics.
- `layout.sections`: the order of sections on the page. A section left out isn't rendered.
- `limits.max_bullets_per_project`. `max_total_bullets` counts job, project and education lines together.

**Validated alternatives.** When you rewrite something for one application and want to keep both versions, add the new one to the master instead of replacing the old one:

- `alt_of: <id>` on a summary, bullet or skill category marks it as another version of that item. The selector never picks both, nor two versions of the same item, and it picks the one that fits the job. A bullet's alternatives stay under the same job or project.
- A summary can carry its own `headline`, which replaces `profile.headline` when that summary is picked.
- `description_alts` on a job or project: other versions of its description, each with an id and tags. A description's `alt_of` can list bullets of the same entry that it already covers.
- Skill categories with `alt_of` form alternative layouts of the same skills (for example "by area" vs. one "Hands-on" line plus a "Working knowledge" line). A skill may repeat between a category and its alternatives. Two categories that replace the same one can be used together, as halves of one layout.
- `skills.languages`: spoken languages, always rendered last, one line per category.

Untranslated text is not allowed: every alternative needs `en` and `pt`.

Any short text can be one string used in both languages, or an `{en, pt}` pair. `validate-master` checks these rules and reports every error at once:

- ids are unique across the file
- every text has non-empty `en` and `pt`
- skills sit in the tiers `hands_on`, `working`, `familiar` and `languages`, and no skill appears twice, except between a category and its alternatives
- every `alt_of` points to an existing item of the same kind (a bullet's to a bullet of the same job or project)
- tags match `^[a-z0-9]+(-[a-z0-9]+)*$`
- dates are `YYYY-MM` or `YYYY`, and `end` is not before `start`
- every alias belongs to a used tag, and no alias is shared between tags
- the limits are consistent, and no section is listed twice in `layout.sections`

**Tag aliases** decide which job descriptions match which tags. A tag also matches itself, with a hyphen read as a hyphen, a space or nothing, so `azure-data-factory` matches "Azure Data Factory". Tags shorter than 3 characters (`go`, `r`, `ai`) only match through an alias. To see what matched and why, run `new --dry-run --verbose`.

**Variant states:** `ok` means the variant is valid and its text is unchanged. `drifted` means some selected text changed since approval: it is still used, with a warning. `stale` means the variant no longer validates against the master, for example because an id was removed or a job was added: it is not used. To fix a stale variant, run `regen <id>` and then `approve <id> --name N --replace`.

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | required for meta and selection calls (not needed with a local model) |
| `CV_MODEL` | `claude-haiku-5-5` | with a local model, its name on the server (`qwen3:8b`) |
| `CV_EFFORT` | `low` | reasoning effort sent with each call; leave it empty for models without the effort parameter. Ignored for local models |
| `CV_LLM_BASE_URL` | | set it to use a local model instead of the Anthropic API (see below) |
| `CV_LLM_API_KEY` | | only if your local server requires a key |
| `CV_LLM_TIMEOUT` | `600` | seconds per local call |
| `CV_LLM_REASONING` | | local models only: sent as `reasoning_effort` (`low`, `medium`, `high`). `low` keeps thinking models fast and inside the context window |
| `ROUTE_THRESHOLD` | `0.6` | |
| `CV_ALLOW_GENERATION` | `false` | enables drafted wording with item-by-item approval |
| `CV_PDF_NAME` | `{name}_CV_{role}_{LANG}` | also accepts `{company}` and `{lang}` |
| `CV_MASTER`, `CV_DATA_DIR`, `CV_OUTPUT_DIR` | `data/master.yaml`, `data/`, `output/` | optional path overrides |

### Using a local model (Ollama, llama.cpp, LM Studio)

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint and JSON-schema output works. Nothing leaves your machine. With Ollama:

```bash
ollama pull qwen3:8b
# Ollama's default context window can be too small for the prompt (about 3-4k tokens
# plus the job description); a cut-off prompt fails validation. Raise it before starting:
#   Windows: setx OLLAMA_CONTEXT_LENGTH 8192   (then restart Ollama)
#   macOS/Linux: OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Then in `.env`:

```
CV_LLM_BASE_URL=http://localhost:11434/v1   # LM Studio: http://localhost:1234/v1, llama.cpp: http://localhost:8080/v1
CV_MODEL=qwen3:8b
```

The model's job is the same as with Claude: it only picks and orders ids. The server enforces the schema while decoding, so the model can only output ids that exist in your master, and with a local model the schema also carries the per-job and per-project bullet counts and the project limit. Local models also get the choice format (`cvtailor/choices.py`): bullets that are versions of each other (`alt_of`) appear as one item with `versions`, and the model picks the wording separately; skill categories that conflict are offered as whole layouts, so a mixed layout can't be expressed at all; and the answer starts with a short `key_requirements` list that the model uses as a scratchpad. The answer is mapped back to plain ids before validation. The total-bullet limit and the other selection rules are checked in code as usual; an invalid answer is retried once with the errors, then the run stops. Small models get rejected more often, so if `new --dry-run` keeps failing validation, try a bigger model (14B or more) or raise `max_total_bullets`.

Notes:
- Meta extraction (company, role, language) is the easiest task. Selection is harder, and drafted wording (`CV_ALLOW_GENERATION`) is the hardest: expect weaker EN/PT drafts from small models. Every draft is still reviewed by you, and the number guard still applies.
- Prompt caching is Anthropic-only, so `cache read` stays 0, and token counts come from the server's report.
- Qwen3 models may think before answering. A `<think>` block at the start of the answer is removed; the thinking still costs time, so a non-thinking model is faster. In a test, `qwen3:8b` with thinking off misread an English posting as Portuguese, so keep thinking on unless your GPU is too slow for it.

**Check the speed first.** `ollama run qwen3:8b --verbose "Write three sentences about rivers."` prints an `eval rate`. A run of `new` produces about 1,500 tokens with thinking, so at 40 tokens/s it takes under a minute, and at 4 tokens/s about 5 minutes. Then run `ollama ps`: PROCESSOR should say `100% GPU`.

**8 GB GPUs on Windows.** Other apps (browsers, Steam, voice tools) also use video memory. When the model doesn't fit in what's left, Windows silently spills it into system RAM and it runs very slowly, even though `ollama ps` still says `100% GPU`. Task Manager > Performance > GPU shows this as "Shared GPU memory" in use. If plenty of memory is free and it still spills (on an RX 7600, 4 tokens/s on both ROCm and Vulkan, even for the 4B model), restart Windows: that brought it to 75 tokens/s. Otherwise close GPU-heavy apps, or use `qwen3:4b`. Two settings shrink the context cache by half: `setx OLLAMA_FLASH_ATTENTION 1` and `setx OLLAMA_KV_CACHE_TYPE q8_0` (restart Ollama after `setx`).

## Implementation notes

- **Structured output:** the selection JSON schema is built from the master, so every id field is an enum of real ids, with one enum per job and per skill tier. The output is then re-validated in code. If validation fails, the tool retries once with the errors appended, then aborts. A selection is never repaired silently. For models without native structured outputs, the tool falls back to a tool call.
- **Prompt caching:** the system prompt and the master index form a cached prefix, and only the job description changes between calls. Haiku-class models need a prefix of at least 512 tokens and use a 5-minute cache, so a small master or widely spaced runs can show `cache read 0`. That is expected.
- **Temperature:** no sampling parameters are sent. Current models (including Haiku 5.5) and the current API reject non-default sampling parameters, so none are sent. Variance is kept low with `effort=low`, enum-constrained output and code validation.
- **Trust boundary:** the template only ever receives a `RenderInput` (master + validated selection + language + i18n). Suggestions, unmatched requirements, job tags and meta fields aren't part of it. Tests check this with sentinel strings, and also check that every word in the rendered CV traces back to the master or i18n.
- **Atomicity:** an application row and its PDF are written in one transaction. On failure, the row is rolled back and the new folder is removed.

## Tests

```bash
python -m pytest
CV_INTEGRATION=1 python -m pytest tests/test_integration.py   # optional, calls the real API
```

The LLM is mocked in all tests except the integration test. PDF-level tests (page count, text extraction) run only when WeasyPrint can load Pango.

The design, its trade-offs and known limitations are described in [docs/DESIGN.md](docs/DESIGN.md).
