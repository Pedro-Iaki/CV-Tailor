# CV Tailor

CV Tailor turns a job posting into a tailored one-page CV PDF. It is built on one idea: **a language model is good at deciding which parts of your experience fit a job, and bad at describing that experience honestly.** So the model only chooses. It reads the posting, then selects and orders items from a master resume you wrote. Every word in the PDF comes from text a person checked.

It runs as a Python CLI, as an MCP server that Claude Code or Claude Desktop can drive in plain language, and as a portable desktop app. It works with the Anthropic API or with a local model (Ollama, llama.cpp, LM Studio). A CV can be in one or two languages: English, Brazilian Portuguese, Spanish, French, German or Italian.

This README explains how it works and why it is built this way. Deeper design notes and known limitations are in [docs/DESIGN.md](docs/DESIGN.md).

## The problem

Tailoring a CV for each application means choosing a summary, the bullets that match the posting, a skills layout and sometimes a language, then fitting it all on one page. That takes judgment, and judgment is what LLMs do well. They also invent experience, round numbers up and drift in wording, and on a CV that can't be allowed.

Asking the model to be careful isn't enough, so the project removes its ability to write CV text at all.

## The core rule

**CV text comes only from human-verified sources. The model selects and orders ids; it never writes CV text.**

The rule is enforced by the code's structure, not by prompts:

- **The model can only name things that exist.** Its output schema is generated from the master for each run, so every id field is an enum of real ids, and each job's bullet list only accepts that job's bullets.
- **The template never sees model output.** Rendering receives a `RenderInput` (master content, the validated selection, the language, fixed i18n strings) and nothing else. The model's suggestions, the posting's tags and the company name aren't part of it, so they can't leak into the PDF.
- **Bad selections are rejected, never repaired.** Validation reports every broken rule at once. The model gets one retry with the errors, then the run stops. A repaired selection would be one nobody chose.
- **Tests check the boundary.** Sentinel strings planted in suggestions and metadata must never appear in the rendered CV, and every line of rendered text must trace back to the master or the i18n file.

## How a CV is made

```
job posting -> meta (LLM, small) -> tags (code) -> duplicate check (code)
            -> reuse an approved variant (code) or select ids (LLM)
            -> validate (code) -> render + fit to one page (code) -> log (SQLite)
```

| Step | Done by | What happens |
|---|---|---|
| Meta | LLM | Company, role, posting language and URL. The call sees only the posting. These values name the output file; they never reach CV content. |
| Tags | Code | Match the master's tag vocabulary and aliases on word boundaries (`c#`, `node.js` and `ci/cd` work). |
| Duplicate check | Code | The same posting is blocked; the same company and role within 60 days asks for confirmation. |
| Route | Code | Score the posting's tags against approved variants. A close enough match reuses that variant and skips the selection call. |
| Select | LLM | Return ids only, as structured JSON. |
| Validate | Code | Check every selection rule; retry once with the errors, then stop. |
| Render and fit | Code | Render the chosen language. If it overflows one page, drop the least relevant lines in a fixed order. Trimming removes content and never changes text. |
| Log | Code | Write the application row and its PDF in one transaction. If anything fails, nothing is kept. |

Code does everything that doesn't need judgment. The model makes two small calls, and the system prompt plus the master index form a cached prompt prefix, so only the posting changes between runs.

### The master file

The master (`data/master.yaml`, gitignored; see [examples/master.example.yaml](examples/master.example.yaml) for fictional data) holds everything that can appear on a CV: profile and links, summaries, jobs, projects, education, skill categories in tiers (hands-on, working knowledge, familiar, spoken languages) and the section order. Every item has an id, tags, and a text per language.

Since the model can't reword anything, the master holds the alternatives instead:

- **Versions.** An item marked `alt_of: <id>` is another wording of an existing summary, bullet or skill category. The selector picks the one that fits the posting and never two versions of the same thing.
- **Skill layouts.** Alternative skill categories form whole layouts, such as "by area" versus "hands-on plus working knowledge". A CV uses one layout and never mixes them.
- **Headlines and descriptions.** A summary can bring its own headline. Alternative headlines (such as "Analytics Engineer") are offered only when the top of the posting names that role. Jobs and projects can have alternative description lines.
- **Gating.** A project with `only_for_tags` is visible to the selector only when the posting mentions one of those tags.

Languages are declared in the file (`languages: {primary: en, secondary: pt}`). The primary language is required on every item and is what the model reads. The secondary one may be missing on some items; a CV in that language leaves those items out instead of translating them. Fixed words (headings, month names) live in [templates/i18n.yaml](templates/i18n.yaml).

`validate-master` checks the whole file and reports every error at once: unique ids, texts present in the primary language, valid tags, dates and `alt_of` links, consistent limits, and each saved variant against the current master.

### Variants and outcomes

When a selection works well, it can be approved as a named **variant**. It must fit on one page in every language of the master. Later postings with similar tags reuse it without a selection call, which makes results faster, cheaper and consistent. Each variant stores a hash of its items' text, so it knows when the master changed under it: `ok`, `drifted` (text changed; still used, with a warning) or `stale` (no longer valid; not used).

Each application's status (screen, interview, offer, rejected, ghosted) is tracked in SQLite. `stats` reports callback rates by variant, language and route, shows the sample size next to every rate and flags small samples, so a choice of variant can be judged by results.

## Keeping drafted text trustworthy

Sometimes the right words aren't in the master. An optional, off-by-default feature (`CV_ALLOW_GENERATION`) lets a model draft wording without breaking the core rule, because nothing a model writes reaches a CV until the user approves that exact text. Where approved text goes depends on what it claims:

| Source | Holds | Can appear on a CV? |
|---|---|---|
| `master.yaml` | Text the user wrote | Yes |
| `facts` table | Approved rewordings of one item, or merges of 2-3 bullets of the same job | Yes, never together with the items it rewords |
| `unsafe_items` table | Approved drafts that make **new claims** | No, until the user promotes one from a terminal |

- Drafts are reviewed one at a time. `y` approves the exact text; `y, <comment>` asks for a revision, which is shown again and never stored directly; `n, <comment>` rejects it and keeps the comment as feedback.
- A **number guard** reclassifies a rewording that contains a number missing from its sources as a new claim.
- Promotion out of `unsafe_items` works only through `apply.py unsafe promote` in a terminal. There is deliberately no MCP tool for it, so a model driving the tool can never move its own claims into a CV.
- In the desktop app's master editor, a model can draft alternative wordings of a bullet or summary. A draft only fills the edit form, drafts with numbers or names missing from the original are flagged, and the user saves the final text themselves, which makes it text they wrote.

## Claude or a local model

The pipeline talks to one interface. By default that is the Anthropic SDK. Setting `CV_LLM_BASE_URL` swaps in `LocalClient`, which mimics the same calls over any OpenAI-compatible server, so prompts, schemas and validation are shared and nothing leaves the machine.

Local servers enforce the schema with grammar-constrained decoding, which can express more than the Anthropic API accepts. Small models make predictable mistakes, so local runs use a **choice format** ([cvtailor/choices.py](cvtailor/choices.py)) that makes most of them impossible to express:

- Bullets are ranked per job (`{id: rank}`, where 0 leaves a bullet out), so a bullet can't be listed twice.
- Versions of one bullet become a single slot plus a separate pick of wording.
- Conflicting skill categories are offered as whole layouts, so a mixed layout can't be written.
- The answer opens with a short list of key requirements, a cheap scratchpad for models that think little.

The answer is mapped back to plain ids by a fixed function that neither adds nor removes anything, then validated like any other selection.

**Careful mode** trades time for better choices. Code first computes hints (items whose tags match, and optionally the posting line each item is closest to by embeddings). The model then writes a free-text analysis, makes the constrained selection with that analysis as context, and reviews it once. The free text only goes back to the model, never into the CV. Worked examples in `data/worked_examples.yaml` show the model the reasoning wanted. The desktop app offers three speeds: Fast, Careful, and Thorough (careful mode with the model thinking at length, on a longer context window).

Choices can be judged on a local rating page (`apply.py rate`): each CV sits next to its posting and gets a blind 1-5 score, with the setup that made it revealed only after rating, and averages per setup.

## Three front ends, one pipeline

[cvtailor/pipeline.py](cvtailor/pipeline.py) holds the operations (`new`, `render`, `regen`, `edit`, `approve`) with no UI code. They take an `IO` object for output and prompts, so the same code prints to a terminal, collects a log for an MCP result, or feeds a progress view.

- **CLI** ([apply.py](apply.py)): `new` takes a posting and needs no flags. Other commands re-render in another language, edit a selection by hand as annotated YAML, approve variants, record statuses and show stats. `python apply.py --help` lists them all.
- **MCP server** ([cvtailor/mcp_server.py](cvtailor/mcp_server.py)): 16 tools over the same pipeline, so Claude can handle requests like "tailor my CV for this posting", "render #42 in Portuguese" or "which variant gets the most callbacks?". The tools take and return ids, never resume text. `update_selection` is the only way for Claude to change a CV, and it validates the new selection without trimming. Confirmations (same company, duplicate posting) are explicit parameters, so the model has to ask the user. The repository includes the Claude Code side: [.mcp.json](.mcp.json), [CLAUDE.md](CLAUDE.md) with the project rules, and a `/tailor` skill.
- **Desktop app** ([cvtailor/desktop.py](cvtailor/desktop.py), [cvtailor/ui/index.html](cvtailor/ui/index.html)): a pywebview window with tabs for a new CV, past applications, the master (a CV builder and editor), rating, setup and settings. The master editor changes `master.yaml` with a round-trip YAML library that keeps comments and layout, validates every change before writing it, and keeps earlier versions for undo. On first start without a master, a short wizard collects the user's own texts and writes a first one. The app can be built as a portable folder that keeps all data next to the executable.

PDFs are printed by headless Edge or Chrome when one is found, with WeasyPrint as the fallback.

## Code map

| Module | Responsibility |
|---|---|
| `schema.py` | Pydantic models for the master, selections, variants and meta |
| `master.py`, `langs.py` | Load and validate the master; tag vocabulary and model index; supported languages |
| `tags.py` | Deterministic tag extraction from a posting |
| `llm.py`, `local_llm.py` | Prompts, schemas built from the master, caching, retry with errors; the OpenAI-compatible client |
| `choices.py`, `careful.py` | The choice format for local models; careful mode |
| `validate.py` | Selection rules, reporting every error at once |
| `route.py`, `variants.py` | Scoring against approved variants; variant hashes and states |
| `render.py`, `trim.py` | HTML and PDF rendering; deterministic page fit |
| `pipeline.py` | The UI-free operations shared by every front end |
| `generation.py`, `facts.py` | Opt-in drafted wording and its review; approved facts |
| `db.py`, `stats.py` | SQLite with numbered migrations; callback rates |
| `editor.py`, `master_edit.py` | Editing a selection as YAML; editing the master from the app |
| `mcp_server.py`, `desktop.py`, `ratings.py` | MCP tools; desktop app; rating page |

## Trying it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Render a hand-written selection from the fictional example data. No API key needed.
export CV_MASTER=examples/master.example.yaml     # PowerShell: $env:CV_MASTER = "..."
python apply.py validate-master
python apply.py render-file examples/selection.example.json
```

For real use, copy [.env.example](.env.example) to `.env` (an Anthropic API key, or `CV_LLM_BASE_URL` and a model name for a local server), put your own master in `data/master.yaml`, and run `python apply.py new job.txt`. `data/`, `output/` and `.env` hold personal data and are gitignored. PDF writing needs Edge or Chrome, or WeasyPrint with the Pango library from the operating system. The desktop app runs from source with `python cv_tailor_app.py`, and `build_portable.ps1` packages it.

## Tests

```bash
python -m pytest
CV_INTEGRATION=1 python -m pytest tests/test_integration.py   # optional, calls the real API
```

The model is replaced by a fake client in every test except the gated integration test. Tests cover each master validation rule with passing and failing fixtures, selection validation, the choice format, routing, variant states, page fit, the MCP tools including their error paths, the master editor, and the trust boundary described above. PDF tests are skipped when their engine (WeasyPrint with Pango, or Edge/Chrome) isn't installed.
