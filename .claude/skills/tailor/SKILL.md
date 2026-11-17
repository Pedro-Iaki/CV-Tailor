---
name: tailor
description: Tailor the user's CV to a job posting with CV Tailor and report the result. Use when the user pastes a job description, gives a job description file, or asks to tailor/create a CV or resume for a job.
argument-hint: "[job description text or path to a .txt file]"
---

# Tailor a CV for a job posting

Input: $ARGUMENTS

1. **Get the job description text.**
   - If the input is a path to an existing file, read it.
   - If it's pasted text, use it as is. Keep a URL on its own first line, because the tool records it as the job URL.
   - If it's only a URL, ask the user to paste the posting text. CV Tailor doesn't fetch or scrape job sites.
   - If there's no input, ask the user to paste the job description.

2. **Run the pipeline.**
   - Prefer the `cv-tailor` MCP server: call `tailor_cv` with `job_description`. Pass `lang` only if the user asked for a language.
   - If the MCP server isn't available, save the text to a temporary file outside the repo. Then run `.venv/Scripts/python apply.py new <file>` (on macOS/Linux, `.venv/bin/python`). Add `--yes` only after the user confirms a same-company warning.

3. **Handle expected failures by asking, not guessing.**
   - Same company and role in the last 60 days: tell the user when they applied and ask whether to continue. If yes, retry with `confirm_same_company=true` (CLI: `--yes`).
   - Identical posting already logged: say which application id it was. Only retry with `allow_duplicate=true` if the user wants a second copy.
   - Metadata extraction failed: ask for the company, the role and the language (en or pt), then retry with those values.
   - Doesn't fit on one page even after trimming: report the message and offer `update_selection` with fewer items.
   - WeasyPrint/Pango missing, or no API key: point to the README setup section.

4. **Report briefly:**
   - `Company | Role | lang`
   - the route: reused variant (with score) or generated
   - the PDF path
   - trimmed ids
   - **unmatched requirements**: skills the posting asks for that the master doesn't cover
   - **suggestions** from the selector. These are notes for the user and are never rendered in the CV.

5. **Optional wording drafts.** Do this only if the user asked for it ("with drafts", "improve the wording", `--generate`), or if they say yes when you offer. It needs `CV_ALLOW_GENERATION=true`; if the tool says generation is disabled, tell the user how to enable it and stop.
   - Call `propose_text(app_id)`. Show every draft **exactly as returned**, numbered, in this format:
     ```
     1. rephrase · job-acme · approve -> FACT (reusable)
        from acme-dbx-pipeline-01: <source text>
        EN: <en>
        PT: <pt>
        why: <rationale>   (note: <notes>, if any)
     ```
     Label UNSAFE drafts clearly: they are new claims the user must confirm are true.
   - Ask for one reply per draft, in order or numbered: `y`, `n`, `y, <comment>` (revise and show again) or `n, <comment>` (reject, comment kept as feedback).
   - Call `review_proposals(app_id, {proposal_id: reply})` with the user's replies **verbatim**. Never answer for the user, fill in missing replies, or reword a draft yourself. If a reply is missing or unclear, ask again.
   - Show any `revised` drafts in the same format and repeat until nothing is pending.
   - Report the facts added, the re-rendered PDF, and any unsafe items stored. Explain that unsafe items are not used in any CV until the user runs `apply.py unsafe promote <id>` in a terminal, and that no tool can do this for them.

6. **Offer next steps** (don't do them unprompted):
   - the other language: `render_application(app_id, lang="pt"|"en")`
   - a reusable selection, if this one looks good: `approve_variant(app_id, name="<topic>-v1")`
   - adjustments: show the current items with `get_application`, pick ids from `get_master_index`, then apply with `update_selection`

## Rules

- Never write, rephrase or translate CV text yourself, and never add text to the PDF. The CV may only contain text from `data/master.yaml` or from facts the user approved through `review_proposals`.
- If an unmatched requirement is something the user really has, suggest they add a bullet to `data/master.yaml` in their own words, in both `en` and `pt`. Never invent experience or fill it in yourself.
- Don't paste the contents of `data/` into other services.
