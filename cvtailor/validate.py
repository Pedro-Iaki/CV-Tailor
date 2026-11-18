"""Selection validation against the master. Rejects, never repairs."""

from __future__ import annotations

from collections import Counter

from .master import alt_conflicts, item_index, skill_item_keys
from .schema import TIERS, Master, Selection


def validate_selection(sel: Selection, master: Master) -> list[str]:
    errors: list[str] = []
    idx = item_index(master)
    lim = master.limits

    # 6. summary
    s = idx.get(sel.summary_id)
    if s is None:
        errors.append(f"summary_id '{sel.summary_id}' does not exist in the master")
    elif s.kind != "summary":
        errors.append(f"summary_id '{sel.summary_id}' is a {s.kind}, not a summary")

    # headline
    if sel.headline_id is not None and sel.headline_id not in {h.id for h in master.profile.headline_alts}:
        errors.append(f"headline_id '{sel.headline_id}' is not a headline option in the master")

    # projects
    for pid in sel.project_ids:
        p = idx.get(pid)
        if p is None:
            errors.append(f"project '{pid}' does not exist in the master")
        elif p.kind != "project":
            errors.append(f"'{pid}' in project_ids is a {p.kind}, not a project")

    # 1-3. bullets, keyed by job, project or education id
    job_ids = [j.id for j in master.jobs]
    for owner, bullet_ids in sel.bullets.items():
        o = idx.get(owner)
        if o is None:
            errors.append(f"bullets: '{owner}' is not a job, project or education entry in the master")
            continue
        if o.kind not in ("job", "project", "education"):
            errors.append(f"bullets: '{owner}' is a {o.kind}, not a job, project or education entry")
            continue
        for bid in bullet_ids:
            b = idx.get(bid)
            if b is None:
                errors.append(f"bullet '{bid}' (under {owner}) does not exist in the master")
            elif b.kind != "bullet":
                errors.append(f"'{bid}' (under {owner}) is a {b.kind}, not a bullet")
            elif b.parent != owner:
                errors.append(f"bullet '{bid}' belongs to {b.parent}, not {owner}")
        n = len(bullet_ids)
        if o.kind == "education":
            pass  # detail lines: any number, counted in the total
        elif o.kind == "job":
            if n < lim.min_bullets_per_job:
                errors.append(f"{owner}: {n} bullets, below min_bullets_per_job ({lim.min_bullets_per_job})")
            if n > lim.max_bullets_per_job:
                errors.append(f"{owner}: {n} bullets, above max_bullets_per_job ({lim.max_bullets_per_job})")
        else:
            if n and owner not in sel.project_ids:
                errors.append(f"bullets: project '{owner}' has bullets but is not in project_ids")
            if n > lim.max_bullets_per_project:
                errors.append(
                    f"{owner}: {n} bullets, above max_bullets_per_project ({lim.max_bullets_per_project})")
    for job_id in job_ids:
        if job_id not in sel.bullets:
            errors.append(f"job '{job_id}' is missing from bullets (every job must appear)")

    # descriptions: one alternative per job or project
    for owner, did in sel.descriptions.items():
        o, d = idx.get(owner), idx.get(did)
        if o is None or o.kind not in ("job", "project"):
            errors.append(f"descriptions: '{owner}' is not a job or project in the master")
        elif d is None or d.kind != "description" or d.parent != owner:
            errors.append(f"descriptions: '{did}' is not a description alternative of {owner}")
        elif o.kind == "project" and owner not in sel.project_ids:
            errors.append(f"descriptions: project '{owner}' is not in project_ids")

    # 7. skills
    for tier in TIERS:
        for cid in getattr(sel.skill_category_order, tier):
            c = idx.get(cid)
            if c is None:
                errors.append(f"skill category '{cid}' does not exist in the master")
            elif c.kind != "skill_category":
                errors.append(f"'{cid}' in skill_category_order.{tier} is a {c.kind}, not a skill category")
            elif c.tier != tier:
                errors.append(f"skill category '{cid}' belongs to tier {c.tier}, not {tier} (tiers cannot change)")

    # skill items: a subset of the category's own items, for a selected category
    chosen_cats = {c for t in TIERS for c in getattr(sel.skill_category_order, t)}
    for cid, texts in sel.skill_items.items():
        c = idx.get(cid)
        if c is None or c.kind != "skill_category":
            errors.append(f"skill_items: '{cid}' is not a skill category in the master")
            continue
        if cid not in chosen_cats:
            errors.append(f"skill_items: category '{cid}' is not in skill_category_order")
        if not texts:
            errors.append(f"skill_items: '{cid}' lists no items (leave the category out instead)")
        own = {k for i in c.obj.items for k in skill_item_keys(i)}
        for text in texts:
            if text not in own:
                errors.append(f"skill_items: '{text}' is not an item of {cid} (items can't move between categories)")
        for text, count in Counter(texts).items():
            if count > 1:
                errors.append(f"skill_items: '{text}' appears {count} times in {cid}")
        lo = c.obj.min_items
        if lo and texts and len(set(texts)) < lo:
            errors.append(f"skill_items: {cid} shows {len(set(texts))} items, below its min_items ({lo})")

    # required categories: listed unless one of their alternatives is
    conflicts_ = alt_conflicts(master)
    for tier in TIERS:
        for c in master.skills.tier(tier):
            if c.required and c.id not in chosen_cats and not (conflicts_.get(c.id, set()) & chosen_cats):
                errors.append(f"skill category '{c.id}' is required (unless one of its alternatives is chosen)")

    # 4. duplicates anywhere
    all_ids = [sel.summary_id]
    for ids in sel.bullets.values():
        all_ids.extend(ids)
    all_ids.extend(sel.project_ids)
    all_ids.extend(sel.descriptions.values())
    for tier in TIERS:
        all_ids.extend(getattr(sel.skill_category_order, tier))
    for id_, count in Counter(all_ids).items():
        if count > 1:
            errors.append(f"id '{id_}' appears {count} times in the selection")

    # 8. alternatives (master alt_of) and approved rewordings (facts) of the same
    # item never appear together
    selected = set(all_ids)
    conflicts = alt_conflicts(master)
    for a, b in sorted({tuple(sorted((x, y))) for x in selected for y in conflicts.get(x, ()) if y in selected}):
        if b in (getattr(idx[a].obj, "based_on", None) or []):
            a, b = b, a
        if a in (getattr(idx[b].obj, "based_on", None) or []):
            errors.append(f"'{b}' is a rewording of '{a}'; select one or the other, not both")
        else:
            errors.append(f"'{a}' and '{b}' are versions of the same item; select one or the other, not both")

    # 5. totals
    total = sel.total_bullets()
    if total > lim.max_total_bullets:
        errors.append(f"{total} bullets in total, above max_total_bullets ({lim.max_total_bullets})")
    if len(sel.project_ids) > lim.max_projects:
        errors.append(f"{len(sel.project_ids)} projects, above max_projects ({lim.max_projects})")
    return errors
