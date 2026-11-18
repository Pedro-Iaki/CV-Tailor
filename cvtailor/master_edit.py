"""Edit master.yaml from the desktop app's builder, keeping its comments and layout.

The master is the user's own file (trust level 1): this module only applies changes
the user makes by hand in the builder. Every operation loads the file (ruamel.yaml
round-trip, so comments, quotes and order survive), changes one thing, validates the
whole master with the same checks as `validate-master`, and only then writes it. The
previous version goes to a history folder (the newest HISTORY_KEEP are kept) so
`undo` can restore it. Invalid results are rejected, never repaired.
"""

from __future__ import annotations

import io
import re
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from .config import folder_slug
from .langs import LANGUAGES
from .langs import name as lang_name
from .master import MasterError, parse_master
from .schema import TIERS, Languages, Master

HISTORY_KEEP = 30
_TEXT_KINDS = ("summary", "bullet", "description", "headline")
_ENTRY_KINDS = ("job", "project", "education")
# Entry fields the builder can change: name -> (kinds, how it is written)
_ENTRY_FIELDS = {
    "title": (("job",), "texts"), "degree": (("education",), "texts"),
    "name": (("project",), "text"), "institution": (("education",), "text"),
    "note": (("education",), "text?"), "description": (("job", "project"), "text?"),
    "company": (("job",), "plain?"), "location": (("job",), "plain?"),
    "start": (_ENTRY_KINDS, "date?"), "end": (_ENTRY_KINDS, "date?"),
    "expected": (("education",), "bool"),
}


class EditError(Exception):
    """A change was rejected; `errors` lists why (shown to the user)."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 100000  # never wrap long lines
    y.indent(mapping=2, sequence=4, offset=2)
    # write None as "null" (ruamel's default is an empty value)
    y.representer.add_representer(
        type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    return y


def _space_braces(text: str) -> str:
    """ruamel writes flow maps as {a: 1}; the master uses { a: 1 }. Restores the spaces
    outside quoted strings, and gives them back from the gap before a trailing comment
    so comments stay in their column."""
    out = []
    for line in text.splitlines(keepends=True):
        res: list[str] = []
        quote: Optional[str] = None
        added = 0
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                res.append(ch)
                if ch == quote and line[i - 1] != "\\":
                    quote = None
            elif ch in "\"'":
                quote = ch
                res.append(ch)
            elif ch == "#" and (i == 0 or line[i - 1] == " "):
                text_before = "".join(res)
                gap = len(text_before) - len(text_before.rstrip(" "))
                take = min(added, max(0, gap - 2))
                res = [text_before[: len(text_before) - take]] if take else res
                res.append(line[i:])
                break
            elif ch == "{" and i + 1 < len(line) and line[i + 1] not in " }":
                res.append("{ ")
                added += 1
            elif ch == "}" and res and res[-1] not in (" ", "{ ") and not res[-1].endswith(" "):
                res.append(" }")
                added += 1
            else:
                res.append(ch)
            i += 1
        out.append("".join(res))
    return "".join(out)


def _q(s: str) -> DoubleQuotedScalarString:
    return DoubleQuotedScalarString(str(s))


def _flow(items: list[str]) -> CommentedSeq:
    seq = CommentedSeq([str(x) for x in items])
    seq.fa.set_flow_style()
    return seq


def doc_languages(doc: Any) -> Languages:
    """The languages a master document declares (English + Portuguese without the key)."""
    raw = doc.get("languages") if isinstance(doc, dict) else None
    try:
        return Languages.model_validate(dict(raw)) if isinstance(raw, dict) else Master.model_fields[
            "languages"].default_factory()
    except Exception as e:  # noqa: BLE001 - reported like any other problem with the file
        raise EditError([f"languages: {e}"]) from None


def _texts(v: Any, langs: list[str], old: Any = None) -> Any:
    """A per-language value as a flow map ({ en: "...", es: "..." }), the file's style.
    Languages the change doesn't mention keep their old text; empty ones are left out.
    A plain string stays a string (the same text in every language)."""
    if not isinstance(v, dict):
        return _q(str(v).strip())
    merged = dict(old) if isinstance(old, dict) else {}
    merged.update({k: str(x).strip() for k, x in v.items() if k in LANGUAGES})
    order = [*langs, *(k for k in merged if k not in langs)]
    m = CommentedMap([(k, _q(merged[k])) for k in order if str(merged.get(k) or "").strip()])
    m.fa.set_flow_style()
    return m


def _lang_values(data: dict[str, Any]) -> dict[str, str]:
    return {k: str(v).strip() for k, v in data.items() if k in LANGUAGES and str(v or "").strip()}


def _skill_items(value: Any, langs: list[str]) -> CommentedSeq:
    """A skill line's items: strings, or a text per language. The same text in every
    language is written once, as a string."""
    items = CommentedSeq()
    for it in value or []:
        if isinstance(it, dict):
            texts = _lang_values(it)
            if texts.get(langs[0]):
                items.append(_q(texts[langs[0]]) if set(texts.values()) == {texts[langs[0]]}
                             else _texts(texts, langs))
        elif isinstance(it, str) and it.strip():
            items.append(_q(it.strip()))
    return items


def _need_primary(texts: dict[str, str], langs: list[str], what: str = "text") -> None:
    if not texts.get(langs[0]):
        raise EditError([f"the {lang_name(langs[0])} {what} can't be empty"])


def _clean_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = re.split(r"[,\s]+", tags)
    return [t.strip().lower() for t in (tags or []) if str(t).strip()]


def create_master(path: Path, data: dict[str, Any]) -> Master:
    """Write a first master.yaml from the setup wizard's answers: the languages, the
    profile, one summary, jobs (each with its first bullets) and education. The file
    is checked like any other master and never replaces an existing one."""
    path = Path(path)
    if path.exists():
        raise EditError([f"{path.name} already exists; edit it in the Master tab"])
    langs = Languages.model_validate(data.get("languages") or {}).codes
    ed = MasterEditor(path, path.parent / "master-history")
    doc = CommentedMap()
    lang_node = CommentedMap([("primary", langs[0]), ("secondary", langs[1] if len(langs) > 1 else None)])
    lang_node.fa.set_flow_style()
    doc["languages"] = lang_node
    doc.yaml_set_start_comment("Your master resume: every CV is picked from what is written here.\n"
                               "Edit it in the app's Master tab, or here (the app checks it on every run).")
    p = data.get("profile") or {}
    profile = CommentedMap([("name", str(p.get("name") or "").strip())])
    headline = _lang_values(p.get("headline") or {})
    if headline:
        profile["headline"] = _texts(headline, langs)
    for key in ("email", "phone", "location"):
        if str(p.get(key) or "").strip():
            profile[key] = _q(str(p[key]).strip())
    links = CommentedSeq()
    for link in p.get("links") or []:
        if str(link.get("url") or "").strip():
            node = CommentedMap([("label", _q(str(link.get("label") or link["url"]).strip())),
                                 ("url", str(link["url"]).strip())])
            node.fa.set_flow_style()
            links.append(node)
    if links:
        profile["links"] = links
    doc["profile"] = profile
    doc["summaries"] = CommentedSeq()
    doc["jobs"] = CommentedSeq()
    s = data.get("summary") or {}
    texts = _lang_values(s)
    errors = [] if profile["name"] else ["your name is missing"]
    if texts.get(langs[0]):
        node = CommentedMap([("id", "sum-main"), ("tags", _flow(_clean_tags(s.get("tags"))))])
        for code in langs:
            if code in texts:
                node[code] = _q(texts[code])
        doc["summaries"].append(node)
    else:
        errors.append(f"the summary needs its {lang_name(langs[0])} text")
    if errors:
        raise EditError(errors)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for kind, key in (("education", "education"), ("job", "jobs")):
            for entry in reversed(data.get(key) or []):  # _add_entry puts each first
                entry = dict(entry)
                bullets = [b for b in entry.pop("bullets", None) or [] if _lang_values(b).get(langs[0])]
                if bullets:
                    entry["bullet"] = bullets[0]
                ed._add_entry(doc, kind, entry, langs, write=False)
                node = doc[key][0]
                for b in bullets[1:]:
                    bt = _lang_values(b)
                    prefix = node["id"].split("-", 1)[-1]
                    bn = CommentedMap([("id", ed.new_id(doc, [], prefix, bt[langs[0]])),
                                       ("tags", _flow(_clean_tags(b.get("tags"))))])
                    for code in langs:
                        if code in bt:
                            bn[code] = _q(bt[code])
                    node["bullets"].append(bn)
        if not doc.get("education"):
            doc.pop("education", None)
        text = ed.dump(doc)
        master = parse_master(yaml.safe_load(text))
    except MasterError as e:
        raise EditError(e.errors) from None
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    tmp.replace(path)
    return master


class MasterEditor:
    def __init__(self, path: Path, history_dir: Path):
        self.path = Path(path)
        self.history = Path(history_dir)

    # ------------------------------------------------------------ io

    def load(self) -> CommentedMap:
        return _yaml().load(self.path.read_text(encoding="utf-8"))

    def dump(self, doc: CommentedMap) -> str:
        buf = io.StringIO()
        _yaml().dump(doc, buf)
        return _space_braces(buf.getvalue())

    def commit(self, doc: CommentedMap) -> Master:
        """Validate the changed document and write it, keeping the old file in history."""
        text = self.dump(doc)
        try:
            master = parse_master(yaml.safe_load(text))
        except MasterError as e:
            raise EditError(e.errors) from None
        self.history.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns():020d}"
        shutil.copy2(self.path, self.history / f"master-{stamp}.yaml")
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="")
        tmp.replace(self.path)
        for old in sorted(self.history.glob("master-*.yaml"))[:-HISTORY_KEEP]:
            old.unlink()
        return master

    def history_count(self) -> int:
        return len(list(self.history.glob("master-*.yaml"))) if self.history.exists() else 0

    def undo(self) -> Master:
        """Restore the version before the last change made here."""
        versions = sorted(self.history.glob("master-*.yaml")) if self.history.exists() else []
        if not versions:
            raise EditError(["nothing to undo"])
        last = versions[-1]
        master = parse_master(yaml.safe_load(last.read_text(encoding="utf-8")))
        shutil.copy2(last, self.path)
        last.unlink()
        return master

    # ------------------------------------------------------------ lookup

    def _locate(self, doc: CommentedMap, id_: str) -> tuple[CommentedSeq, int, str, Optional[str]]:
        """(list holding the item, index, kind, owner id) for any item with an id."""
        def scan(seq: Any, kind: str, owner: Optional[str] = None):
            for i, node in enumerate(seq or []):
                if isinstance(node, dict) and node.get("id") == id_:
                    return seq, i, kind, owner
            return None

        hits = [scan(doc.get("summaries"), "summary"),
                scan((doc.get("profile") or {}).get("headline_alts"), "headline")]
        for key, kind in (("jobs", "job"), ("projects", "project"), ("education", "education")):
            hits.append(scan(doc.get(key), kind))
            for entry in doc.get(key) or []:
                hits += [scan(entry.get("bullets"), "bullet", entry.get("id")),
                         scan(entry.get("description_alts"), "description", entry.get("id"))]
        for tier in TIERS:
            hits.append(scan((doc.get("skills") or {}).get(tier), "skill_category", tier))
        found = [h for h in hits if h is not None and h[0] is not None]
        if not found:
            raise EditError([f"'{id_}' is not in the master"])
        return found[-1]

    def _entry(self, doc: CommentedMap, owner: str) -> tuple[CommentedMap, str]:
        for key, kind in (("jobs", "job"), ("projects", "project"), ("education", "education")):
            for entry in doc.get(key) or []:
                if entry.get("id") == owner:
                    return entry, kind
        raise EditError([f"'{owner}' is not a job, project or education entry"])

    def all_ids(self, doc: CommentedMap) -> set[str]:
        ids: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if isinstance(node.get("id"), str):
                    ids.add(node["id"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(doc)
        return ids

    def referrers(self, doc: CommentedMap, id_: str) -> list[str]:
        """Ids of items whose alt_of points at id_ (they would break if it went away)."""
        out: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                alt = node.get("alt_of")
                alts = [alt] if isinstance(alt, str) else list(alt or [])
                if id_ in alts and node.get("id") != id_:
                    out.append(str(node.get("id")))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(doc)
        return out

    def new_id(self, doc: CommentedMap, siblings: list[Any], fallback: str, text: str = "") -> str:
        """A fresh id in the style of the siblings: their common prefix, a word from
        the text and a number (e.g. brd-pipeline-08)."""
        sib = [str(s.get("id")) for s in siblings if isinstance(s, dict) and s.get("id")]
        prefix = sib[0].split("-")[0] if sib else fallback
        word = next((w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3), "item")
        taken = self.all_ids(doc)
        n = len(sib) + 1
        while f"{prefix}-{word}-{n:02d}" in taken:
            n += 1
        return f"{prefix}-{word}-{n:02d}"

    # ------------------------------------------------------------ operations

    def update(self, id_: str, changes: dict[str, Any]) -> Master:
        """Change an item's text in one or more languages (keys: language codes), tags
        or alt_of; a skill category's name, items, required or min_items; a job's,
        project's or education entry's own fields (title, dates, description...)."""
        doc = self.load()
        langs = doc_languages(doc).codes
        seq, i, kind, _ = self._locate(doc, id_)
        node = seq[i]
        for key, value in changes.items():
            if key in LANGUAGES and kind in _TEXT_KINDS:
                if str(value or "").strip():
                    node[key] = _q(str(value).strip())
                elif key == langs[0]:
                    raise EditError([f"the {lang_name(key)} text can't be empty"])
                else:
                    node.pop(key, None)
            elif key in _ENTRY_FIELDS and kind in _ENTRY_FIELDS[key][0]:
                self._set_entry_field(node, key, value, langs)
            elif key == "tags" and kind in (*_TEXT_KINDS, "project"):
                tags = _clean_tags(value)
                if not tags and kind != "headline":
                    raise EditError(["give it at least one tag"])
                node["tags"] = _flow(tags)
            elif key == "only_for_tags" and kind == "project":
                tags = _clean_tags(value)
                if tags:
                    node["only_for_tags"] = _flow(tags)
                else:
                    node.pop("only_for_tags", None)
            elif key == "alt_of" and kind in ("bullet", "summary", "skill_category"):
                alts = _clean_tags(value)
                if alts:
                    node["alt_of"] = _flow(alts) if len(alts) > 1 or kind == "skill_category" else alts[0]
                else:
                    node.pop("alt_of", None)
            elif key == "category" and kind == "skill_category":
                node["category"] = _texts(value, langs, node.get("category"))
            elif key == "items" and kind == "skill_category":
                items = _skill_items(value, langs)
                if not items:
                    raise EditError(["a skill line needs at least one item"])
                node["items"] = items
            elif key in ("required", "min_items") and kind == "skill_category":
                if key == "required":
                    if value:
                        node["required"] = True
                    else:
                        node.pop("required", None)
                elif value in (None, "", 0):
                    node.pop("min_items", None)
                else:
                    node["min_items"] = int(value)
            else:
                raise EditError([f"can't change '{key}' of a {kind.replace('_', ' ')}"])
        return self.commit(doc)

    def _set_entry_field(self, node: CommentedMap, key: str, value: Any, langs: list[str]) -> None:
        """One field of a job, project or education entry (from the entry form)."""
        how = _ENTRY_FIELDS[key][1]
        empty = value is None or (isinstance(value, str) and not value.strip()) or (
            isinstance(value, dict) and not _lang_values(value))
        if how.endswith("?") and empty:
            node.pop(key, None)
            return
        how = how.rstrip("?")
        if how == "bool":
            if value:
                node[key] = True
            else:
                node.pop(key, None)
        elif how == "date":
            node[key] = str(value).strip()
        elif how == "plain":
            node[key] = _q(str(value).strip())
        else:  # "texts" (always per language) or "text" (a string, or per language)
            if isinstance(value, dict):
                texts = _lang_values(value)
                _need_primary(texts, langs, key)
                node[key] = _texts(texts, langs, node.get(key))
            elif how == "texts":
                raise EditError([f"{key} needs a text per language"])
            elif str(value).strip():
                node[key] = _q(str(value).strip())
            else:
                raise EditError([f"{key} can't be empty"])

    def add(self, kind: str, parent: Optional[str], data: dict[str, Any],
            after: Optional[str] = None) -> tuple[Master, str]:
        """Add a bullet (parent: job, project or education id), a summary, a description
        alternative (parent: job or project id), a headline option, a skill category
        (parent: its tier), or a job, project or education entry (data: its fields; a
        job also needs its first bullet, as data["bullet"]). Returns the new master and
        the new id."""
        doc = self.load()
        langs = doc_languages(doc).codes
        if kind in _ENTRY_KINDS:
            return self._add_entry(doc, kind, data, langs)
        texts = _lang_values(data)
        first = texts.get(langs[0], "")
        if kind == "bullet":
            entry, _ = self._entry(doc, parent or "")
            seq = entry.setdefault("bullets", CommentedSeq())
            fallback = (parent or "item").split("-")[-1]
        elif kind == "description":
            entry, _ = self._entry(doc, parent or "")
            seq = entry.setdefault("description_alts", CommentedSeq())
            fallback = (parent or "desc").split("-")[-1] + "-desc"
        elif kind == "summary":
            seq, fallback = doc.setdefault("summaries", CommentedSeq()), "sum"
        elif kind == "headline":
            profile = doc.setdefault("profile", CommentedMap())
            seq, fallback = profile.setdefault("headline_alts", CommentedSeq()), "hl"
        elif kind == "skill_category":
            if parent not in TIERS:
                raise EditError([f"'{parent}' is not a skill tier ({', '.join(TIERS)})"])
            skills = doc.setdefault("skills", CommentedMap())
            seq, fallback = skills.setdefault(parent, CommentedSeq()), "skills"
        else:
            raise EditError([f"can't add a {kind}"])

        category = _lang_values(data.get("category") or {}) if isinstance(data.get("category"), dict) else {}
        new_id = str(data.get("id") or "").strip() or self.new_id(doc, list(seq), fallback, first or str(
            category.get(langs[0], "")))
        if new_id in self.all_ids(doc):
            raise EditError([f"id '{new_id}' is already used"])
        node = CommentedMap([("id", new_id)])
        if kind == "skill_category":
            node["category"] = _texts(category or texts, langs)
            node["items"] = _skill_items(data.get("items"), langs)
        else:
            _need_primary(texts, langs)
            if data.get("alt_of"):
                node["alt_of"] = _clean_tags(data["alt_of"])[0] if len(_clean_tags(data["alt_of"])) == 1 \
                    else _flow(_clean_tags(data["alt_of"]))
            node["tags"] = _flow(_clean_tags(data.get("tags")))
            for code in [*langs, *(c for c in texts if c not in langs)]:
                if code in texts:
                    node[code] = _q(texts[code])
        if kind == "headline":
            node.fa.set_flow_style()
        pos = len(seq)
        if after:
            ids = [str(x.get("id")) for x in seq]
            if after in ids:
                pos = ids.index(after) + 1
        seq.insert(pos, node)
        return self.commit(doc), new_id

    def _add_entry(self, doc: CommentedMap, kind: str, data: dict[str, Any],
                   langs: list[str], write: bool = True) -> tuple[Optional[Master], str]:
        key = {"job": "jobs", "project": "projects", "education": "education"}[kind]
        seq = doc.setdefault(key, CommentedSeq())
        label = data.get({"job": "company", "project": "name", "education": "institution"}[kind]) or ""
        if isinstance(label, dict):
            label = _lang_values(label).get(langs[0], "")
        if kind == "job" and not label:
            label = _lang_values(data.get("title") or {}).get(langs[0], "")
        stem = {"job": "job", "project": "proj", "education": "edu"}[kind]
        base = f"{stem}-{folder_slug(str(label), 24)}"
        new_id, n, taken = base, 2, self.all_ids(doc)
        while new_id in taken:
            new_id, n = f"{base}-{n}", n + 1
        node = CommentedMap([("id", new_id)])
        order = {"job": ("title", "company", "location", "start", "end", "description"),
                 "project": ("name", "start", "end", "description"),
                 "education": ("degree", "institution", "note", "start", "end", "expected")}[kind]
        for f in order:
            if f in data:
                self._set_entry_field(node, f, data[f], langs)
        need = {"job": ("title", "start"), "project": ("name",), "education": ("degree", "institution", "start")}[kind]
        missing = [f for f in need if f not in node]
        if missing:
            raise EditError([f"the new {kind} needs: {', '.join(missing)}"])
        if kind == "project":
            node.insert(2, "tags", _flow(_clean_tags(data.get("tags"))))
        bullet = data.get("bullet")
        if bullet:
            texts = _lang_values(bullet)
            _need_primary(texts, langs, "bullet")
            prefix = new_id.split("-", 1)[-1] or stem
            b = CommentedMap([("id", self.new_id(doc, [], prefix, texts[langs[0]])),
                              ("tags", _flow(_clean_tags(bullet.get("tags"))))])
            for code in langs:
                if code in texts:
                    b[code] = _q(texts[code])
            node["bullets"] = CommentedSeq([b])
        seq.insert(0, node)  # newest first, as master files usually are (the CV sorts by date)
        return (self.commit(doc) if write else None), new_id

    def set_languages(self, primary: str, secondary: Optional[str]) -> Master:
        """Change the CV's languages. Texts are never translated or removed: items
        without the new secondary language are left out of CVs in it until their text
        is added, and texts in a language no longer used stay in the file."""
        from pydantic import ValidationError

        from .master import load_master, missing_texts
        try:
            langs = Languages(primary=primary, secondary=secondary)
        except ValidationError as e:
            raise EditError([x["msg"].removeprefix("Value error, ") for x in e.errors()]) from None
        lost = missing_texts(load_master(self.path), langs.primary)
        if lost:
            raise EditError([f"{len(lost)} text{'s have' if len(lost) > 1 else ' has'} no {lang_name(langs.primary)} "
                             f"yet, and the main language must be on every item. Add them first (in the Master "
                             f"tab, switch to {langs.primary.upper()}: items without it are marked), or keep "
                             f"{lang_name(langs.primary)} as the second language.",
                             *(f"  {what}" for _, what in lost[:8]), *(["  ..."] if len(lost) > 8 else [])])
        doc = self.load()
        node = CommentedMap([("primary", langs.primary), ("secondary", langs.secondary)])
        node.fa.set_flow_style()
        if "languages" in doc:
            doc["languages"] = node
        else:
            doc.insert(0, "languages", node, comment="the CV's languages; every text is keyed by these codes")
        return self.commit(doc)

    def delete(self, id_: str) -> Master:
        doc = self.load()
        seq, i, kind, _ = self._locate(doc, id_)
        if kind in ("job", "project", "education"):
            raise EditError([f"deleting a whole {kind} isn't done here; edit master.yaml for that"])
        refs = self.referrers(doc, id_)
        if refs:
            raise EditError([f"'{id_}' has alternatives that point at it ({', '.join(refs)}); "
                             "delete or change those first"])
        del seq[i]
        return self.commit(doc)

    def move(self, id_: str, delta: int) -> Master:
        """Move an item up (delta < 0) or down within its own list."""
        doc = self.load()
        seq, i, _, _ = self._locate(doc, id_)
        j = max(0, min(len(seq) - 1, i + delta))
        if j == i:
            return parse_master(yaml.safe_load(self.dump(doc)))
        node = seq.pop(i)
        seq.insert(j, node)
        return self.commit(doc)
