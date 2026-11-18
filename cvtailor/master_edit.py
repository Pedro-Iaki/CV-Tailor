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

from .master import MasterError, parse_master
from .schema import TIERS, Master

HISTORY_KEEP = 30
_TEXT_KINDS = ("summary", "bullet", "description", "headline")


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


def _bilingual(v: Any) -> Any:
    """{en, pt} for a bilingual value, written as a flow map like the rest of the file."""
    if isinstance(v, dict):
        m = CommentedMap([("en", _q(v.get("en", ""))), ("pt", _q(v.get("pt", "")))])
        m.fa.set_flow_style()
        return m
    return _q(v)


def _clean_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = re.split(r"[,\s]+", tags)
    return [t.strip().lower() for t in (tags or []) if str(t).strip()]


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
        """Change the text (en, pt), tags or alt_of of an item; a skill category's name,
        items, required or min_items; an entry's default description or a project's name."""
        doc = self.load()
        seq, i, kind, _ = self._locate(doc, id_)
        node = seq[i]
        for key, value in changes.items():
            if key in ("en", "pt") and kind in _TEXT_KINDS:
                if not str(value).strip():
                    raise EditError([f"{key.upper()} text can't be empty"])
                node[key] = _q(str(value).strip())
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
                node["category"] = _bilingual(value)
            elif key == "items" and kind == "skill_category":
                items = CommentedSeq()
                for it in value or []:
                    if isinstance(it, dict) and it.get("en", "").strip():
                        same = it.get("pt", "").strip() in ("", it["en"].strip())
                        items.append(_q(it["en"].strip()) if same else _bilingual(
                            {"en": it["en"].strip(), "pt": it["pt"].strip()}))
                    elif isinstance(it, str) and it.strip():
                        items.append(_q(it.strip()))
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
            elif key == "description" and kind in ("job", "project"):
                if value is None:
                    node.pop("description", None)
                else:
                    node["description"] = _bilingual(value)
            elif key == "name" and kind == "project":
                node["name"] = _bilingual(value)
            else:
                raise EditError([f"can't change '{key}' of a {kind.replace('_', ' ')}"])
        return self.commit(doc)

    def add(self, kind: str, parent: Optional[str], data: dict[str, Any],
            after: Optional[str] = None) -> tuple[Master, str]:
        """Add a bullet (parent: job, project or education id), a summary, a description
        alternative (parent: job or project id), a headline option, or a skill category
        (parent: its tier). Returns the new master and the new id."""
        doc = self.load()
        en, pt = str(data.get("en", "")).strip(), str(data.get("pt", "")).strip()
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

        new_id = str(data.get("id") or "").strip() or self.new_id(doc, list(seq), fallback, en or str(
            (data.get("category") or {}).get("en", "")))
        if new_id in self.all_ids(doc):
            raise EditError([f"id '{new_id}' is already used"])
        node = CommentedMap([("id", new_id)])
        if kind == "skill_category":
            node["category"] = _bilingual(data.get("category") or {"en": en, "pt": pt})
            node["items"] = CommentedSeq([_q(x) for x in data.get("items") or []])
        else:
            if data.get("alt_of"):
                node["alt_of"] = _clean_tags(data["alt_of"])[0] if len(_clean_tags(data["alt_of"])) == 1 \
                    else _flow(_clean_tags(data["alt_of"]))
            node["tags"] = _flow(_clean_tags(data.get("tags")))
            node["en"], node["pt"] = _q(en), _q(pt)
        if kind == "headline":
            node.fa.set_flow_style()
        pos = len(seq)
        if after:
            ids = [str(x.get("id")) for x in seq]
            if after in ids:
                pos = ids.index(after) + 1
        seq.insert(pos, node)
        return self.commit(doc), new_id

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
