"""Deterministic job-tag extraction: master tag vocabulary + aliases, no LLM."""

from __future__ import annotations

import re
from functools import lru_cache

from .schema import Master
from .master import tag_vocabulary

# Characters that count as part of a word: alphanumerics plus + # (so c#, c++ work).
# '.' and '/' join words only when followed by an alphanumeric (node.js, ci/cd), so a
# trailing period or slash still ends a word ("... with Spark.").
_BEFORE = r"(?<![A-Za-z0-9_+#])(?<![A-Za-z0-9_][./])"
_AFTER = r"(?![A-Za-z0-9_+#])(?![./][A-Za-z0-9])"

MIN_TAG_LEN = 3


def _phrase_pattern(phrase: str) -> str:
    words = phrase.strip().split()
    return r"\s+".join(re.escape(w) for w in words)


def _tag_pattern(tag: str) -> str:
    return r"(?:-|\s+)?".join(re.escape(p) for p in tag.split("-"))


def build_patterns(vocabulary: set[str], aliases: dict[str, list[str]]) -> dict[str, re.Pattern]:
    patterns: dict[str, re.Pattern] = {}
    for tag in sorted(vocabulary):
        alts: list[str] = []
        if len(tag) >= MIN_TAG_LEN:
            alts.append(_tag_pattern(tag))
        for a in aliases.get(tag, []) or []:
            if str(a).strip():
                alts.append(_phrase_pattern(str(a)))
        if not alts:
            continue
        # Longest alternatives first so the reported phrase is the most specific one.
        alts.sort(key=len, reverse=True)
        patterns[tag] = re.compile(_BEFORE + "(?:" + "|".join(alts) + ")" + _AFTER, re.IGNORECASE)
    return patterns


def match_tags(text: str, master: Master) -> dict[str, str]:
    """Return {tag: phrase that triggered it} for every tag mentioned in text."""
    pats = _patterns_for(master)
    found: dict[str, str] = {}
    for tag, pat in pats.items():
        m = pat.search(text or "")
        if m:
            found[tag] = m.group(0)
    return dict(sorted(found.items()))


def extract_tags(text: str, master: Master) -> list[str]:
    return sorted(match_tags(text, master))


def _patterns_for(master: Master) -> dict[str, re.Pattern]:
    key = (
        tuple(sorted(tag_vocabulary(master))),
        tuple(sorted((k, tuple(v or [])) for k, v in master.tag_aliases.items())),
    )
    return _cached_patterns(key)


@lru_cache(maxsize=16)
def _cached_patterns(key: tuple) -> dict[str, re.Pattern]:
    vocab, aliases = key
    return build_patterns(set(vocab), {k: list(v) for k, v in aliases})
