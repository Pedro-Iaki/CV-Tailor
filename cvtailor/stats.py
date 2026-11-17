"""Callback-rate computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .db import age_days

LOW_SAMPLE = 10


@dataclass
class Group:
    key: str
    n: int = 0
    screen: int = 0
    interview: int = 0

    @property
    def screen_rate(self) -> float:
        return self.screen / self.n if self.n else 0.0

    @property
    def interview_rate(self) -> float:
        return self.interview / self.n if self.n else 0.0

    @property
    def low_sample(self) -> bool:
        return self.n < LOW_SAMPLE

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        if row["furthest_stage"] >= 1:
            self.screen += 1
        if row["furthest_stage"] >= 2:
            self.interview += 1


@dataclass
class Stats:
    overall: Group
    by_variant: list[Group] = field(default_factory=list)
    by_lang: list[Group] = field(default_factory=list)
    by_route: list[Group] = field(default_factory=list)
    pending: int = 0
    discarded: int = 0


def is_pending(row: dict[str, Any], min_age: int, now: Optional[datetime] = None) -> bool:
    return row["status"] == "applied" and age_days(row["created_at"], now) < min_age


def counted(rows: list[dict[str, Any]], min_age: int, now: Optional[datetime] = None) -> tuple[list[dict], int, int]:
    out, pending, discarded = [], 0, 0
    for r in rows:
        if r["status"] == "discarded":
            discarded += 1
        elif is_pending(r, min_age, now):
            pending += 1
        else:
            out.append(r)
    return out, pending, discarded


def _group(rows: list[dict], keyfn: Callable[[dict], str]) -> list[Group]:
    groups: dict[str, Group] = {}
    for r in rows:
        k = keyfn(r)
        groups.setdefault(k, Group(k)).add(r)
    return sorted(groups.values(), key=lambda g: (-g.n, g.key))


def compute_stats(rows: list[dict[str, Any]], min_age: int = 14, now: Optional[datetime] = None) -> Stats:
    rows, pending, discarded = counted(rows, min_age, now)
    overall = Group("overall")
    for r in rows:
        overall.add(r)
    return Stats(
        overall=overall,
        by_variant=_group(rows, lambda r: r.get("variant_name") or "(one-off)"),
        by_lang=_group(rows, lambda r: r["lang"]),
        by_route=_group(rows, lambda r: r["route"]),
        pending=pending,
        discarded=discarded,
    )


@dataclass
class VariantUsage:
    applications: int  # rows that used the variant, excluding discarded
    screen_plus: int   # counted rows that reached screen or further
    n: int             # counted rows (not discarded, not pending)


def variant_usage(rows: list[dict[str, Any]], name: str, min_age: int = 14,
                  now: Optional[datetime] = None) -> VariantUsage:
    mine = [r for r in rows if r["variant_name"] == name]
    done, _, _ = counted(mine, min_age, now)
    return VariantUsage(
        applications=sum(1 for r in mine if r["status"] != "discarded"),
        screen_plus=sum(1 for r in done if r["furthest_stage"] >= 1),
        n=len(done),
    )


def fmt_rate(count: int, n: int) -> str:
    if not n:
        return "-"
    return f"{100 * count / n:.0f}% ({count}/{n})"
