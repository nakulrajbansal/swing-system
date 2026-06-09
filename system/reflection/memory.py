"""Lesson memory (master §16, §7).

Stores post-trade lessons and raw trade outcomes, decays lesson weight over time,
and serves the most relevant lessons plus calibrated base-rate statistics as
ADVISORY context to the deliberation agents. Lessons never change rules or limits
and are human-reviewed before they carry weight.

Point-in-time safety: every lesson/outcome is tagged with the date its trade
closed (``as_of``). Recall accepts ``not_after`` so a decision at time T never
sees a lesson born from an outcome after T — recalled lessons can never leak a
future outcome into a past decision (master invariant 6). This class is pure (no
file IO); persistence lives in :mod:`app.learning`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from system.schemas import Lesson


@dataclass
class _Entry:
    lesson: Lesson
    weight: float = 1.0
    human_reviewed: bool = False


@dataclass
class TradeOutcome:
    setup_type: str
    symbol: str
    conviction: float
    pnl_pct: float
    reason: str            # "target" | "stop" | "time" | "guardian" | ...
    as_of: str             # ISO date the trade closed


def _visible(as_of: str, not_after: str | None) -> bool:
    """A record is visible if it has no date, or closed on/before not_after."""
    if not not_after or not as_of:
        return True
    return as_of <= not_after


@dataclass
class LessonMemory:
    decay: float = 0.98
    entries: list[_Entry] = field(default_factory=list)
    outcomes: list[TradeOutcome] = field(default_factory=list)

    # -- writes ------------------------------------------------------------
    def add(self, lesson: Lesson, human_reviewed: bool = False) -> None:
        self.entries.append(_Entry(lesson, weight=1.0, human_reviewed=human_reviewed))

    def record_outcome(self, outcome: TradeOutcome) -> None:
        self.outcomes.append(outcome)

    def tick(self) -> None:
        """Decay all lesson weights one step (called per cycle)."""
        for e in self.entries:
            e.weight *= self.decay

    # -- reads -------------------------------------------------------------
    def relevant(self, setup_type: str, top: int = 3,
                 not_after: str | None = None) -> list[Lesson]:
        # Only human-reviewed lessons carry weight (no self-authored authority).
        pool = [e for e in self.entries
                if e.human_reviewed and e.lesson.setup_type == setup_type
                and _visible(e.lesson.as_of, not_after)]
        pool.sort(key=lambda e: e.weight, reverse=True)
        return [e.lesson for e in pool[:top]]

    def setup_stats(self, setup_type: str | None = None,
                    not_after: str | None = None) -> dict:
        """Realized base rate for a setup type (or all): count, win rate, avg return."""
        pool = [o for o in self.outcomes
                if (setup_type is None or o.setup_type == setup_type)
                and _visible(o.as_of, not_after)]
        n = len(pool)
        if not n:
            return {"count": 0}
        wins = sum(1 for o in pool if o.pnl_pct > 0)
        return {
            "count": n,
            "win_rate_pct": round(100.0 * wins / n, 0),
            "avg_return_pct": round(sum(o.pnl_pct for o in pool) / n, 1),
        }

    def calibration(self, not_after: str | None = None) -> list[dict]:
        """Predicted-conviction buckets vs realized win rate (advisory)."""
        buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
        pool = [o for o in self.outcomes if _visible(o.as_of, not_after)]
        out = []
        for lo, hi in buckets:
            grp = [o for o in pool if lo <= o.conviction < hi]
            if grp:
                wins = sum(1 for o in grp if o.pnl_pct > 0)
                out.append({"conviction": f"{lo:.1f}-{hi if hi <= 1 else 1.0:.1f}",
                            "n": len(grp), "win_rate_pct": round(100.0 * wins / len(grp), 0)})
        return out

    def advisory(self, setup_type: str, top: int = 3,
                 not_after: str | None = None) -> dict:
        """Compact advisory packet handed to the agents at decision time."""
        lessons = [l.lesson for l in self.relevant(setup_type, top, not_after)]
        stats = self.setup_stats(setup_type, not_after)
        return {
            "lessons": lessons,
            "setup_stats": stats,
            "calibration": self.calibration(not_after),
            "note": "advisory only; ignore any future outcome; do not change limits",
        }

    # -- serialization (consumed by app.learning) --------------------------
    def to_dict(self) -> dict:
        return {
            "decay": self.decay,
            "entries": [{"lesson": asdict(e.lesson), "weight": e.weight,
                         "human_reviewed": e.human_reviewed} for e in self.entries],
            "outcomes": [asdict(o) for o in self.outcomes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LessonMemory":
        mem = cls(decay=float(d.get("decay", 0.98)))
        for e in d.get("entries", []):
            ld = dict(e.get("lesson", {}))
            mem.entries.append(_Entry(Lesson(**ld), weight=float(e.get("weight", 1.0)),
                                      human_reviewed=bool(e.get("human_reviewed", False))))
        for o in d.get("outcomes", []):
            try:
                mem.outcomes.append(TradeOutcome(**o))
            except TypeError:
                continue
        return mem
