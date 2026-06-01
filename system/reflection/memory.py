"""Lesson memory (master §16, §7).

Stores post-trade lessons, decays them over time, and serves the most relevant
ones as ADVISORY context to the Hypothesis agent. Lessons never change rules or
limits and are human-reviewed before they carry weight. Crucially, recalled
lessons must never leak future outcomes into a decision — they are tagged with
the cycle they came from and are advisory only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from system.schemas import Lesson


@dataclass
class _Entry:
    lesson: Lesson
    weight: float
    human_reviewed: bool = False


@dataclass
class LessonMemory:
    decay: float = 0.98
    entries: list[_Entry] = field(default_factory=list)

    def add(self, lesson: Lesson, human_reviewed: bool = False):
        self.entries.append(_Entry(lesson, weight=1.0, human_reviewed=human_reviewed))

    def tick(self):
        """Decay all weights one step (called per cycle)."""
        for e in self.entries:
            e.weight *= self.decay

    def relevant(self, setup_type: str, top: int = 3) -> list[Lesson]:
        # Only human-reviewed lessons carry weight (no self-authored authority).
        pool = [e for e in self.entries
                if e.human_reviewed and e.lesson.setup_type == setup_type]
        pool.sort(key=lambda e: e.weight, reverse=True)
        return [e.lesson for e in pool[:top]]
