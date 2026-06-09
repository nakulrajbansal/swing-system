"""Persistent learning store: lessons + trade outcomes that survive across runs.

This is the file-IO wrapper around the pure :class:`LessonMemory` (the system
layer stays IO-free). It loads accumulated lessons at the start of a run so the
agents can recall them, and saves new lessons/outcomes at the end — so the desk
genuinely gets smarter over time. Stored under the user's home, never committed:

    ~/.swing_system/learning.json
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import CONFIG_DIR
from system.reflection.memory import LessonMemory

LEARNING_PATH = CONFIG_DIR / "learning.json"


def load_memory(path: Path | None = None) -> LessonMemory:
    p = path or LEARNING_PATH
    if p.exists():
        try:
            return LessonMemory.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return LessonMemory()


def save_memory(mem: LessonMemory, path: Path | None = None) -> Path:
    p = path or LEARNING_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mem.to_dict(), indent=2, default=str), encoding="utf-8")
    return p


def summarize(mem: LessonMemory) -> str:
    """Human-readable digest for the GUI 'Lessons' view."""
    lines = []
    n_les = len(mem.entries)
    n_out = len(mem.outcomes)
    reviewed = sum(1 for e in mem.entries if e.human_reviewed)
    lines.append(f"Learning memory: {n_les} lesson(s) ({reviewed} active), "
                 f"{n_out} recorded trade outcome(s).")
    setups = sorted({o.setup_type for o in mem.outcomes})
    for st in setups:
        s = mem.setup_stats(st)
        if s.get("count"):
            lines.append(f"  [{st}] {s['count']} trades, win rate {s.get('win_rate_pct', 0):.0f}%, "
                         f"avg return {s.get('avg_return_pct', 0):+.1f}%")
    cal = mem.calibration()
    if cal:
        lines.append("  calibration (predicted conviction -> realized win rate):")
        for b in cal:
            lines.append(f"    {b['conviction']}: {b['win_rate_pct']:.0f}% over {b['n']} trades")
    if mem.entries:
        lines.append("  recent lessons:")
        for e in mem.entries[-8:]:
            mark = "*" if e.human_reviewed else " "
            l = e.lesson
            lines.append(f"   [{mark}] ({l.setup_type}) {l.symbol} {l.as_of}: {l.lesson}")
        lines.append("   (* = active / human-reviewed)")
    return "\n".join(lines)
