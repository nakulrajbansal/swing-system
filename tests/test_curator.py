"""The Lesson Curator: the desk reviews its own record — evidence-gated
activation, contradiction retirement, and pattern lessons from aggregates."""

from system.reflection.curator import MIN_BUCKET, assess, curate
from system.reflection.memory import LessonMemory, TradeOutcome
from system.schemas import Lesson


def _row(ret, conviction=0.6, outcome="time", gem=False, moat=None):
    return {"status": "evaluated", "return_pct": ret, "conviction": conviction,
            "outcome": outcome, "hidden_gem": gem, "moat_stance": moat,
            "evaluated_on": "2026-06-24"}


def _mem_with(setup="confluence_swing", wins=6, losses=2, pending_lessons=()):
    mem = LessonMemory()
    for i in range(wins):
        mem.record_outcome(TradeOutcome(setup, f"W{i}", 0.6, 5.0, "time", "2026-06-20"))
    for i in range(losses):
        mem.record_outcome(TradeOutcome(setup, f"L{i}", 0.6, -4.0, "stop", "2026-06-20"))
    for text, worked in pending_lessons:
        mem.add(Lesson(setup, text, worked, "clean"), human_reviewed=False)
    return mem


def test_curator_activates_evidence_backed_anecdotes():
    mem = _mem_with(wins=6, losses=2,                     # 75% win rate
                    pending_lessons=[("momentum paid", True)])
    rep = curate(mem, [_row(5.0)] * 6)
    assert rep["activated"] == 1 and rep["pending"] == 0
    assert all(e.human_reviewed for e in mem.entries)


def test_curator_retires_contradicted_anecdotes():
    mem = _mem_with(wins=1, losses=9,                     # 10% win rate, n=10
                    pending_lessons=[("this setup pays", True)])
    rep = curate(mem, [_row(-4.0)] * 6)
    assert rep["retired"] == 1
    assert not any(e.lesson.lesson == "this setup pays" for e in mem.entries)


def test_curator_waits_on_thin_evidence():
    mem = _mem_with(wins=2, losses=1,                     # n=3 < MIN_BUCKET
                    pending_lessons=[("anecdote", True)])
    rep = curate(mem, [_row(5.0)] * 3)
    assert rep["activated"] == 0 and rep["retired"] == 0 and rep["pending"] == 1


def test_assess_writes_calibration_and_stop_mix_patterns():
    # High stated conviction (0.8 avg) vs 33% realized hit rate, mostly stops.
    rows = ([_row(-5.0, conviction=0.8, outcome="stop")] * 4
            + [_row(6.0, conviction=0.8, outcome="target")] * 2)
    rep = assess(rows)
    texts = [l.lesson for l in rep["pattern_lessons"]]
    assert rep["calibration"] and "ABOVE the realized win rate" in rep["calibration"]
    assert any("STOPS" in t for t in texts)               # exit-mix pattern


def test_assess_grades_the_gem_lens_and_gates_on_n():
    gems_good = [_row(7.0, gem=True)] * 4 + [_row(-2.0, gem=True)]
    rep = assess(gems_good)                               # 80% of 5 gems
    assert any("hidden-gem picks are PAYING" in l.lesson
               for l in rep["pattern_lessons"])
    rep_thin = assess(gems_good[:MIN_BUCKET - 1])         # n=4: silent
    assert not any("hidden-gem" in l.lesson for l in rep_thin["pattern_lessons"])


def test_curator_never_duplicates_pattern_lessons():
    mem = _mem_with(wins=0, losses=0)
    rows = [_row(7.0, gem=True)] * 5
    curate(mem, rows)
    n_after_first = len(mem.entries)
    curate(mem, rows)                                     # second pass: no dupes
    assert len(mem.entries) == n_after_first
