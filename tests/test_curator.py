"""The Lesson Curator: the desk reviews its own record — evidence-gated
activation, contradiction retirement, and pattern lessons from aggregates."""

from system.reflection.curator import MIN_BUCKET, assess, curate
from system.reflection.memory import LessonMemory, TradeOutcome
from system.schemas import Lesson


def _row(ret, conviction=0.6, outcome="time", gem=False, moat=None):
    return {"status": "evaluated", "return_pct": ret, "conviction": conviction,
            "outcome": outcome, "hidden_gem": gem, "moat_stance": moat,
            "evaluated_on": "2026-06-24"}


def _mem_with(pending_lessons=()):
    # pending_lessons: (text, worked, cohort) anecdotes, entered PENDING.
    mem = LessonMemory()
    for text, worked, cohort in pending_lessons:
        mem.add(Lesson("confluence_swing", text, worked, "clean", cohort=cohort),
                human_reviewed=False)
    return mem


def _cohort_rows(cohort, ret, n):
    """n scored rows in a given lens cohort with a fixed return."""
    gem = cohort == "hidden-gem"
    moat = "bullish" if cohort == "moat-bullish" else None
    return [{"status": "evaluated", "return_pct": ret, "conviction": 0.6,
             "outcome": "time", "hidden_gem": gem, "moat_stance": moat,
             "symbol": f"{cohort[:2]}{i}", "evaluated_on": "2026-06-24"}
            for i in range(n)]


def test_curator_activates_a_lesson_its_cohort_backs():
    # A 'core paid' anecdote activates once the CORE cohort's realized avg return
    # is clearly positive — automatically, no human sign-off.
    mem = _mem_with([("core momentum pays", True, "core")])
    rep = curate(mem, _cohort_rows("core", ret=6.0, n=6))
    assert rep["activated"] == 1 and rep["pending"] == 0
    assert all(e.human_reviewed for e in mem.entries)


def test_curator_activates_a_cautionary_lesson_when_cohort_loses():
    # A 'gem did NOT pay' anecdote activates when the gem cohort is clearly
    # negative — the cautionary lesson is evidence-backed.
    mem = _mem_with([("gem did not pay", False, "hidden-gem")])
    rep = curate(mem, _cohort_rows("hidden-gem", ret=-3.0, n=6))
    assert rep["activated"] == 1
    assert any(not e.lesson.kind and e.human_reviewed for e in mem.entries)


def test_curator_retires_a_lesson_its_cohort_contradicts():
    # A 'gem paid' claim in a cohort that clearly LOSES is retired outright.
    mem = _mem_with([("gems pay big", True, "hidden-gem")])
    rep = curate(mem, _cohort_rows("hidden-gem", ret=-3.0, n=9))
    assert rep["retired"] >= 1
    assert not any(e.lesson.lesson == "gems pay big" for e in mem.entries)


def test_curator_waits_on_thin_cohort_evidence():
    # Under MIN_BUCKET scored calls in the cohort, neither activate nor retire.
    mem = _mem_with([("core pays", True, "core")])
    rep = curate(mem, _cohort_rows("core", ret=6.0, n=MIN_BUCKET - 1))
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
    mem = _mem_with()
    rows = [_row(7.0, gem=True)] * 5
    curate(mem, rows)
    n_after_first = len(mem.entries)
    curate(mem, rows)                                     # second pass: no dupes
    assert len(mem.entries) == n_after_first


def test_curator_replaces_reworded_pattern_lessons_instead_of_bloating():
    # The lens lesson embeds the running count/avg, so its WORDING changes every
    # run as more trades score. The curator must keep ONE current lesson per kind
    # (replacing the stale numbers), not append a near-duplicate each pass.
    mem = _mem_with()
    curate(mem, [_row(7.0, gem=True)] * 5)               # "5 scored, +7.0%"
    after_first = len(mem.entries)
    lens = [e for e in mem.entries if e.lesson.kind == "lens:hidden-gem"]
    assert len(lens) == 1

    curate(mem, [_row(7.0, gem=True)] * 8)               # now "8 scored" -> reworded
    lens = [e for e in mem.entries if e.lesson.kind == "lens:hidden-gem"]
    assert len(lens) == 1                                 # still exactly one
    assert "8 scored" in lens[0].lesson.lesson            # refreshed to new numbers
    assert len(mem.entries) == after_first                # memory did NOT grow


def test_assess_flags_inverted_conviction_buckets():
    # Aggregate calibration nets out (avg conviction ~ win rate), but the HIGH-
    # conviction half loses while the LOW-conviction half wins — the desk is most
    # wrong when it bets biggest. The bucket check must catch this.
    low = [_row(8.0, conviction=0.45)] * 8                # cautious calls: all win
    high = [_row(-5.0, conviction=0.75)] * 8              # confident calls: all lose
    rep = assess(low + high)
    inv = [l for l in rep["pattern_lessons"] if l.kind == "calibration_buckets"]
    assert inv and "INVERTED conviction" in inv[0].lesson

    # Balanced/positive calibration: no inversion lesson.
    ok = [_row(8.0, conviction=0.75)] * 8 + [_row(-5.0, conviction=0.45)] * 8
    assert not any(l.kind == "calibration_buckets"
                   for l in assess(ok)["pattern_lessons"])
