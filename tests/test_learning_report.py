"""The Learning-tab synthesis: report structure, journal, and text rendering."""

from app import learning_report as LR
from system.reflection.memory import LessonMemory, TradeOutcome
from system.schemas import Lesson


def _ledger(n=20):
    rows = []
    months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    secs = ["Tech", "Health", "Energy", "Financials"]
    for i in range(n):
        win = i % 2 == 0
        rows.append({"status": "evaluated", "symbol": f"S{i}",
                     "return_pct": 6.0 if win else -3.0, "conviction": 0.6,
                     "evaluated_on": months[i % 4] + "-15", "date": months[i % 4] + "-01",
                     "sector": secs[i % 4], "hidden_gem": i % 5 == 0,
                     "moat_stance": "bullish" if i % 3 == 0 else None})
    return rows


def _mem():
    mem = LessonMemory()
    mem.add(Lesson("confluence_swing", "hidden-gem picks are PAYING", True,
                   "curated", kind="lens:hidden-gem"), human_reviewed=True)
    mem.add(Lesson("confluence_swing", "executed AAA paid +6.0%", True, "clean",
                   symbol="AAA", as_of="2026-03-15"), human_reviewed=True)
    mem.add(Lesson("confluence_swing", "pending idea", True, "clean"),
            human_reviewed=False)
    for i in range(20):
        mem.record_outcome(TradeOutcome("confluence_swing", f"S{i}", 0.6,
                                        6.0 if i % 2 == 0 else -3.0, "time", "2026-03-15"))
    return mem


def test_build_report_has_all_sections():
    rep = LR.build_report(led=_ledger(), mem=_mem())
    for key in ("headline", "readiness", "parameters", "learnings",
                "strategy", "evolution"):
        assert key in rep
    # The four adaptive knobs are each named with the agents they move.
    names = {p["param"] for p in rep["parameters"]}
    assert any("calibration" in n.lower() for n in names)
    assert any("throttle" in n.lower() for n in names)
    assert all(p["affects"] for p in rep["parameters"])
    # Headline carries the readiness recommendation.
    assert rep["headline"]["n_scored"] == 20
    assert "stage" in rep["headline"] and "verdict" in rep["headline"]


def test_report_maps_lessons_into_patterns_and_anecdotes():
    rep = LR.build_report(led=_ledger(), mem=_mem())
    L = rep["learnings"]
    assert any(p["kind"] == "lens:hidden-gem" for p in L["patterns"])
    assert L["pending"] == 1                              # the unreviewed lesson
    assert L["cohorts"]["hidden_gem"]["n"] >= 1


def test_evolution_has_monthly_cohorts():
    rep = LR.build_report(led=_ledger(), mem=_mem())
    months = [m["month"] for m in rep["evolution"]["monthly"]]
    assert "2026-01" in months and "2026-04" in months


def test_journal_append_is_idempotent_per_day(tmp_path):
    p = tmp_path / "journal.json"
    rep = LR.build_report(led=_ledger(), mem=_mem())
    LR.append_snapshot(rep, path=p)
    LR.append_snapshot(rep, path=p)                       # same day -> replaces
    journal = LR.load_journal(p)
    assert len(journal) == 1
    assert journal[0]["n_scored"] == 20 and "readiness_score" in journal[0]


def test_render_text_is_human_readable():
    txt = LR.render_text(LR.build_report(led=_ledger(), mem=_mem()))
    assert "DEPLOYMENT READINESS" in txt
    assert "WHAT THE DESK IS ADJUSTING" in txt
    assert "CURRENT STRATEGY" in txt
    assert "STRATEGY EVOLUTION" in txt
