"""Cross-run learning: reflection -> lessons + outcomes, PIT-safe recall, persistence."""

from system.agents.llm_client import MockLLMClient
from system.agents.meta import ReflectionAgent
from system.config import SystemConfig
from system.reflection.memory import LessonMemory, TradeOutcome
from system.schemas import Lesson

M = SystemConfig().models


def _seed() -> LessonMemory:
    mem = LessonMemory()
    # Two wins, one loss, dated across time.
    mem.record_outcome(TradeOutcome("confluence_swing", "AAA", 0.65, 4.0, "target", "2021-02-01"))
    mem.record_outcome(TradeOutcome("confluence_swing", "BBB", 0.62, -3.0, "stop", "2021-03-01"))
    mem.record_outcome(TradeOutcome("confluence_swing", "CCC", 0.70, 6.0, "target", "2021-04-01"))
    mem.add(Lesson("confluence_swing", "targets paid", True, "clean", as_of="2021-04-01"),
            human_reviewed=True)
    return mem


def test_setup_stats_and_calibration():
    mem = _seed()
    s = mem.setup_stats("confluence_swing")
    assert s["count"] == 3
    assert s["win_rate_pct"] == 67  # 2 of 3
    cal = mem.calibration()
    assert any(b["n"] for b in cal)


def test_recall_is_point_in_time():
    mem = _seed()
    # Before any trade closed, nothing is visible (no future leakage).
    assert mem.setup_stats("confluence_swing", not_after="2020-01-01") == {"count": 0}
    # Mid-stream only the earlier outcomes are visible.
    mid = mem.setup_stats("confluence_swing", not_after="2021-02-15")
    assert mid["count"] == 1
    # A lesson dated after the decision is not recalled.
    assert mem.relevant("confluence_swing", not_after="2021-01-01") == []
    assert mem.relevant("confluence_swing", not_after="2021-12-31")


def test_only_reviewed_lessons_recalled():
    mem = LessonMemory()
    mem.add(Lesson("confluence_swing", "unreviewed", True, "clean", as_of="2020-01-01"),
            human_reviewed=False)
    assert mem.relevant("confluence_swing") == []      # carries no weight until reviewed


def test_reflection_builds_lesson_with_provenance():
    agent = ReflectionAgent(MockLLMClient(), M.framing)
    les = agent.run({"trade": {"pnl": -120.0, "pnl_pct": -5.0, "reason": "stop",
                               "symbol": "ZZZ", "setup_type": "momentum_swing",
                               "as_of": "2026-06-05", "conviction": 0.4}})
    assert les.setup_type == "momentum_swing" and les.symbol == "ZZZ"
    assert les.thesis_correct is False and les.as_of == "2026-06-05"
    assert "ZZZ" in les.lesson


def test_memory_round_trips_through_json(tmp_path, monkeypatch):
    from app import learning
    mem = _seed()
    path = tmp_path / "learning.json"
    learning.save_memory(mem, path)
    back = learning.load_memory(path)
    assert len(back.outcomes) == 3 and len(back.entries) == 1
    assert back.setup_stats("confluence_swing")["count"] == 3
    # missing file -> empty memory, never raises
    assert learning.load_memory(tmp_path / "nope.json").outcomes == []


def test_paper_run_records_outcomes(synth_store):
    from system.run_live import PaperTradingEngine
    store, sector_map = synth_store
    mem = LessonMemory()
    PaperTradingEngine(store, sector_map, memory=mem).run()
    # The synthetic store closes at least one trade -> at least one recorded outcome+lesson.
    assert len(mem.outcomes) >= 1
    assert len(mem.entries) >= 1
    assert all(o.as_of for o in mem.outcomes)          # every outcome is PIT-tagged
