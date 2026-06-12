"""Phase A/B/C upgrades: calibration, preset self-tuning, debate escalation,
watchlist triggers."""

import numpy as np
import pandas as pd

from system.reflection.calibration import (calibrated_probability,
                                           calibration_table, describe)


def _scored(conviction, ret, n):
    return [{"status": "evaluated", "conviction": conviction, "return_pct": ret,
             "evaluated_on": f"2026-06-{10 + i % 19:02d}"} for i in range(n)]


def test_calibration_shrinks_honestly():
    # No evidence: the desk's claim comes back unchanged, labeled uncalibrated.
    empty = calibration_table([])
    assert calibrated_probability(0.62, empty) == 0.62
    assert "uncalibrated" in describe(0.62, empty)
    # A seasoned band pulls toward the realized rate: 0.6-conviction calls that
    # won only 30% of 20 tries must read well below the stated 0.60.
    rows = _scored(0.60, -3.0, 14) + _scored(0.60, 5.0, 6)     # 30% win rate
    table = calibration_table(rows)
    p = calibrated_probability(0.60, table)
    assert p < 0.45
    assert "calibrated on 20" in describe(p, table)
    # Garbage input never crashes.
    assert calibrated_probability(None, table) is None


def test_calibration_uses_trailing_window():
    old_losses = _scored(0.60, -5.0, 80)        # ancient losing streak
    recent_wins = [{"status": "evaluated", "conviction": 0.60, "return_pct": 4.0,
                    "evaluated_on": "2026-06-30"} for _ in range(20)]
    table = calibration_table(old_losses + recent_wins, trailing=20)
    band = next(b for b in table["bands"] if b["lo"] == 0.55)
    assert band["win_rate_pct"] == 100.0        # only the recent window counts


def test_preset_selector_picks_the_winning_personality():
    from app.screen import _metrics
    from app.strategy import WEIGHT_PRESETS, select_preset

    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-06-01", periods=320, freq="B")
    closes = pd.DataFrame({
        sym: 50 * np.cumprod(1 + rng.normal(d, 0.004, 320))
        for sym, d in [("SPY", 0.0004), ("AAA", 0.0012), ("BBB", 0.0008),
                       ("CCC", 0.0002), ("DDD", -0.0004)]}, index=idx)
    name, weights, board = select_preset(closes, _metrics, top_k=2, hold_days=10)
    assert name in WEIGHT_PRESETS and weights == WEIGHT_PRESETS[name]
    assert set(board) == set(WEIGHT_PRESETS)
    assert all("total" in v for v in board.values())
    # Too-short history degrades to base, never crashes.
    nm, w, _ = select_preset(closes.iloc[:50], _metrics)
    assert nm == "base"


def test_contested_debate_fires_second_round(synth_store):
    from harness.signals import Edge01Filing, Edge06Insider, Edge08Momentum
    from system.agents.core import (HypothesisAgent, PortfolioManagerAgent,
                                    SkepticAgent)
    from system.agents.llm_client import MockLLMClient
    from system.agents.specialists import EdgeSpecialist
    from system.config import SystemConfig
    from system.orchestrator import Orchestrator
    from harness.data.loader import available_at_for_session

    store, sector_map = synth_store
    cfg = SystemConfig()
    client = MockLLMClient()

    class SevereSkeptic(SkepticAgent):
        def deterministic(self, inputs):
            crit = super().deterministic(inputs)
            for o in crit.objections:
                o.severity = 0.65                       # contested, not a kill
            crit.verdict = "caution"
            return crit

    specs = [EdgeSpecialist(E(), client, cfg.models.framing)
             for E in (Edge01Filing, Edge06Insider, Edge08Momentum)]
    orch = Orchestrator(store, specs, HypothesisAgent(client, cfg.models.synthesis),
                        SevereSkeptic(client, cfg.models.adversarial),
                        PortfolioManagerAgent(client, cfg.models.adversarial), cfg)
    sessions = pd.to_datetime(sorted(store.read_table("prices")["date"].unique()))
    saw_rejoinder = False
    for d in sessions[len(sessions) // 2:]:
        res = orch.run_cycle(available_at_for_session(d))
        for rec in res.deliberation.values():
            agents = [s["agent"] for s in rec.get("steps", [])]
            if "skeptic_rejoinder" in agents:
                saw_rejoinder = True
                out = next(s["output"] for s in rec["steps"]
                           if s["agent"] == "skeptic_rejoinder")
                assert out["stands"] is True            # deterministic: unchanged
        if saw_rejoinder:
            break
    assert saw_rejoinder, "high-severity critique should trigger round 2"


def test_pm_weighs_rejoinder_outcome():
    from system.agents.core import PortfolioManagerAgent
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    pm = PortfolioManagerAgent(MockLLMClient(), SystemConfig().models.adversarial)
    base = {"symbol": "AAA",
            "hypothesis": {"decision": "propose", "raw_conviction": 0.85},
            "critique": {"verdict": "caution", "max_severity": 0.6},
            "rebuttal": "addressed", "price": 100.0, "atr": 2.0}
    conceded = pm.run({**base, "rejoinder": {"stands": False, "final_severity": 0.6}})
    stands = pm.run({**base, "rejoinder": {"stands": True, "final_severity": 0.8}})
    # A conceded objection frees conviction; a standing one drags it down.
    assert conceded.final_conviction > stands.final_conviction


def test_factor_weights_stacks_on_a_preset_base():
    from app.strategy import WEIGHT_PRESETS, factor_weights

    w = factor_weights(None, None, base=WEIGHT_PRESETS["timing"])
    assert w["timing"] == WEIGHT_PRESETS["timing"]["timing"]
    risk_off = factor_weights({"available": True, "above_200dma": False}, None,
                              base=WEIGHT_PRESETS["timing"])
    assert risk_off["rs"] < WEIGHT_PRESETS["timing"]["rs"]   # regime still applies
