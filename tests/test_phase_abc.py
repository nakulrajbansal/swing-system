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


def test_infer_exit_reason_names_the_trigger():
    from app.runner import _infer_exit_reason

    # AMAT-style plan: stop 443.79, target 588.92.
    assert _infer_exit_reason(588.10, 443.79, 588.92) == "target"   # ~target fill
    assert _infer_exit_reason(443.00, 443.79, 588.92) == "stop"     # stopped out
    assert _infer_exit_reason(440.00, 443.79, 588.92) == "stop"     # gapped through
    assert _infer_exit_reason(520.00, 443.79, 588.92) == "manual"   # neither level
    assert _infer_exit_reason(None, 443.79, 588.92) == "manual"
    assert _infer_exit_reason(100.0, None, None) == "manual"


def test_broker_fills_pages_within_alpaca_cap(monkeypatch):
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("key", "secret", env="paper")
    calls = []
    # 150 fills across two pages; Alpaca caps page_size at 100.
    full = [{"id": f"f{i}", "symbol": "AMAT", "side": "sell"} for i in range(150)]

    def fake(method, path, payload=None):
        calls.append(path)
        assert "page_size=100" in path or "page_size=50" in path   # never >100
        if "page_token=" in path:
            return full[100:]
        return full[:100]

    monkeypatch.setattr(b, "_req", fake)
    fills = b.fills(200)
    assert len(fills) == 150
    assert "activities?activity_types=FILL" in calls[0]
    assert any("page_token=f99" in c for c in calls)               # cursor advanced


def test_executed_link_and_full_open_list(tmp_path):
    from app import reco_ledger

    p = tmp_path / "ledger.json"
    recs = [{"symbol": s, "entry": 100.0, "hold_days": 10,
             "exit_by": f"2026-06-{17 + i}", "conviction": 0.5}
            for i, s in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE",
                                   "FFF", "GGG", "HHH", "III", "JJJ"])]
    reco_ledger.record(recs, "screen-sp500", "2026-06-11", path=p)
    assert reco_ledger.mark_executed("CCC", qty=80, path=p)
    assert not reco_ledger.mark_executed("ZZZ", path=p)        # unknown: no-op
    text = reco_ledger.summarize(path=p)
    # ALL ten open entries are listed (the old view truncated to 8).
    assert all(s in text for s in ("AAA", "JJJ"))
    assert "10 open (1 executed in your account)" in text
    assert "● 2026-06-11  CCC" in text and "x80" in text
    assert "○ 2026-06-11  AAA" in text                          # advisory-only


def test_wrap_never_truncates():
    from app.runner import _wrap

    long = ("The fundamental engine is real and durable: revenue re-accelerating "
            "for two consecutive quarters while operating margins inflect, and "
            "the moat read confirms a defensible niche position. " * 3)
    lines = _wrap(long, indent="      ")
    assert "".join(lines).replace(" ", "") == ("      " + long).replace(" ", "")[:0] \
        or " ".join(ln.strip() for ln in lines) == " ".join(long.split())
    assert all(len(ln) <= 100 for ln in lines)
    assert not any(ln.rstrip().endswith("...") for ln in lines)
    assert _wrap("") == [] and _wrap(None) == []


def test_weekly_trend_quality_rewards_confirmed_uptrends():
    from app.strategy import composite_score

    base = {"rs": 0.2, "mom6": 0.2, "mom3": 0.1, "above_200dma": True,
            "dist_high": -0.1, "earnings_gap": 0.0, "accel": 0.0, "ext20": 0.01}
    confirmed = composite_score({**base, "trend_quality": 1})
    bounce = composite_score({**base, "trend_quality": -1})
    assert confirmed > bounce          # confirmed structure beats a fragile bounce


def test_news_sentiment_scores_tone_and_flags_bearish():
    from system.data_plane.evidence import _news_sentiment

    bull = _news_sentiment(["Company beats earnings, raises guidance",
                            "Analyst upgrade as orders surge to record"])
    assert bull["tone"] == "bullish" and bull["bear_hits"] == 0
    bear = _news_sentiment(["SEC probe over accounting; shares plunge",
                            "Analyst downgrade after guidance cut"])
    assert bear["tone"] == "bearish" and bear["bearish_items"]
    assert _news_sentiment([])["available"] is False


def test_guardian_exits_on_thesis_contradicting_news():
    from system.agents.llm_client import MockLLMClient
    from system.agents.meta import GuardianAgent
    from system.config import SystemConfig

    g = GuardianAgent(MockLLMClient(), SystemConfig().models.framing)
    d = g.run({"symbol": "AAA", "evidence": {"news_sentiment": {
        "available": True, "tone": "bearish",
        "bearish_items": ["recall and federal probe announced"]}}})
    assert d.action == "exit" and "probe" in d.reason
    # Bullish/mixed news never forces an exit.
    assert g.run({"symbol": "AAA", "evidence": {"news_sentiment": {
        "available": True, "tone": "bullish", "bearish_items": []}}}).action == "hold"


def test_self_throttle_dormant_until_enough_data_then_derisks():
    from system.reflection.calibration import desk_throttle

    def calls(n, ret, conv=0.6):
        return [{"status": "evaluated", "return_pct": ret, "conviction": conv,
                 "evaluated_on": f"2026-06-{10 + i % 19:02d}"} for i in range(n)]

    # Thin record: dormant no matter how bad.
    assert desk_throttle(calls(5, -8.0))["active"] is False
    # Cold streak with enough evidence: de-risk (raise bar, cut gross).
    cold = desk_throttle(calls(12, -6.0))
    assert cold["active"] and cold["conviction_bump"] > 0 and cold["gross_scale"] < 1.0
    # Healthy record: never throttles (and never ADDS risk).
    warm = desk_throttle(calls(12, 7.0))
    assert warm["active"] is False and warm["gross_scale"] == 1.0


def test_correlation_diversify_drops_near_duplicates_only():
    import numpy as np
    from app.strategy import correlation_diversify

    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    rng = np.random.default_rng(3)
    base = np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    twin = base * (1 + rng.normal(0, 0.0005, 120))      # ~identical to base
    indep = np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    closes = pd.DataFrame({"AAA": base, "ATWIN": twin, "BBB": indep}, index=idx)
    # AAA ranked first, its twin second, independent BBB third; k=2 should skip
    # the twin and take the independent name instead.
    out = correlation_diversify(["AAA", "ATWIN", "BBB"], closes, k=2)
    assert out == ["AAA", "BBB"]
    # k>=len always returns everyone (redundant ones just pushed to the back).
    assert set(correlation_diversify(["AAA", "ATWIN", "BBB"], closes, k=3)) == {
        "AAA", "ATWIN", "BBB"}
    # No data / single slot: graceful passthrough.
    assert correlation_diversify(["AAA", "BBB"], None, k=2) == ["AAA", "BBB"]


def test_growth_analyst_reads_estimate_revisions():
    from system.agents.analysts import GrowthAnalyst
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    m = SystemConfig().models.framing
    up = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 18.0, "eps_revision_90d_pct": 12.0, "num_analysts": 20}}}
    r = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "X", "evidence": up})
    assert any("REVISED UP" in p for p in r.positives)
    down = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 5.0, "eps_revision_90d_pct": -10.0, "num_analysts": 15}}}
    r2 = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "Y", "evidence": down})
    assert any("CUT" in c for c in r2.concerns) and r2.score < 0.5
    # Thin coverage (no analysts) -> the revision is ignored, not trusted.
    thin = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 5.0, "eps_revision_90d_pct": 30.0, "num_analysts": 1}}}
    r3 = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "Z", "evidence": thin})
    assert not any("REVISED UP" in p for p in r3.positives)


def test_skeptic_flags_poor_earnings_quality():
    from system.agents.core import SkepticAgent
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    crit = SkepticAgent(MockLLMClient(), SystemConfig().models.adversarial).run(
        {"symbol": "AAA",
         "evidence": {"earnings_quality": {"rating": "poor", "profit_margin_pct": 22.0,
                                           "fcf_margin_pct": 3.0, "accrual_gap_pp": 19.0}},
         "max_corr_to_book": 0.0, "min_read_confidence": 0.9, "priced_in": 0.0})
    assert any(o.kind == "earnings_quality" for o in crit.objections)


def test_watchlist_lifecycle_and_triggers(tmp_path):
    from app import watchlist as wl

    p = tmp_path / "watch.json"
    n = wl.add([{"symbol": "SITM", "reason": "PM pullback", "pullback_target": 630.0},
                {"symbol": "ARCB", "reason": "WATCH tier", "breakout_level": 150.0},
                {"symbol": "NOLEVELS", "reason": "ignored"}],     # no triggers: skipped
               "2026-06-11", path=p)
    assert n == 2
    assert len(wl.active("2026-06-12", path=p)) == 2
    assert wl.active("2026-08-01", path=p) == []                  # expired + pruned

    wl.add([{"symbol": "SITM", "reason": "PM pullback", "pullback_target": 630.0},
            {"symbol": "ARCB", "reason": "WATCH tier", "breakout_level": 150.0}],
           "2026-06-11", path=p)
    idx = pd.date_range("2026-05-01", periods=30, freq="B")
    closes = pd.DataFrame({"SITM": 660.0, "ARCB": 140.0}, index=idx)
    closes.loc[idx[-1], "SITM"] = 633.0                           # entered the window
    vols = pd.DataFrame(1_000_000.0, index=idx, columns=closes.columns)
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert [h["kind"] for h in hits] == ["pullback"]
    # Breakout requires the level AND unusual volume.
    closes.loc[idx[-1], "ARCB"] = 151.0
    vols.loc[idx[-1], "ARCB"] = 900_000.0                         # rvol < 1.5: no hit
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert not any(h["symbol"] == "ARCB" for h in hits)
    vols.loc[idx[-1], "ARCB"] = 2_000_000.0                       # rvol 2.0: hit
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert any(h["symbol"] == "ARCB" and h["kind"] == "breakout" for h in hits)
    wl.remove({"ARCB"}, path=p)
    assert [it["symbol"] for it in wl.active("2026-06-12", path=p)] == ["SITM"]


def test_factor_weights_stacks_on_a_preset_base():
    from app.strategy import WEIGHT_PRESETS, factor_weights

    w = factor_weights(None, None, base=WEIGHT_PRESETS["timing"])
    assert w["timing"] == WEIGHT_PRESETS["timing"]["timing"]
    risk_off = factor_weights({"available": True, "above_200dma": False}, None,
                              base=WEIGHT_PRESETS["timing"])
    assert risk_off["rs"] < WEIGHT_PRESETS["timing"]["rs"]   # regime still applies
