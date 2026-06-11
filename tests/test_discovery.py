"""Discovery upgrades: XBRL quarterly trajectory (PIT-safe), the moat analyst's
inflection read, earnings event risk, lens-cohort learning, gem-cohort backtest,
and the mid/small-cap universes."""

import tempfile
from types import SimpleNamespace

import pandas as pd

from harness.data.loader import parse_companyfacts
from harness.data.pit_store import PITStore


def _facts_json():
    """9 quarters: revenue YoY ACCELERATING into the latest quarters while gross
    margin expands — the inflection fingerprint."""
    ends = pd.date_range("2023-03-31", periods=9, freq="QE")
    revs = [100, 100, 100, 100, 110, 115, 122, 132, 154]
    gms = [0.50, 0.50, 0.50, 0.50, 0.51, 0.52, 0.53, 0.55, 0.56]

    def entries(vals):
        out = []
        for end, v in zip(ends, vals):
            start = (end - pd.Timedelta(days=90)).date().isoformat()
            filed = (end + pd.Timedelta(days=40)).date().isoformat()
            out.append({"start": start, "end": end.date().isoformat(),
                        "val": v, "filed": filed, "form": "10-Q"})
        return out

    return {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": entries(revs)}},
        "GrossProfit": {"units": {"USD": entries([r * g for r, g in zip(revs, gms)])}},
        "OperatingIncomeLoss": {"units": {"USD": entries([r * 0.2 for r in revs])}},
    }}}


def test_companyfacts_parser_builds_pit_rows():
    df = parse_companyfacts(_facts_json(), "GEMX")
    assert len(df) == 9
    assert {"symbol", "available_at", "period_end", "revenue", "gross_profit"} <= set(df.columns)
    # available_at must be tz-aware and AFTER the period end (the filing date) —
    # a quarter is never visible before it was reported.
    assert df["available_at"].dt.tz is not None
    assert (df["available_at"].dt.tz_localize(None) > df["period_end"]).all()
    # Annual (non-quarterly) entries are ignored entirely.
    j = _facts_json()
    j["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
        {"start": "2024-01-01", "end": "2024-12-31", "val": 999,
         "filed": "2025-02-15", "form": "10-K"})
    assert len(parse_companyfacts(j, "GEMX")) == 9


def _store_with_history():
    store = PITStore(tempfile.mkdtemp(prefix="traj_"))
    store.write_fundamentals_history(parse_companyfacts(_facts_json(), "GEMX"))
    return store


def test_trajectory_flags_inflection_and_respects_pit():
    from system.data_plane.evidence import _trajectory

    store = _store_with_history()
    t = _trajectory(store.as_of(pd.Timestamp("2025-06-01", tz="UTC")), "GEMX")
    assert t["available"]
    assert t["revenue_accelerating"] and t["revenue_accelerating_2q"]
    assert t["margins_expanding"]
    assert t["revenue_yoy_latest_pct"] > 15
    # Point-in-time: a decision in mid-2024 must NOT see the 2025 acceleration.
    early = _trajectory(store.as_of(pd.Timestamp("2024-06-01", tz="UTC")), "GEMX")
    assert not early.get("revenue_accelerating_2q", False)


def test_moat_analyst_reads_the_inflection():
    from system.agents.analysts import MoatAnalyst
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    m = SystemConfig().models
    ev = {"fundamentals": {
        "available": True, "growth": {},
        "moat": {"gross_margin_pct": 56.0, "operating_margin_pct": 20.0},
        "trajectory": {"available": True, "revenue_accelerating": True,
                       "revenue_accelerating_2q": True, "margins_expanding": True,
                       "revenue_yoy_latest_pct": 40.0, "gross_margin_trend_pct": 5.0}}}
    read = MoatAnalyst(MockLLMClient(), m.framing).run({"symbol": "GEMX", "evidence": ev})
    assert read.stance == "bullish"
    assert any("INFLECTING" in p for p in read.positives)
    # Decelerating + margin compression reads as a melting story.
    ev2 = {"fundamentals": {
        "available": True, "growth": {},
        "moat": {"gross_margin_pct": 30.0},
        "trajectory": {"available": True, "revenue_decelerating_2q": True,
                       "margins_expanding": False, "gross_margin_trend_pct": -4.0}}}
    read2 = MoatAnalyst(MockLLMClient(), m.framing).run({"symbol": "MELT", "evidence": ev2})
    assert read2.score < 0.5


def test_skeptic_flags_earnings_inside_hold_window():
    from system.agents.core import SkepticAgent
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    crit = SkepticAgent(MockLLMClient(), SystemConfig().models.adversarial).run(
        {"symbol": "AAA",
         "evidence": {"events": {"available": True, "days_to_earnings": 5,
                                 "next_earnings_date": "2026-06-15",
                                 "earnings_within_swing_window": True}},
         "max_corr_to_book": 0.0, "min_read_confidence": 0.9, "priced_in": 0.0})
    assert any(o.kind == "event_risk" for o in crit.objections)


def test_events_block_computes_days_to_earnings():
    from system.data_plane.evidence import _events

    view = SimpleNamespace(asof_date=pd.Timestamp("2026-06-10"))
    e = _events(view, "AAA", {"next_earnings_date": "2026-06-18"})
    assert e["available"] and e["days_to_earnings"] == 8
    assert e["earnings_within_swing_window"]
    assert not _events(view, "AAA", {"next_earnings_date": None})["available"]


def test_ledger_cohorts_split_by_lens(tmp_path):
    from app import reco_ledger

    p = tmp_path / "ledger.json"
    recs = [
        {"symbol": "GEM1", "entry": 10, "hold_days": 5, "exit_by": "2026-01-10",
         "hidden_gem": True, "moat_stance": "bullish", "conviction": 0.6},
        {"symbol": "CORE1", "entry": 10, "hold_days": 5, "exit_by": "2026-01-10",
         "hidden_gem": False, "moat_stance": "neutral", "conviction": 0.6},
    ]
    assert reco_ledger.record(recs, "test", "2026-01-02", path=p) == 2
    led = reco_ledger.load(p)
    assert led[0]["hidden_gem"] is True and led[0]["moat_stance"] == "bullish"
    # Score them and check the cohort split.
    led[0].update(status="evaluated", return_pct=8.0)
    led[1].update(status="evaluated", return_pct=-2.0)
    reco_ledger.save(led, p)
    co = reco_ledger.cohort_stats(path=p)
    assert co["hidden_gem"]["n"] == 1 and co["hidden_gem"]["avg_return_pct"] == 8.0
    assert co["core"]["n"] == 1 and co["core"]["avg_return_pct"] == -2.0
    assert co["moat_bullish"]["n"] == 1 and co["moat_other"]["n"] == 1


def test_walk_forward_reports_gem_cohort():
    import numpy as np
    from app import strategy
    from app.screen import _metrics

    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-02", periods=320, freq="B")
    closes = pd.DataFrame({
        sym: 50 * np.cumprod(1 + rng.normal(d, 0.004, 320))
        for sym, d in [("SPY", 0.0004), ("AAA", 0.0010), ("BBB", 0.0006),
                       ("CCC", 0.0002), ("DDD", -0.0004)]}, index=idx)
    res = strategy.walk_forward_backtest(closes, _metrics, strategy.BASE_WEIGHTS,
                                         top_k=2, hold_days=10)
    assert "gem_total_return_pct" in res and "gem_picks" in res
    assert res["periods"] > 0


def test_midsmall_universe_static_fallback(monkeypatch):
    from harness.data import midsmall as ms

    monkeypatch.setattr(ms, "_live", lambda idx: ((), ()))
    syms, sect = ms.screen_universe("midsmall")
    assert len(syms) >= 100
    assert all(s in sect for s in syms)          # every name has a sector
    capped, _ = ms.screen_universe("sp400", 20)
    assert len(capped) == 20
