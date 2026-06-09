"""Strategy brain: PEAD proxy, adaptive weights, sector RS, portfolio, backtest."""

import numpy as np
import pandas as pd

from app import strategy
from app.screen import _metrics


def _series(start, drift, vol=0.005, n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(start * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)


def test_earnings_gap_drift_detects_held_up_gap():
    c = _series(100, 0.0002, vol=0.004, seed=1).reset_index(drop=True)
    # Inject a +12% gap 10 sessions ago, then a mild hold.
    c.iloc[-10] = c.iloc[-11] * 1.12
    c.iloc[-9:] = c.iloc[-10] * 1.01
    val = strategy.earnings_gap_drift(c)
    assert val > 0.05                       # positive PEAD signal
    flat = strategy.earnings_gap_drift(_series(100, 0.0, vol=0.003, seed=2))
    assert abs(flat) < 0.05                 # no event -> ~0


def test_factor_weights_adapt_to_regime():
    base = strategy.factor_weights(None, None)
    risk_off = strategy.factor_weights({"available": True, "above_200dma": False}, None)
    assert risk_off["rs"] < base["rs"]          # momentum trusted less in a downtrend
    assert risk_off["trend"] > base["trend"]    # demand the 200-DMA more


def test_factor_weights_adapt_to_memory():
    from system.reflection.memory import LessonMemory, TradeOutcome
    bad = LessonMemory()
    for i in range(12):
        bad.record_outcome(TradeOutcome("confluence_swing", "X", 0.6, -2.0, "stop", "2025-01-01"))
    w = strategy.factor_weights(None, bad)
    assert w["rs"] < strategy.BASE_WEIGHTS["rs"]   # losing streak -> trim momentum


def test_sector_strength_uses_etfs():
    df = pd.DataFrame({
        "SPY": _series(100, 0.0003, n=300, seed=3),
        "XLK": _series(100, 0.0012, n=300, seed=4),     # strong sector
        "XLU": _series(100, -0.0006, n=300, seed=5),    # weak sector
    })
    rs = strategy.sector_strength(df, {"Information Technology": "XLK", "Utilities": "XLU"})
    assert rs["Information Technology"] > rs["Utilities"]


def test_construct_portfolio_caps_and_regime_budget():
    # Eight names so a fully-invested book is reachable within the per-name cap.
    recs = [{"symbol": f"S{i}", "entry": 50, "conviction": 0.9 - i * 0.05,
             "sector": "Tech" if i % 2 else "Energy"} for i in range(8)]
    on = strategy.construct_portfolio(recs, 100000, {"available": True, "above_200dma": True})
    assert all(p["weight_pct"] <= 25.01 for p in on)            # per-name cap holds
    by_sec = {}
    for p in on:
        by_sec[p["sector"]] = by_sec.get(p["sector"], 0) + p["weight_pct"]
    assert all(v <= 45.5 for v in by_sec.values())             # per-sector cap holds
    on_inv = sum(p["weight_pct"] for p in on)
    off = strategy.construct_portfolio(recs, 100000, {"available": True, "above_200dma": False})
    off_inv = sum(p["weight_pct"] for p in off)
    assert off_inv < on_inv                                     # risk-off invests less
    assert off_inv <= 51


def test_walk_forward_backtest_runs_and_reports():
    idx = pd.date_range("2022-01-01", periods=500, freq="B")
    rng = np.random.default_rng(7)
    cols = {"SPY": 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, 500))}
    for i in range(6):
        cols[f"S{i}"] = 50 * np.cumprod(1 + rng.normal(0.0002 + i * 0.0001, 0.012, 500))
    df = pd.DataFrame(cols, index=idx)
    res = strategy.walk_forward_backtest(df, _metrics, strategy.BASE_WEIGHTS,
                                         top_k=2, hold_days=10)
    assert res["periods"] > 5
    for k in ("strategy_total_return_pct", "benchmark_total_return_pct",
              "excess_return_pct", "win_rate_pct", "max_drawdown_pct", "strategy_sharpe"):
        assert k in res
