"""S&P 500 pre-filter: ranking, relative strength, regime, universe."""

import numpy as np
import pandas as pd

from app.screen import market_regime, prescreen


def _series(start, drift, vol=0.004, n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(start * np.cumprod(1 + rng.normal(drift, vol, n)), index=idx)


def _frame():
    # Low vol so drift (and thus relative strength) is deterministic-ish.
    return pd.DataFrame({
        "SPY": _series(100, 0.0004, seed=1),      # mild uptrend benchmark
        "LEAD": _series(50, 0.0016, seed=2),       # leader (high RS)
        "MID": _series(70, 0.0004, seed=3),        # in-line with market
        "LAG": _series(80, -0.0012, seed=4),       # laggard downtrend
    })


def test_prescreen_relative_strength_ordering_and_laggard_last():
    ranked, regime = prescreen(_frame(), top=10)
    syms = [r["symbol"] for r in ranked]
    assert "SPY" not in syms                   # benchmark excluded from picks
    assert syms[-1] == "LAG"                   # laggard ranks last
    rs = {r["symbol"]: r["rs"] for r in ranked}
    assert rs["LEAD"] > rs["MID"] > rs["LAG"]  # relative strength ordering holds
    score = {r["symbol"]: r["score"] for r in ranked}
    assert score["LEAD"] > score["LAG"]


def test_regime_detects_trend():
    up = pd.DataFrame({"SPY": _series(100, 0.0010, 0.006, seed=5)})
    down = pd.DataFrame({"SPY": _series(100, -0.0010, 0.006, seed=6)})
    assert market_regime(up)["above_200dma"] is True
    assert market_regime(down)["above_200dma"] is False


def test_prescreen_top_limit_and_short_series_dropped():
    df = _frame()
    df["NEW"] = df["LEAD"].copy()
    df.loc[df.index[:280], "NEW"] = np.nan      # too few points -> dropped
    ranked, _ = prescreen(df, top=2)
    assert len(ranked) <= 2
    assert "NEW" not in [r["symbol"] for r in ranked]


def test_sp500_universe_is_broad():
    from harness.data.sp500 import screen_universe
    syms = screen_universe()
    assert len(syms) >= 150                      # static fallback alone is broad
    assert "AAPL" in syms and "JPM" in syms
    assert len(screen_universe(20)) == 20
