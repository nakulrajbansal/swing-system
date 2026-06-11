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


def test_prescreen_drops_single_day_discontinuities():
    df = _frame()
    # Corrupt data = a single-session discontinuity (a missed split / vendor
    # error), e.g. a one-day 10x jump. That must be excluded.
    bad = _series(50, 0.0004, n=300, seed=9)
    bad.iloc[200:] *= 10.0                       # 10x overnight = split artifact
    df["BADX"] = bad
    ranked, regime = prescreen(df, top=10)
    assert "BADX" not in [r["symbol"] for r in ranked]
    assert regime.get("dropped_bad_data", 0) >= 1


def test_prescreen_keeps_genuine_big_winners():
    df = _frame()
    # A smooth +200%-style rally (the NVDA-2023 kind) is NOT corrupt data — it
    # is exactly what the screen exists to find, and must rank at the top.
    df["WINR"] = _series(50, 0.006, vol=0.004, n=300, seed=11)
    ranked, regime = prescreen(df, top=10)
    syms = [r["symbol"] for r in ranked]
    assert "WINR" in syms
    assert syms[0] == "WINR"                     # strongest name ranks first
    assert regime.get("dropped_bad_data", 0) == 0


def test_hidden_gem_slots_surface_accelerating_names():
    from app import strategy

    def m(sym, score, accel=0.0, mom3=0.0, mom6=0.0, mom12=0.0, gap=0.0,
          up=True, dist=-0.2, rsi=60.0):
        return {"symbol": sym, "score": score, "accel": accel, "mom3": mom3,
                "mom6": mom6, "mom12": mom12, "earnings_gap": gap,
                "above_200dma": up, "dist_high": dist, "rsi": rsi}

    ranked = [
        m("HOT1", 2.0, mom3=0.20, mom6=0.45, mom12=0.9, dist=-0.02),   # consensus leader
        m("HOT2", 1.8, mom3=0.18, mom6=0.40, mom12=0.8, dist=-0.03),
        m("MEH1", 0.9, mom3=0.05, mom6=0.10),
        m("MEH2", 0.8, mom3=0.04, mom6=0.09),
        # Igniting: 3-month pace way above the 6-month average pace, still well
        # below its high, modest 12-month record — the pre-consensus profile.
        m("GEM", 0.3, accel=0.25, mom3=0.30, mom6=0.10, mom12=0.15,
          gap=0.10, dist=-0.20),
        m("DOWN", -0.5, accel=0.30, mom3=0.30, up=False),              # in a downtrend
    ]
    out = strategy.select_shortlist(ranked, k=4, gem_slots=2)
    syms = [m_["symbol"] for m_ in out]
    assert len(out) == 4
    assert "GEM" in syms                          # gem slot found the igniter
    assert "DOWN" not in syms                     # never a falling knife
    gem = next(m_ for m_ in out if m_["symbol"] == "GEM")
    assert gem.get("hidden_gem") is True
    assert syms[0] == "HOT1"                      # core slots still score-ordered


def test_prescreen_liquidity_floor_drops_untradable_names():
    df = _frame()
    df["THIN"] = _series(40, 0.0010, seed=12)     # decent trend ...
    df["PENNY"] = _series(2.0, 0.0008, seed=13)   # ... and a $2 micro-name
    vols = pd.DataFrame(1_000_000.0, index=df.index, columns=df.columns)
    vols["THIN"] = 5_000.0                        # ~$200k/day: below the floor
    ranked, regime = prescreen(df, top=10, volumes=vols)
    syms = [r["symbol"] for r in ranked]
    assert "THIN" not in syms and "PENNY" not in syms
    assert regime.get("dropped_illiquid", 0) >= 2
    assert "LEAD" in syms                          # liquid names unaffected


def test_accumulation_volume_lifts_score():
    df = _frame()
    closes = df[["SPY", "LEAD"]].copy()
    closes["ACCU"] = closes["LEAD"] * 1.001        # near-identical price paths
    rets = closes["ACCU"].pct_change()
    base = pd.Series(1_000_000.0, index=closes.index)
    accu_vol = base.where(rets <= 0, base * 3)     # volume piles into UP days
    dist_vol = base.where(rets > 0, base * 3)      # volume piles into DOWN days
    vols = pd.DataFrame({"SPY": base, "LEAD": dist_vol, "ACCU": accu_vol})
    ranked, _ = prescreen(closes, top=10, volumes=vols)
    score = {r["symbol"]: r["score"] for r in ranked}
    assert score["ACCU"] > score["LEAD"]           # accumulation footprint rewarded


def test_sp500_universe_is_broad():
    from harness.data.sp500 import screen_universe
    syms = screen_universe()
    assert len(syms) >= 150                      # static fallback alone is broad
    assert "AAPL" in syms and "JPM" in syms
    assert len(screen_universe(20)) == 20
