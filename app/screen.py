"""Cheap, deterministic pre-filter that ranks a large universe (the S&P 500) so
only the most promising names spend LLM credits on the full agent panel.

No LLM, no fundamentals, no network beyond the one batched price download done by
the caller. Per-name metrics are price-only; the blended opportunity score and
the adaptive weighting live in :mod:`app.strategy` (relative strength, momentum,
trend, proximity to highs, an earnings-gap/PEAD proxy, and sector rotation). A
market-regime read (benchmark vs its 200-DMA) is returned so the desk can stand
down in a downtrend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import strategy

BENCH = "SPY"


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    v = out.dropna()
    return float(v.iloc[-1]) if len(v) else float("nan")


def _metrics(close: pd.Series, bench_mom6: float) -> dict | None:
    c = close.dropna()
    if len(c) < 150:
        return None
    last = float(c.iloc[-1])
    if last <= 0:
        return None
    mom6 = last / float(c.iloc[-126]) - 1 if len(c) > 126 else 0.0
    mom3 = last / float(c.iloc[-63]) - 1 if len(c) > 63 else 0.0
    sma200 = float(c.iloc[-200:].mean())
    win = c.iloc[-252:]
    high52 = float(win.max())
    low52 = float(win.min())
    dist_high = last / high52 - 1 if high52 else 0.0
    pct_above_low = last / low52 - 1 if low52 else 0.0
    # Implausible single-name moves are almost always corrupt source data (bad
    # split, vendor error), not real opportunities — flag so the ranker can drop
    # them instead of wasting a deep-dive slot on garbage.
    bad_data = abs(mom6) > 1.5 or pct_above_low > 4.0
    return {"price": round(last, 2), "mom6": mom6, "mom3": mom3,
            "above_200dma": last > sma200, "dist_high": dist_high,
            "rsi": _rsi(c), "rs": mom6 - bench_mom6,
            "earnings_gap": strategy.earnings_gap_drift(c), "bad_data": bad_data}


def market_regime(closes: pd.DataFrame, benchmark: str = BENCH) -> dict:
    if benchmark not in closes.columns:
        return {"available": False}
    m = _metrics(closes[benchmark], 0.0)
    if not m:
        return {"available": False}
    return {"available": True, "benchmark": benchmark,
            "above_200dma": m["above_200dma"], "mom6_pct": round(m["mom6"] * 100, 1),
            "regime": "risk-on (uptrend)" if m["above_200dma"] else "risk-off (downtrend)"}


def prescreen(closes: pd.DataFrame, top: int = 25, benchmark: str = BENCH,
              weights: dict | None = None, sector_etfs: dict | None = None,
              sector_of: dict | None = None,
              exclude: set | None = None) -> tuple[list[dict], dict]:
    """Rank the universe; return (ranked[:top], regime). The regime dict also
    carries 'sector_rs' (per-sector relative strength) for display."""
    regime = market_regime(closes, benchmark)
    w = weights or strategy.BASE_WEIGHTS
    sector_etfs = sector_etfs or {}
    sector_of = sector_of or {}
    sector_rs = strategy.sector_strength(closes, sector_etfs, benchmark)
    regime["sector_rs"] = sector_rs
    skip = set(exclude or set()) | set(sector_etfs.values()) | {benchmark}

    bench_mom6 = 0.0
    if benchmark in closes.columns:
        bm = _metrics(closes[benchmark], 0.0)
        bench_mom6 = bm["mom6"] if bm else 0.0
    rows, dropped = [], 0
    for sym in closes.columns:
        if sym in skip:
            continue
        m = _metrics(closes[sym], bench_mom6)
        if m is None:
            continue
        if m.get("bad_data"):                 # corrupt source data — never rank it
            dropped += 1
            continue
        m["symbol"] = sym
        m["sector"] = sector_of.get(sym)
        m["score"] = strategy.composite_score(m, w, sector_rs)
        rows.append(m)
    rows.sort(key=lambda r: r["score"], reverse=True)
    regime["dropped_bad_data"] = dropped
    return rows[:top], regime
