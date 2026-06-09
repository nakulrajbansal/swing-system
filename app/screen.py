"""Cheap, deterministic pre-filter that ranks a large universe (the S&P 500) so
only the most promising names spend LLM credits on the full agent panel.

No LLM, no fundamentals, no network beyond the one batched price download done by
the caller. The score rewards what tends to lead the market over a 2-20 day swing:
relative strength vs the benchmark (beating the market), a confirmed uptrend,
positive medium-term momentum, proximity to highs, and NOT being blown-off /
overbought. A market-regime read (benchmark vs its 200-DMA) is returned so the
desk can stand down in a downtrend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    dist_high = last / high52 - 1 if high52 else 0.0
    rsi = _rsi(c)
    rs = mom6 - bench_mom6                      # relative strength vs benchmark
    return {"price": round(last, 2), "mom6": mom6, "mom3": mom3,
            "above_200dma": last > sma200, "dist_high": dist_high,
            "rsi": rsi, "rs": rs}


def _score(m: dict) -> float:
    """Blend the metrics into one opportunity score (higher = more promising)."""
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))

    s = 0.0
    s += clamp(m["rs"], -0.5, 0.5) * 2.0           # relative strength dominates
    s += clamp(m["mom6"], -0.5, 0.5) * 1.0         # absolute medium-term momentum
    s += clamp(m["mom3"], -0.3, 0.3) * 0.5         # recent acceleration
    s += 0.30 if m["above_200dma"] else -0.30      # trend regime
    if m["dist_high"] > -0.08:                      # near highs (breakout proximity)
        s += 0.20
    elif m["dist_high"] < -0.30:                    # deep drawdown (falling knife)
        s -= 0.20
    rsi = m["rsi"]
    if rsi == rsi:                                  # not NaN
        if rsi > 80:
            s -= 0.45                               # blow-off / overbought
        elif rsi > 72:
            s -= 0.15
        elif rsi < 35:
            s -= 0.10                               # weak, no confirmation
    return s


def market_regime(closes: pd.DataFrame, benchmark: str = BENCH) -> dict:
    if benchmark not in closes.columns:
        return {"available": False}
    m = _metrics(closes[benchmark], 0.0)
    if not m:
        return {"available": False}
    return {"available": True, "benchmark": benchmark,
            "above_200dma": m["above_200dma"], "mom6_pct": round(m["mom6"] * 100, 1),
            "regime": "risk-on (uptrend)" if m["above_200dma"] else "risk-off (downtrend)"}


def prescreen(closes: pd.DataFrame, top: int = 25,
              benchmark: str = BENCH) -> tuple[list[dict], dict]:
    """Rank the universe; return (ranked[:top], regime). Each row is a metrics dict
    plus 'symbol' and 'score'."""
    regime = market_regime(closes, benchmark)
    bench_mom6 = 0.0
    if benchmark in closes.columns:
        bm = _metrics(closes[benchmark], 0.0)
        bench_mom6 = bm["mom6"] if bm else 0.0
    rows = []
    for sym in closes.columns:
        if sym == benchmark:
            continue
        m = _metrics(closes[sym], bench_mom6)
        if m is None:
            continue
        m["symbol"] = sym
        m["score"] = round(_score(m), 3)
        rows.append(m)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top], regime
