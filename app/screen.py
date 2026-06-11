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


def _volume_metrics(c: pd.Series, volume: pd.Series | None) -> dict:
    """Liquidity + accumulation reads from share volume (all None if absent).

    dollar_vol_m: median daily dollar volume over ~3 months (liquidity floor).
    vol_ratio: avg volume last 20 sessions vs the prior ~100 (expanding interest).
    updown_vol: up-day volume / down-day volume over ~3 months (>1 = accumulation
    — institutions buying shows in WHERE the volume happens before it shows in price).
    """
    out = {"dollar_vol_m": None, "vol_ratio": None, "updown_vol": None}
    if volume is None:
        return out
    v = volume.reindex(c.index).fillna(0.0)
    if len(v) < 130 or float(v.iloc[-60:].sum()) <= 0:
        return out
    out["dollar_vol_m"] = float((c.iloc[-63:] * v.iloc[-63:]).median()) / 1e6
    base = float(v.iloc[-120:-20].mean())
    if base > 0:
        out["vol_ratio"] = float(v.iloc[-20:].mean()) / base
    rets = c.pct_change().iloc[-63:]
    up = float(v.iloc[-63:][rets > 0].sum())
    dn = float(v.iloc[-63:][rets < 0].sum())
    if dn > 0:
        out["updown_vol"] = up / dn
    return out


def _metrics(close: pd.Series, bench_mom6: float,
             volume: pd.Series | None = None) -> dict | None:
    c = close.dropna()
    if len(c) < 150:
        return None
    last = float(c.iloc[-1])
    if last <= 0:
        return None
    mom6 = last / float(c.iloc[-126]) - 1 if len(c) > 126 else 0.0
    mom3 = last / float(c.iloc[-63]) - 1 if len(c) > 63 else 0.0
    mom12 = last / float(c.iloc[-252]) - 1 if len(c) > 252 else None
    sma200 = float(c.iloc[-200:].mean())
    win = c.iloc[-252:]
    high52 = float(win.max())
    low52 = float(win.min())
    dist_high = last / high52 - 1 if high52 else 0.0
    pct_above_low = last / low52 - 1 if low52 else 0.0
    # Corrupt source data (missed split, vendor error) shows up as a single-
    # session discontinuity — a one-day ±45%+ jump. A large CUMULATIVE move is
    # not corruption: the genuine multi-month leaders (the NVDA-2023 kind) are
    # exactly what the screen exists to find and must never be dropped.
    d1 = c.pct_change().iloc[-252:].abs()
    bad_data = bool(len(d1) and float(d1.max()) > 0.45)
    return {"price": round(last, 2), "mom6": mom6, "mom3": mom3, "mom12": mom12,
            "accel": mom3 - mom6 / 2.0,
            "above_200dma": last > sma200, "dist_high": dist_high,
            "pct_above_low": pct_above_low,
            "rsi": _rsi(c), "rs": mom6 - bench_mom6,
            "earnings_gap": strategy.earnings_gap_drift(c), "bad_data": bad_data,
            **_volume_metrics(c, volume)}


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
              sector_of: dict | None = None, exclude: set | None = None,
              volumes: pd.DataFrame | None = None,
              min_dollar_vol_m: float = 3.0,
              min_price: float = 5.0) -> tuple[list[dict], dict]:
    """Rank the universe; return (ranked[:top], regime). The regime dict also
    carries 'sector_rs' (per-sector relative strength) for display.

    With ``volumes`` provided, names below the liquidity floor (median daily
    dollar volume < min_dollar_vol_m millions, or price < min_price) are dropped
    before ranking — essential once the universe includes small caps, so an
    untradable micro-name can never reach the shortlist or the Risk Governor."""
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
    rows, dropped, illiquid = [], 0, 0
    for sym in closes.columns:
        if sym in skip:
            continue
        vol = volumes[sym] if volumes is not None and sym in volumes.columns else None
        m = _metrics(closes[sym], bench_mom6, volume=vol)
        if m is None:
            continue
        if m.get("bad_data"):                 # corrupt source data — never rank it
            dropped += 1
            continue
        dv = m.get("dollar_vol_m")
        if (m["price"] < min_price
                or (dv is not None and dv < min_dollar_vol_m)):
            illiquid += 1                     # tradability floor (small caps)
            continue
        m["symbol"] = sym
        m["sector"] = sector_of.get(sym)
        m["score"] = strategy.composite_score(m, w, sector_rs)
        rows.append(m)
    rows.sort(key=lambda r: r["score"], reverse=True)
    regime["dropped_bad_data"] = dropped
    regime["dropped_illiquid"] = illiquid
    return rows[:top], regime
