"""The strategy brain: the edges that aim to beat the average investor.

Pure, testable functions (no IO, no network) used by the S&P 500 screen and the
walk-forward backtest:

  * regime- and memory-adaptive FACTOR WEIGHTS (the desk leans into what is
    working now and into the prevailing market regime),
  * a composite OPPORTUNITY SCORE blending relative strength, momentum, trend,
    proximity to highs, an earnings-gap (post-earnings-drift) proxy, and SECTOR
    rotation,
  * SECTOR strength ranking,
  * conviction-scaled PORTFOLIO CONSTRUCTION with per-name and per-sector caps and
    a regime-scaled gross-exposure budget,
  * a WALK-FORWARD backtest of the screen strategy vs the benchmark.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Base factor weights (risk-on). Relative strength dominates; earnings-gap drift
# (PEAD) and sector rotation are meaningful secondary edges.
BASE_WEIGHTS = {
    "rs": 2.0, "mom6": 1.0, "mom3": 0.5, "trend": 0.30,
    "near_high": 0.20, "earnings_gap": 0.80, "sector": 0.60,
}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def earnings_gap_drift(close: pd.Series, lookback: int = 30) -> float:
    """Post-earnings-drift proxy from price alone: the largest recent 1-day move
    (an earnings reaction shows as a gap) blended with how price has drifted
    since. Positive = a held/extended up-gap (bullish PEAD); negative = a down-gap
    or a faded pop. No earnings calendar needed, so it scales to 500 names free."""
    c = close.dropna()
    if len(c) < lookback + 3:
        return 0.0
    rets = c.pct_change()
    recent = rets.iloc[-lookback:]
    if recent.abs().max() != recent.abs().max():     # all-NaN guard
        return 0.0
    i = recent.abs().idxmax()
    gap = float(recent.loc[i])
    if abs(gap) < 0.04:                              # no material event move
        return 0.0
    after = c.loc[i:]
    drift = float(after.iloc[-1] / after.iloc[0] - 1) if len(after) > 1 else 0.0
    return _clamp(0.5 * gap + 0.5 * drift, -0.25, 0.35)


def sector_strength(closes: pd.DataFrame, sector_etfs: dict[str, str],
                    benchmark: str = "SPY", lookback: int = 126) -> dict[str, float]:
    """Per-sector relative strength vs the benchmark, from sector-ETF prices when
    available. Returns {sector: rs}. Empty if no ETFs are present."""
    if benchmark not in closes.columns:
        return {}
    b = closes[benchmark].dropna()
    if len(b) <= lookback:
        return {}
    bench_mom = float(b.iloc[-1] / b.iloc[-lookback] - 1)
    out = {}
    for sector, etf in sector_etfs.items():
        if etf in closes.columns:
            s = closes[etf].dropna()
            if len(s) > lookback:
                out[sector] = float(s.iloc[-1] / s.iloc[-lookback] - 1) - bench_mom
    return out


def factor_weights(regime: dict | None = None, memory=None) -> dict:
    """Adapt the base weights to (a) the market regime and (b) what the desk's own
    recent realized trades say is working. Advisory/heuristic; never changes risk
    limits."""
    w = dict(BASE_WEIGHTS)
    risk_off = bool(regime) and regime.get("available") and not regime.get("above_200dma")
    if risk_off:
        # In a downtrend, trust trend/momentum chasing less; keep PEAD.
        w["rs"] *= 0.75
        w["mom6"] *= 0.60
        w["mom3"] *= 0.50
        w["near_high"] *= 0.50
        w["trend"] *= 1.30                            # demand the 200-DMA even more
    if memory is not None:
        try:
            stats = memory.setup_stats("confluence_swing")
            n, wr = stats.get("count", 0), stats.get("win_rate_pct", 50)
            if n >= 10:
                if wr < 40:                          # momentum picks aren't paying
                    w["rs"] *= 0.80
                    w["mom6"] *= 0.80
                elif wr > 58:                        # they are — lean in
                    w["rs"] *= 1.15
                    w["mom6"] *= 1.10
        except Exception:
            pass
    return w


def composite_score(m: dict, weights: dict | None = None,
                    sector_rs: dict[str, float] | None = None) -> float:
    """Blend a name's metrics into one opportunity score (higher = more promising)."""
    w = weights or BASE_WEIGHTS
    sec_rs = sector_rs or {}
    s = 0.0
    s += w["rs"] * _clamp(m.get("rs", 0.0), -0.5, 0.5)
    s += w["mom6"] * _clamp(m.get("mom6", 0.0), -0.5, 0.5)
    s += w["mom3"] * _clamp(m.get("mom3", 0.0), -0.3, 0.3)
    s += w["trend"] * (1.0 if m.get("above_200dma") else -1.0)
    dh = m.get("dist_high", -1.0)
    if dh > -0.08:
        s += w["near_high"]
    elif dh < -0.30:
        s -= w["near_high"]
    s += w["earnings_gap"] * m.get("earnings_gap", 0.0)
    sec = m.get("sector")
    if sec and sec in sec_rs:
        s += w["sector"] * _clamp(sec_rs[sec], -0.3, 0.3)
    rsi = m.get("rsi", float("nan"))
    if rsi == rsi:
        if rsi > 80:
            s -= 0.45
        elif rsi > 72:
            s -= 0.15
        elif rsi < 35:
            s -= 0.10
    return round(s, 4)


def construct_portfolio(recs: list[dict], equity: float, regime: dict | None = None,
                        max_name: float = 0.25, max_sector: float = 0.45) -> list[dict]:
    """Conviction-scaled portfolio with per-name / per-sector caps and a
    regime-scaled gross-exposure budget. ``recs`` each need symbol, entry,
    conviction (and optionally sector). Returns sized allocations."""
    buys = [r for r in recs if r.get("entry")]
    if not buys:
        return []
    gross = 1.0
    if regime and regime.get("available") and not regime.get("above_200dma"):
        gross = 0.50                                  # risk-off: half invested
    raw = {r["symbol"]: max(0.05, float(r.get("conviction", 0.0))) for r in buys}
    total = sum(raw.values()) or 1.0
    weights = {s: min(max_name, v / total) for s, v in raw.items()}
    # Enforce per-sector cap.
    by_sector: dict[str, list[str]] = {}
    for r in buys:
        by_sector.setdefault(r.get("sector") or "?", []).append(r["symbol"])
    for sec, syms in by_sector.items():
        tot = sum(weights[s] for s in syms)
        if sec != "?" and tot > max_sector:
            scale = max_sector / tot
            for s in syms:
                weights[s] *= scale
    # Do NOT re-normalize up to 1 — that would breach the per-name/sector caps.
    # Concentration-limited weights simply leave the remainder in cash. Only scale
    # DOWN to the regime gross-exposure budget if the caps still sum above it.
    invested = sum(weights.values())
    if invested > gross and invested > 0:
        weights = {s: w * gross / invested for s, w in weights.items()}
    out = []
    for r in buys:
        sym = r["symbol"]
        w = weights[sym]
        dollars = w * equity
        entry = float(r["entry"]) or 1.0
        out.append({"symbol": sym, "weight_pct": round(w * 100, 1),
                    "dollars": round(dollars, 0), "shares": int(dollars // entry),
                    "sector": r.get("sector") or "?",
                    "conviction": round(float(r.get("conviction", 0.0)), 2)})
    out.sort(key=lambda x: x["weight_pct"], reverse=True)
    return out


def _ann(returns: list[float], periods_per_year: float) -> dict:
    if not returns:
        return {"cagr_pct": 0.0, "sharpe": 0.0}
    arr = np.array(returns, dtype=float)
    growth = float(np.prod(1 + arr))
    n_years = len(arr) / periods_per_year if periods_per_year else 1
    cagr = growth ** (1 / n_years) - 1 if n_years > 0 and growth > 0 else 0.0
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = (float(arr.mean()) / sd * math.sqrt(periods_per_year)) if sd else 0.0
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(sharpe, 2)}


def walk_forward_backtest(closes: pd.DataFrame, metrics_fn, weights: dict,
                          top_k: int = 10, hold_days: int = 10,
                          benchmark: str = "SPY", warmup: int = 210) -> dict:
    """Honest walk-forward of the screen: every `hold_days` sessions, rank the
    universe by the composite score using ONLY history up to that day, buy the
    top-K equal-weight, hold for the period, record the realized return. Compares
    to the benchmark over the same span. `metrics_fn(close_slice, bench_mom6)`
    returns the metrics dict for one name (see app.screen._metrics)."""
    cols = [c for c in closes.columns if c != benchmark]
    dates = closes.index
    if len(dates) <= warmup + hold_days:
        return {"error": "not enough history"}
    period_rets, bench_rets, wins = [], [], 0
    i = warmup
    while i + hold_days < len(dates):
        hist = closes.iloc[: i + 1]
        bench_mom6 = 0.0
        if benchmark in hist.columns and len(hist) > 126:
            b = hist[benchmark].dropna()
            bench_mom6 = float(b.iloc[-1] / b.iloc[-126] - 1) if len(b) > 126 else 0.0
        scored = []
        for sym in cols:
            m = metrics_fn(hist[sym], bench_mom6)
            if m is None:
                continue
            scored.append((sym, composite_score(m, weights)))
        scored.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in scored[:top_k]]
        if picks:
            fwd = []
            for s in picks:
                p0 = closes[s].iloc[i]
                p1 = closes[s].iloc[i + hold_days]
                if p0 == p0 and p1 == p1 and p0 > 0:
                    fwd.append(p1 / p0 - 1)
            if fwd:
                r = float(np.mean(fwd))
                period_rets.append(r)
                wins += sum(1 for x in fwd if x > 0)
        if benchmark in closes.columns:
            b0, b1 = closes[benchmark].iloc[i], closes[benchmark].iloc[i + hold_days]
            if b0 == b0 and b1 == b1 and b0 > 0:
                bench_rets.append(b1 / b0 - 1)
        i += hold_days
    n_picks = sum(min(top_k, 1) for _ in period_rets)  # periods with picks
    ppy = 252 / hold_days
    strat = _ann(period_rets, ppy)
    bench = _ann(bench_rets, ppy)
    total = float(np.prod([1 + r for r in period_rets]) - 1) if period_rets else 0.0
    btot = float(np.prod([1 + r for r in bench_rets]) - 1) if bench_rets else 0.0
    # Max drawdown of the strategy equity curve.
    eq = np.cumprod([1 + r for r in period_rets]) if period_rets else np.array([1.0])
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100) if len(eq) else 0.0
    total_names = max(1, len(period_rets) * top_k)
    return {
        "periods": len(period_rets), "hold_days": hold_days, "top_k": top_k,
        "strategy_total_return_pct": round(total * 100, 1),
        "benchmark_total_return_pct": round(btot * 100, 1),
        "excess_return_pct": round((total - btot) * 100, 1),
        "strategy_cagr_pct": strat["cagr_pct"], "strategy_sharpe": strat["sharpe"],
        "benchmark_cagr_pct": bench["cagr_pct"], "benchmark_sharpe": bench["sharpe"],
        "win_rate_pct": round(100.0 * wins / total_names, 0),
        "max_drawdown_pct": round(mdd, 1),
    }
