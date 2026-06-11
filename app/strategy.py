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
    "near_high": 0.20, "earnings_gap": 0.80, "sector": 0.60, "accel": 0.60,
    "volume": 0.30,
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
    # Momentum ACCELERATION: is the recent 3-month pace running ahead of the
    # 6-month average pace? An igniting trend scores before it is consensus.
    s += w.get("accel", 0.0) * _clamp(m.get("accel", 0.0), -0.3, 0.4)
    # Accumulation: volume concentrated on up days (institutions leave volume
    # footprints before the price move is obvious). Neutral when volume absent.
    uv = m.get("updown_vol")
    if isinstance(uv, (int, float)):
        s += w.get("volume", 0.0) * _clamp(uv - 1.0, -0.5, 0.5)
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


def hidden_gem_score(m: dict) -> float:
    """Early-trend score: how much a name looks like a future leader BEFORE the
    crowd owns it. Favors acceleration (the 3-month pace overtaking the 6-month
    average pace), a held earnings gap, an intact trend, a 12-month record that
    is NOT yet a consensus winner, and room left below the prior high. A pure
    momentum ranking only surfaces names after the rally is obvious; this is the
    complementary lens."""
    if not m.get("above_200dma"):
        return -1.0                        # igniting, not a falling knife
    accel = m.get("accel", 0.0)
    s = 2.0 * _clamp(accel, -0.3, 0.4)
    s += 1.0 * m.get("earnings_gap", 0.0)
    s += 0.5 * _clamp(m.get("mom3", 0.0), -0.3, 0.5)
    if accel > 0.05:                       # bonuses only when genuinely igniting
        mom12 = m.get("mom12")
        if mom12 is not None and mom12 < 0.35:
            s += 0.25                      # not already a crowded 12-month story
        dh = m.get("dist_high", 0.0)
        if -0.35 <= dh <= -0.10:
            s += 0.20                      # re-rating room back to / through highs
        vr = m.get("vol_ratio")
        if isinstance(vr, (int, float)) and vr > 1.3:
            s += 0.20                      # volume expanding into the move
        uv = m.get("updown_vol")
        if isinstance(uv, (int, float)) and uv > 1.3:
            s += 0.15                      # up-day volume dominates: accumulation
    rsi = m.get("rsi", float("nan"))
    if rsi == rsi and rsi > 80:
        s -= 0.30
    return round(s, 4)


def select_shortlist(ranked: list[dict], k: int, gem_slots: int = 2) -> list[dict]:
    """The deep-dive shortlist: top names by composite score PLUS reserved
    'hidden gem' slots for early-acceleration names the momentum ranking alone
    would miss. Gems must clear a positive bar (marked ``hidden_gem=True``);
    unused slots fall back to score order, so the list is always full when the
    universe allows."""
    k = max(1, k)
    base = [m for m in ranked if m["score"] > 0]
    core = base[: max(1, k - gem_slots)]
    chosen = {m["symbol"] for m in core}
    gems = sorted((m for m in ranked if m["symbol"] not in chosen),
                  key=hidden_gem_score, reverse=True)
    out = list(core)
    for g in gems:
        if len(out) >= k:
            break
        if hidden_gem_score(g) > 0.15:
            g = dict(g)
            g["hidden_gem"] = True
            out.append(g)
    for m in base:                          # backfill if too few gems qualified
        if len(out) >= k:
            break
        if m["symbol"] not in {x["symbol"] for x in out}:
            out.append(m)
    return out


def diversify(ranked: list[dict], k: int, max_per_sector: int = 2) -> list[dict]:
    """Pick the top-k for the deep-dive while capping names per sector, so a single
    hot sector cannot monopolize the shortlist (breadth beats concentration). Falls
    back to filling any remaining slots in score order if sectors run out."""
    out, per = [], {}
    for m in ranked:
        sec = m.get("sector") or "?"
        if per.get(sec, 0) >= max_per_sector:
            continue
        out.append(m)
        per[sec] = per.get(sec, 0) + 1
        if len(out) >= k:
            return out
    for m in ranked:                          # backfill if breadth left slots open
        if m not in out:
            out.append(m)
            if len(out) >= k:
                break
    return out


def construct_portfolio(recs: list[dict], equity: float, regime: dict | None = None,
                        max_name: float = 0.25, max_sector: float = 0.65) -> list[dict]:
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
    gem_rets, gem_wins, gem_n = [], 0, 0
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
            scored.append((sym, composite_score(m, weights), m))
        scored.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _, _ in scored[:top_k]]

        def _fwd(syms):
            out = []
            for s in syms:
                p0 = closes[s].iloc[i]
                p1 = closes[s].iloc[i + hold_days]
                if p0 == p0 and p1 == p1 and p0 > 0:
                    out.append(p1 / p0 - 1)
            return out

        if picks:
            fwd = _fwd(picks)
            if fwd:
                period_rets.append(float(np.mean(fwd)))
                wins += sum(1 for x in fwd if x > 0)
        # Hidden-gem cohort, tracked separately: would the early-acceleration
        # lens have paid? (Same walk-forward discipline: scored on history up to
        # day i only, returns measured forward.)
        gems = sorted(((s, m) for s, _, m in scored[top_k:]
                       if hidden_gem_score(m) > 0.15),
                      key=lambda x: hidden_gem_score(x[1]), reverse=True)[:3]
        gfwd = _fwd([s for s, _ in gems])
        if gfwd:
            gem_rets.append(float(np.mean(gfwd)))
            gem_wins += sum(1 for x in gfwd if x > 0)
            gem_n += len(gfwd)
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
    gem_total = float(np.prod([1 + r for r in gem_rets]) - 1) if gem_rets else 0.0
    return {
        "gem_periods": len(gem_rets), "gem_picks": gem_n,
        "gem_total_return_pct": round(gem_total * 100, 1),
        "gem_win_rate_pct": round(100.0 * gem_wins / gem_n, 0) if gem_n else 0.0,
        "periods": len(period_rets), "hold_days": hold_days, "top_k": top_k,
        "strategy_total_return_pct": round(total * 100, 1),
        "benchmark_total_return_pct": round(btot * 100, 1),
        "excess_return_pct": round((total - btot) * 100, 1),
        "strategy_cagr_pct": strat["cagr_pct"], "strategy_sharpe": strat["sharpe"],
        "benchmark_cagr_pct": bench["cagr_pct"], "benchmark_sharpe": bench["sharpe"],
        "win_rate_pct": round(100.0 * wins / total_names, 0),
        "max_drawdown_pct": round(mdd, 1),
    }
