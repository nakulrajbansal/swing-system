"""Market-based macro snapshot (free, via yfinance ETFs/indices).

A compact read of the economic backdrop that matters for swing-long equity risk:
the equity regime, volatility (VIX), rates (long-bond trend), credit risk
appetite (high-yield vs investment-grade), the dollar, commodities, and
cyclical-vs-defensive leadership. Price-based only (no unit pitfalls, no API key).
Pure compute + one batched download; the app layer caches the result daily.
"""

from __future__ import annotations

import pandas as pd

# Index/ETF proxies for each macro driver.
_TICKERS = ["SPY", "^VIX", "TLT", "HYG", "LQD", "UUP", "USO", "GLD", "XLY", "XLP"]


def _ret(closes: pd.DataFrame, sym: str, n: int):
    if sym not in closes.columns:
        return None
    s = closes[sym].dropna()
    if len(s) <= n:
        return None
    return float(s.iloc[-1] / s.iloc[-n] - 1)


def _last(closes: pd.DataFrame, sym: str):
    if sym not in closes.columns:
        return None
    s = closes[sym].dropna()
    return float(s.iloc[-1]) if len(s) else None


def fetch_macro_snapshot(emit=print) -> dict:
    """Return a structured macro snapshot, or {'available': False} on failure."""
    from harness.data.loader import fetch_closes_batch  # lazy
    import datetime

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=320)).isoformat()
    try:
        closes = fetch_closes_batch(_TICKERS, start, today.isoformat(), emit=emit)
    except Exception as exc:
        emit(f"[macro] fetch failed: {exc}")
        return {"available": False}
    if closes.empty or "SPY" not in closes.columns:
        return {"available": False}

    spy = closes["SPY"].dropna()
    spy_sma200 = float(spy.iloc[-200:].mean()) if len(spy) >= 200 else float(spy.mean())
    equity_up = float(spy.iloc[-1]) > spy_sma200
    spy_mom6 = _ret(closes, "SPY", 126) or 0.0

    vix = _last(closes, "^VIX")
    vix_state = ("unavailable" if vix is None else
                 "calm" if vix < 16 else "elevated" if vix < 24 else "stressed")

    tlt_3m = _ret(closes, "TLT", 63)            # bonds up => rates falling (supportive)
    rates_trend = ("unavailable" if tlt_3m is None else
                   "falling" if tlt_3m > 0.02 else "rising" if tlt_3m < -0.02 else "stable")
    hyg_3m, lqd_3m = _ret(closes, "HYG", 63), _ret(closes, "LQD", 63)
    credit_spread = (hyg_3m - lqd_3m) if (hyg_3m is not None and lqd_3m is not None) else None
    credit_state = ("unavailable" if credit_spread is None else
                    "risk-on" if credit_spread > 0 else "risk-off")
    uup_3m = _ret(closes, "UUP", 63)
    usd_trend = ("unavailable" if uup_3m is None else
                 "strengthening" if uup_3m > 0.03 else "weakening" if uup_3m < -0.03 else "flat")
    oil_3m, gold_3m = _ret(closes, "USO", 63), _ret(closes, "GLD", 63)
    xly_3m, xlp_3m = _ret(closes, "XLY", 63), _ret(closes, "XLP", 63)
    cyc = (xly_3m - xlp_3m) if (xly_3m is not None and xlp_3m is not None) else None
    cyc_state = ("unavailable" if cyc is None else
                 "cyclicals leading (risk-on)" if cyc > 0 else "defensives leading (risk-off)")

    # Backdrop score for swing-long equity risk.
    sc = 0.0
    sc += 1.0 if equity_up else -1.0
    sc += 0.5 if spy_mom6 > 0 else -0.5
    if vix is not None:
        sc += 1.0 if vix < 16 else 0.0 if vix < 22 else -0.6 if vix < 28 else -1.5
    if credit_spread is not None:
        sc += 0.8 if credit_spread > 0 else -0.8
    if tlt_3m is not None:
        sc += 0.4 if tlt_3m > 0.02 else -0.4 if tlt_3m < -0.02 else 0.0
    if uup_3m is not None:
        sc += -0.4 if uup_3m > 0.03 else 0.2 if uup_3m < -0.03 else 0.0
    if cyc is not None:
        sc += 0.5 if cyc > 0 else -0.5
    backdrop = "supportive" if sc >= 1.5 else "hostile" if sc <= -1.0 else "neutral"

    def pct(x):
        return round(x * 100, 1) if isinstance(x, (int, float)) else None

    return {
        "available": True,
        "as_of": today.isoformat(),
        "equity_regime": "uptrend" if equity_up else "downtrend",
        "spy_mom6_pct": pct(spy_mom6),
        "vix": round(vix, 1) if vix is not None else None,
        "vix_state": vix_state,
        "rates_trend": rates_trend,
        "credit": credit_state,
        "usd_trend": usd_trend,
        "oil_3mo_pct": pct(oil_3m), "gold_3mo_pct": pct(gold_3m),
        "cyclical_vs_defensive": cyc_state,
        "backdrop": backdrop,
        "score": round(sc, 1),
        "summary": (f"{backdrop.upper()} backdrop: equity {('uptrend' if equity_up else 'downtrend')}, "
                    f"VIX {vix:.0f} ({vix_state}), rates {rates_trend}, credit {credit_state}, "
                    f"USD {usd_trend}, {cyc_state}." if vix is not None else
                    f"{backdrop.upper()} backdrop (partial data)."),
    }
