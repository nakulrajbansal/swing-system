"""Assemble domain-specific evidence for the agents (master §6/§9).

The agents can only reason as well as the evidence they're given. This gathers,
for one candidate symbol, the concrete data each specialist's expertise needs —
technical state, the actual filing-text change (with a readable snippet), recent
8-K events, and insider purchases — into a compact, JSON-serializable packet that
the Hypothesis / Skeptic / Portfolio Manager reason over. Bounded for token cost.
"""

from __future__ import annotations

import pandas as pd

from system.data_plane import indicators as ind


def _tok(text) -> list[str]:
    return str(text).split() if text is not None else []


def _technicals(view, symbol: str) -> dict:
    px = view.prices(symbol, adjust=True)
    if len(px) < 30:
        return {"available": False}
    close = px["close"]
    last = float(close.iloc[-1])
    win = close.iloc[-252:]
    high52 = float(win.max())
    low52 = float(win.min())
    atr = ind.last_atr(px, 14)
    rsi = ind.rsi(close, 14).dropna()
    sma200 = ind.sma(close, 200).dropna()
    mom126 = float(last / close.iloc[-126] - 1) if len(close) > 126 else None
    return {
        "available": True,
        "price": round(last, 2),
        "pct_below_52wk_high": round((last / high52 - 1) * 100, 1) if high52 else None,
        "pct_above_52wk_low": round((last / low52 - 1) * 100, 1) if low52 else None,
        "momentum_6mo_pct": round(mom126 * 100, 1) if mom126 is not None else None,
        "atr_pct_of_price": round(atr / last * 100, 1) if last else None,
        "rsi14": round(float(rsi.iloc[-1]), 0) if len(rsi) else None,
        "above_200dma": bool(len(sma200) and last > float(sma200.iloc[-1])),
    }


def _filings(view, symbol: str) -> dict:
    f = view.filings(symbol)
    if f.empty:
        return {"available": False}
    periodic = f[f["form_type"].isin(["10-K", "10-Q"])].sort_values("available_at")
    eightk = f[f["form_type"] == "8-K"].sort_values("available_at")
    out: dict = {"available": True,
                 "recent_8k_count": int(len(eightk)),
                 "last_8k_date": str(eightk["available_at"].iloc[-1])[:10] if len(eightk) else None}
    if len(periodic) >= 1:
        cur = periodic.iloc[-1]
        out["latest_periodic"] = {"form": cur["form_type"],
                                  "date": str(cur["available_at"])[:10]}
        cur_txt = cur.get("section_text_riskfactors")
        if cur_txt:
            # Readable snippet of the actual risk-factor text (bounded).
            out["risk_text_snippet"] = " ".join(_tok(cur_txt)[:300])
        # Compare LIKE-FOR-LIKE: latest filing vs the prior filing of the SAME form
        # type (10-Q vs prior 10-Q), not 10-Q vs 10-K (a length artifact).
        same = periodic[periodic["form_type"] == cur["form_type"]]
        if len(same) >= 2 and cur_txt:
            prev = same.iloc[-2]
            cur_len = len(set(_tok(cur_txt)))
            prev_len = len(set(_tok(prev.get("section_text_riskfactors")))) or 1
            out["text_change_vs_prior_pct"] = round((cur_len - prev_len) / prev_len * 100, 1)
    return out


def _insider(view, symbol: str) -> dict:
    f4 = view.form4(symbol)
    if f4.empty:
        return {"available": False, "recent_purchases": []}
    buys = f4[f4["txn_code"] == "P"].sort_values("available_at").tail(6)
    return {
        "available": True,
        "recent_purchases": [
            {"date": str(r["available_at"])[:10], "role": r["insider_role"],
             "shares": int(r["shares"]), "value": round(float(r["value"]), 0)}
            for _, r in buys.iterrows()
        ],
    }


def _news(view, symbol: str) -> list[str]:
    n = view.news(symbol)
    if n.empty:
        return []
    return [str(h) for h in n.sort_values("available_at")["headline"].tail(5)]


def assemble_evidence(view, symbol: str) -> dict:
    """Compact, readable, domain-specific evidence packet for one symbol."""
    return {
        "symbol": symbol,
        "as_of": str(view.asof_date.date()),
        "technicals": _technicals(view, symbol),
        "filings": _filings(view, symbol),
        "insider": _insider(view, symbol),
        "recent_news": _news(view, symbol),
    }
