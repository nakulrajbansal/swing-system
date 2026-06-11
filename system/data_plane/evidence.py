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
    pct_high = round((last / high52 - 1) * 100, 1) if high52 else None
    pct_low = round((last / low52 - 1) * 100, 1) if low52 else None
    mom = round(mom126 * 100, 1) if mom126 is not None else None
    out = {
        "available": True,
        "price": round(last, 2),
        "pct_below_52wk_high": pct_high,
        "pct_above_52wk_low": pct_low,
        "momentum_6mo_pct": mom,
        "atr_pct_of_price": round(atr / last * 100, 1) if last else None,
        "rsi14": round(float(rsi.iloc[-1]), 0) if len(rsi) else None,
        "above_200dma": bool(len(sma200) and last > float(sma200.iloc[-1])),
    }
    # Plausibility guard: corrupt source data (a missed split, vendor error)
    # shows up as a single-session discontinuity, NOT as a large cumulative
    # move — genuine multi-month leaders rally hundreds of percent and must not
    # be smeared as bad data. Flag only true discontinuities.
    warns = []
    d1 = close.pct_change().iloc[-252:].abs()
    jump = float(d1.max()) if len(d1) else 0.0
    if jump > 0.45:
        warns.append(f"a {jump * 100:.0f}% single-session move suggests a "
                     "split/data error in the price series")
    if out["atr_pct_of_price"] is not None and out["atr_pct_of_price"] > 20:
        warns.append(f"ATR {out['atr_pct_of_price']:.0f}% of price is implausibly high")
    if warns:
        out["data_quality_warning"] = "; ".join(warns) + " - likely bad price data; treat with caution"
    return out


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


def _f(row, key):
    v = row.get(key)
    if v is None:
        return None
    try:
        fv = float(v)
        return None if fv != fv else round(fv, 3)
    except (TypeError, ValueError):
        return v


def _fundamentals(view, symbol: str) -> dict:
    """Latest PIT-visible valuation / growth / guidance snapshot for the symbol.

    forward_pe and forward_eps embed the analyst-consensus forward view (a proxy
    for company guidance); target_mean_price is the Street's price guidance. The
    valuation and growth analysts reason over this block.
    """
    fund = view.fundamentals(symbol) if hasattr(view, "fundamentals") else None
    if fund is None or fund.empty:
        return {"available": False}
    r = fund.sort_values("available_at").iloc[-1].to_dict()
    price = _f(r, "price") or None
    tgt = _f(r, "target_mean_price")
    fwd_eps, tr_eps = _f(r, "forward_eps"), _f(r, "trailing_eps")
    implied_eps_growth = None
    eps_reliable = True
    if fwd_eps is not None and tr_eps not in (None, 0):
        implied_eps_growth = round((fwd_eps / tr_eps - 1) * 100, 1)
        # A tiny/negative trailing EPS or an enormous implied jump makes the ratio
        # an artifact, not a real growth signal — flag it so agents discount it.
        if tr_eps < 1.0 or abs(implied_eps_growth) > 80:
            eps_reliable = False
    fcf, rev = _f(r, "free_cash_flow"), _f(r, "total_revenue")
    fcf_margin = round(fcf / rev * 100, 1) if fcf is not None and rev else None
    mc = _f(r, "market_cap")
    summary = r.get("business_summary")
    return {
        "available": True,
        "valuation": {
            "trailing_pe": _f(r, "trailing_pe"),
            "forward_pe": _f(r, "forward_pe"),
            "price_to_sales": _f(r, "price_to_sales"),
            "price_to_book": _f(r, "price_to_book"),
            "peg_ratio": _f(r, "peg_ratio"),
            "ev_to_ebitda": _f(r, "ev_to_ebitda"),
            "analyst_target_price": tgt,
        },
        "growth": {
            "revenue_growth_pct": round(_f(r, "revenue_growth") * 100, 1)
            if _f(r, "revenue_growth") is not None else None,
            "earnings_growth_pct": round(_f(r, "earnings_growth") * 100, 1)
            if _f(r, "earnings_growth") is not None else None,
            "earnings_q_growth_pct": round(_f(r, "earnings_q_growth") * 100, 1)
            if _f(r, "earnings_q_growth") is not None else None,
            "trailing_eps": tr_eps,
            "forward_eps_guidance": fwd_eps,
            "implied_fwd_eps_growth_pct": implied_eps_growth,
            "fwd_eps_reliable": eps_reliable,
            "profit_margin_pct": round(_f(r, "profit_margin") * 100, 1)
            if _f(r, "profit_margin") is not None else None,
            "return_on_equity_pct": round(_f(r, "return_on_equity") * 100, 1)
            if _f(r, "return_on_equity") is not None else None,
        },
        # Business-quality / durability block: what the moat & secular-trend
        # analyst reasons over (margins = pricing power, FCF = self-funding,
        # the description = which structural theme the company is levered to).
        "moat": {
            "gross_margin_pct": round(_f(r, "gross_margin") * 100, 1)
            if _f(r, "gross_margin") is not None else None,
            "operating_margin_pct": round(_f(r, "operating_margin") * 100, 1)
            if _f(r, "operating_margin") is not None else None,
            "fcf_margin_pct": fcf_margin,
            "return_on_assets_pct": round(_f(r, "return_on_assets") * 100, 1)
            if _f(r, "return_on_assets") is not None else None,
            "market_cap_bn": round(mc / 1e9, 1) if mc else None,
            "insider_ownership_pct": round(_f(r, "held_percent_insiders") * 100, 1)
            if _f(r, "held_percent_insiders") is not None else None,
            "industry": r.get("industry"),
            "business_summary": str(summary)[:700] if summary else None,
        },
        "analyst_recommendation": r.get("recommendation"),
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
        "fundamentals": _fundamentals(view, symbol),
        "filings": _filings(view, symbol),
        "insider": _insider(view, symbol),
        "recent_news": _news(view, symbol),
    }
