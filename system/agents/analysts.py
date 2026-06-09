"""Domain analyst agents (master §6): a Technical analyst and a Fundamental
analyst that each produce a written, swing-trade-oriented read over the evidence
packet. They run BEFORE the Hypothesis agent and their reads are both shown in
the deliberation and fed to the trio — so the technical and financial analysis is
visible work by named agents, not a hidden number in the evidence blob.

Deterministic mode grounds each read in the same numeric evidence (so offline/CI
runs still produce coherent reads); with a real client the same prompt/schema
drives the LLM.
"""

from __future__ import annotations

from system.agents.base import Agent
from system.agents.prompts import (
    FUNDAMENTAL_ANALYST,
    GROWTH_ANALYST,
    TECHNICAL_ANALYST,
    VALUATION_ANALYST,
)
from system.schemas import AnalystRead


def _f(raw: dict, key: str, default=0.0) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _lst(x) -> list[str]:
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return [str(x)] if x and str(x).strip() else []


def _stance(score: float) -> str:
    return "bullish" if score >= 0.6 else "bearish" if score <= 0.4 else "neutral"


class TechnicalAnalyst(Agent):
    name = "technical_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, TECHNICAL_ANALYST, max_tokens=600)

    def deterministic(self, inputs: dict) -> AnalystRead:
        t = inputs.get("evidence", {}).get("technicals", {})
        if not t.get("available"):
            return AnalystRead("technical", "neutral", 0.5,
                               "Insufficient price history for a technical read.",
                               [], ["thin price history"])
        pos, con, score = [], [], 0.5
        if t.get("above_200dma"):
            pos.append("price is above its 200-DMA (uptrend)"); score += 0.12
        else:
            con.append("price is below its 200-DMA (downtrend)"); score -= 0.15
        m = t.get("momentum_6mo_pct")
        if isinstance(m, (int, float)):
            if m > 10:
                pos.append(f"strong 6-month momentum ({m:+.0f}%)"); score += 0.12
            elif m < -10:
                con.append(f"weak 6-month momentum ({m:+.0f}%)"); score -= 0.12
        p = t.get("pct_below_52wk_high")
        if isinstance(p, (int, float)):
            if p > -8:
                pos.append(f"trading near its 52-week high ({p:+.0f}%)"); score += 0.08
            elif p < -30:
                con.append(f"{abs(p):.0f}% below its 52-week high (falling-knife risk)"); score -= 0.08
        rsi = t.get("rsi14")
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                con.append(f"RSI {rsi:.0f} is overbought (poor entry)"); score -= 0.05
            elif rsi < 30:
                pos.append(f"RSI {rsi:.0f} is oversold (possible bounce)"); score += 0.05
        atrp = t.get("atr_pct_of_price")
        if isinstance(atrp, (int, float)) and atrp > 6:
            con.append(f"high volatility (ATR {atrp:.1f}% of price)")
        score = max(0.0, min(1.0, score))
        regime = "an uptrend" if t.get("above_200dma") else "a downtrend"
        assess = (f"At ${t.get('price')} the stock is in {regime} with "
                  f"{m:+.0f}% 6-month momentum and RSI {rsi:.0f}." if isinstance(m, (int, float))
                  and isinstance(rsi, (int, float)) else f"At ${t.get('price')} the stock is in {regime}.")
        return AnalystRead("technical", _stance(score), score, assess, pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("technical", stance, score, str(raw.get("assessment", "")).strip(),
                           _lst(raw.get("positives")), _lst(raw.get("concerns")))


class FundamentalAnalyst(Agent):
    name = "fundamental_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, FUNDAMENTAL_ANALYST, max_tokens=600)

    def deterministic(self, inputs: dict) -> AnalystRead:
        ev = inputs.get("evidence", {})
        f = ev.get("filings", {})
        ins = ev.get("insider", {})
        if not f.get("available"):
            return AnalystRead("fundamental", "neutral", 0.5,
                               "No recent filings available to analyze.", [],
                               ["no recent filings"])
        pos, con, score = [], [], 0.5
        buys = (ins or {}).get("recent_purchases", [])
        if buys:
            tot = sum(b.get("value", 0) for b in buys)
            pos.append(f"{len(buys)} insider purchase(s) (~${tot:,.0f}) - informed buying")
            score += 0.12 if tot > 100_000 else 0.06
        chg = f.get("text_change_vs_prior_pct")
        if isinstance(chg, (int, float)):
            if abs(chg) > 25:
                con.append(f"risk-factor text changed {chg:+.0f}% vs the prior filing "
                           "(possible new disclosures)"); score -= 0.08
            else:
                pos.append(f"risk-factor text broadly stable ({chg:+.0f}% vs prior)")
        n8k = f.get("recent_8k_count", 0)
        if n8k >= 8:
            con.append(f"heavy 8-K cadence ({n8k} recently) — elevated event risk"); score -= 0.05
        lp = f.get("latest_periodic", {})
        score = max(0.0, min(1.0, score))
        assess = (f"Latest report {lp.get('form', 'filing')} ({lp.get('date', '?')}); "
                  f"{len(buys)} recent insider purchase(s); {n8k} recent 8-K(s).")
        return AnalystRead("fundamental", _stance(score), score, assess, pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("fundamental", stance, score, str(raw.get("assessment", "")).strip(),
                           _lst(raw.get("positives")), _lst(raw.get("concerns")))


class ValuationAnalyst(Agent):
    name = "valuation_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, VALUATION_ANALYST, max_tokens=600)

    def deterministic(self, inputs: dict) -> AnalystRead:
        fu = inputs.get("evidence", {}).get("fundamentals", {})
        if not fu.get("available"):
            return AnalystRead("valuation", "neutral", 0.5,
                               "No valuation multiples available.", [], ["no fundamentals data"])
        v = fu.get("valuation", {})
        pos, con, score = [], [], 0.5
        fpe = v.get("forward_pe")
        if isinstance(fpe, (int, float)):
            if fpe <= 0:
                con.append("negative forward earnings (no meaningful P/E)")
            elif fpe < 15:
                pos.append(f"low forward P/E ({fpe:.0f}x) - inexpensive"); score += 0.12
            elif fpe > 35:
                con.append(f"rich forward P/E ({fpe:.0f}x) - expensive"); score -= 0.12
        peg = v.get("peg_ratio")
        if isinstance(peg, (int, float)) and peg > 0:
            if peg < 1:
                pos.append(f"PEG {peg:.2f} (<1) - cheap vs its growth"); score += 0.1
            elif peg > 2:
                con.append(f"PEG {peg:.2f} (>2) - expensive vs its growth"); score -= 0.1
        tgt = v.get("analyst_target_price")
        px = inputs.get("evidence", {}).get("technicals", {}).get("price")
        if isinstance(tgt, (int, float)) and isinstance(px, (int, float)) and px:
            up = (tgt / px - 1) * 100
            if up > 10:
                pos.append(f"analyst target ${tgt:.0f} is {up:+.0f}% above price"); score += 0.08
            elif up < -5:
                con.append(f"price is above the analyst target ${tgt:.0f} ({up:+.0f}%)"); score -= 0.08
        ps = v.get("price_to_sales")
        if isinstance(ps, (int, float)) and ps > 20:
            con.append(f"very high price/sales ({ps:.0f}x)")
        score = max(0.0, min(1.0, score))
        assess = (f"Forward P/E {fpe if fpe is not None else 'n/a'}, "
                  f"PEG {peg if peg is not None else 'n/a'}, "
                  f"P/S {ps if ps is not None else 'n/a'}.")
        return AnalystRead("valuation", _stance(score), score, assess, pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("valuation", stance, score, str(raw.get("assessment", "")).strip(),
                           _lst(raw.get("positives")), _lst(raw.get("concerns")))


class GrowthAnalyst(Agent):
    name = "growth_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, GROWTH_ANALYST, max_tokens=600)

    def deterministic(self, inputs: dict) -> AnalystRead:
        fu = inputs.get("evidence", {}).get("fundamentals", {})
        if not fu.get("available"):
            return AnalystRead("growth", "neutral", 0.5,
                               "No growth/guidance data available.", [], ["no fundamentals data"])
        g = fu.get("growth", {})
        pos, con, score = [], [], 0.5
        rg = g.get("revenue_growth_pct")
        if isinstance(rg, (int, float)):
            if rg > 15:
                pos.append(f"strong revenue growth ({rg:+.0f}%)"); score += 0.12
            elif rg < 0:
                con.append(f"revenue is shrinking ({rg:+.0f}%)"); score -= 0.12
        eg = g.get("earnings_growth_pct")
        if isinstance(eg, (int, float)):
            if eg > 15:
                pos.append(f"strong earnings growth ({eg:+.0f}%)"); score += 0.1
            elif eg < 0:
                con.append(f"earnings are declining ({eg:+.0f}%)"); score -= 0.1
        ig = g.get("implied_fwd_eps_growth_pct")
        if isinstance(ig, (int, float)):
            if ig > 5:
                pos.append(f"forward EPS guidance above trailing ({ig:+.0f}%) - guided up"); score += 0.12
            elif ig < -5:
                con.append(f"forward EPS guidance below trailing ({ig:+.0f}%) - guided down"); score -= 0.12
        roe = g.get("return_on_equity_pct")
        if isinstance(roe, (int, float)) and roe > 20:
            pos.append(f"high return on equity ({roe:.0f}%)")
        pm = g.get("profit_margin_pct")
        if isinstance(pm, (int, float)) and pm < 0:
            con.append(f"unprofitable (margin {pm:+.0f}%)"); score -= 0.05
        score = max(0.0, min(1.0, score))
        assess = (f"Revenue growth {rg if rg is not None else 'n/a'}%, earnings growth "
                  f"{eg if eg is not None else 'n/a'}%, forward-EPS guidance "
                  f"{('+' + str(ig) + '%') if isinstance(ig, (int, float)) else 'n/a'} vs trailing.")
        return AnalystRead("growth", _stance(score), score, assess, pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("growth", stance, score, str(raw.get("assessment", "")).strip(),
                           _lst(raw.get("positives")), _lst(raw.get("concerns")))
