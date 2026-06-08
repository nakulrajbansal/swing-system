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
from system.agents.prompts import FUNDAMENTAL_ANALYST, TECHNICAL_ANALYST
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
