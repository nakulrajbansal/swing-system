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
    MACRO_ANALYST,
    MOAT_ANALYST,
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


class MacroAnalyst(Agent):
    name = "macro_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, MACRO_ANALYST, max_tokens=600)

    def deterministic(self, inputs: dict) -> AnalystRead:
        mac = inputs.get("evidence", {}).get("macro", {})
        if not mac.get("available"):
            return AnalystRead("macro", "neutral", 0.5,
                               "No macro snapshot available.", [], ["no macro data"])
        pos, con = [], []
        if mac.get("equity_regime") == "uptrend":
            pos.append("equity index in an uptrend (above 200-DMA)")
        else:
            con.append("equity index below its 200-DMA (downtrend)")
        vs = mac.get("vix_state")
        if vs == "calm":
            pos.append(f"VIX calm ({mac.get('vix')})")
        elif vs == "stressed":
            con.append(f"VIX stressed ({mac.get('vix')})")
        elif vs == "elevated":
            con.append(f"VIX elevated ({mac.get('vix')})")
        if mac.get("credit") == "risk-on":
            pos.append("credit risk-on (high-yield leading)")
        elif mac.get("credit") == "risk-off":
            con.append("credit risk-off (high-yield lagging)")
        if mac.get("rates_trend") == "rising":
            con.append("long rates rising (a headwind for risk multiples)")
        elif mac.get("rates_trend") == "falling":
            pos.append("rates falling (supportive)")
        if mac.get("usd_trend") == "strengthening":
            con.append("US dollar strengthening (headwind)")
        if "cyclicals leading" in (mac.get("cyclical_vs_defensive") or ""):
            pos.append("cyclicals leading defensives (risk appetite)")
        elif "defensives leading" in (mac.get("cyclical_vs_defensive") or ""):
            con.append("defensives leading (risk aversion)")
        # Map the snapshot's backdrop score (~[-3,3]) to a 0..1 stance score.
        score = max(0.0, min(1.0, 0.5 + float(mac.get("score", 0)) / 6.0))
        return AnalystRead("macro", _stance(score), score,
                           mac.get("summary", "")[:400], pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("macro", stance, score, str(raw.get("assessment", "")).strip(),
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
        peg0 = v.get("peg_ratio")
        if isinstance(fpe, (int, float)):
            if fpe <= 0:
                con.append("negative forward earnings (no meaningful P/E)")
            elif fpe < 15:
                pos.append(f"low forward P/E ({fpe:.0f}x) - inexpensive"); score += 0.12
            elif fpe > 35:
                # Growth-adjust: a high multiple with a PEG near/below ~1.5 is a
                # compounder priced for its growth, not an expensive stock — the
                # raw-multiple reflex is how the future leaders get missed.
                if isinstance(peg0, (int, float)) and 0 < peg0 <= 1.5:
                    pos.append(f"forward P/E {fpe:.0f}x looks rich but PEG {peg0:.2f} "
                               "- the growth justifies the multiple")
                else:
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


# Deterministic-mode keyword map for secular tailwinds. The real thematic read
# comes from the LLM over the business summary; this keeps offline/CI mode
# coherent (and is deliberately conservative: theme + numbers, never theme alone).
_SECULAR_THEMES = [
    ("artificial intelligence", "AI"), ("machine learning", "AI"),
    ("gpu", "AI compute"), ("semiconductor", "AI/compute supply chain"),
    ("data center", "AI/cloud infrastructure"), ("cloud", "cloud migration"),
    ("cybersecurity", "cybersecurity"), ("electric vehicle", "electrification"),
    ("battery", "electrification"), ("renewable", "energy transition"),
    ("solar", "energy transition"), ("obesity", "GLP-1 therapeutics"),
    ("weight loss", "GLP-1 therapeutics"), ("automation", "automation/robotics"),
    ("robot", "automation/robotics"), ("aerospace and defense", "defense spending"),
]


class MoatAnalyst(Agent):
    name = "moat_analyst"
    schema_name = "AnalystRead"

    def __init__(self, client, model):
        super().__init__(client, model, MOAT_ANALYST, max_tokens=700)

    def deterministic(self, inputs: dict) -> AnalystRead:
        fu = inputs.get("evidence", {}).get("fundamentals", {})
        if not fu.get("available"):
            return AnalystRead("moat", "neutral", 0.5,
                               "No business-quality data available.", [],
                               ["no fundamentals data"])
        m = fu.get("moat", {}) or {}
        g = fu.get("growth", {}) or {}
        pos, con, score = [], [], 0.5
        gm = m.get("gross_margin_pct")
        if isinstance(gm, (int, float)):
            if gm >= 55:
                pos.append(f"high gross margin ({gm:.0f}%) - pricing power"); score += 0.12
            elif gm < 25:
                con.append(f"thin gross margin ({gm:.0f}%) - commodity economics"); score -= 0.10
        om = m.get("operating_margin_pct")
        if isinstance(om, (int, float)):
            if om >= 25:
                pos.append(f"strong operating margin ({om:.0f}%)"); score += 0.08
            elif om < 5:
                con.append(f"weak operating margin ({om:.0f}%)"); score -= 0.08
        fcf = m.get("fcf_margin_pct")
        if isinstance(fcf, (int, float)):
            if fcf >= 15:
                pos.append(f"free-cash-flow margin {fcf:.0f}% - self-funding compounder"); score += 0.08
            elif fcf < 0:
                con.append("negative free cash flow (depends on external funding)"); score -= 0.05
        roe = g.get("return_on_equity_pct")
        if isinstance(roe, (int, float)) and roe >= 25:
            pos.append(f"return on equity {roe:.0f}% - high returns on capital"); score += 0.06
        rg, eg = g.get("revenue_growth_pct"), g.get("earnings_growth_pct")
        if (isinstance(rg, (int, float)) and isinstance(eg, (int, float))
                and rg > 0 and eg > rg):
            pos.append(f"operating leverage: earnings ({eg:+.0f}%) outpacing revenue "
                       f"({rg:+.0f}%)"); score += 0.08
        if (isinstance(rg, (int, float)) and rg >= 25
                and isinstance(gm, (int, float)) and gm >= 50):
            pos.append("hyper-growth WITH high margins - the classic emerging-leader "
                       "profile"); score += 0.08
        # TRAJECTORY (EDGAR quarterly history): where the business is GOING.
        # Revenue growth accelerating while margins expand is the pre-rally
        # inflection fingerprint; a 2-quarter deceleration is the reverse.
        traj = fu.get("trajectory", {}) or {}
        if traj.get("available"):
            acc2 = traj.get("revenue_accelerating_2q")
            acc = traj.get("revenue_accelerating")
            mexp = traj.get("margins_expanding")
            yoy = traj.get("revenue_yoy_latest_pct")
            if acc and mexp:
                pos.append(f"INFLECTING: revenue growth accelerating "
                           f"({yoy:+.0f}% YoY latest) with expanding gross margins "
                           f"({traj.get('gross_margin_trend_pct', 0):+.1f}pp vs a year ago) "
                           "- the pre-rally fingerprint")
                score += 0.18 if acc2 else 0.12
            elif acc2:
                pos.append(f"revenue growth accelerating two quarters running "
                           f"(latest {yoy:+.0f}% YoY)"); score += 0.08
            if traj.get("revenue_decelerating_2q"):
                con.append("revenue growth decelerating two quarters running"); score -= 0.10
            if mexp is False and isinstance(traj.get("gross_margin_trend_pct"),
                                            (int, float)) \
                    and traj["gross_margin_trend_pct"] < -2:
                con.append(f"gross margin contracting "
                           f"({traj['gross_margin_trend_pct']:+.1f}pp vs a year ago) "
                           "- pricing power eroding"); score -= 0.08
        # Secular-tailwind scan: theme keywords in the business description, only
        # credited because the margin/growth checks above run independently.
        text = (str(m.get("business_summary") or "") + " "
                + str(m.get("industry") or "")).lower()
        themes = sorted({theme for kw, theme in _SECULAR_THEMES if kw in text})
        if themes:
            pos.append("levered to secular trend(s): " + ", ".join(themes[:3]))
            score += min(0.10, 0.05 * len(themes))
        score = max(0.0, min(1.0, score))
        moat_s = ("durable-advantage traits" if score >= 0.6
                  else "no clear moat signal" if score > 0.4 else "weak business quality")
        trend_s = f"; secular tailwind: {', '.join(themes[:3])}" if themes else ""
        assess = (f"Gross margin {gm if gm is not None else 'n/a'}%, operating margin "
                  f"{om if om is not None else 'n/a'}%, FCF margin "
                  f"{fcf if fcf is not None else 'n/a'}% - {moat_s}{trend_s}.")
        return AnalystRead("moat", _stance(score), score, assess, pos, con)

    def parse(self, raw: dict, inputs: dict) -> AnalystRead:
        score = _f(raw, "score", 0.5)
        stance = str(raw.get("stance") or _stance(score))
        return AnalystRead("moat", stance, score, str(raw.get("assessment", "")).strip(),
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
        reliable = g.get("fwd_eps_reliable", True)
        if isinstance(ig, (int, float)):
            if not reliable:
                con.append(f"forward-EPS jump ({ig:+.0f}%) looks like a trailing-year "
                           "artifact, not real acceleration - discounted")
            elif ig > 5:
                pos.append(f"forward EPS guidance above trailing ({ig:+.0f}%) - guided up"); score += 0.12
            elif ig < -5:
                con.append(f"forward EPS guidance below trailing ({ig:+.0f}%) - guided down"); score -= 0.12
        roe = g.get("return_on_equity_pct")
        if isinstance(roe, (int, float)) and roe > 20:
            pos.append(f"high return on equity ({roe:.0f}%)")
        pm = g.get("profit_margin_pct")
        if isinstance(pm, (int, float)) and pm < 0:
            con.append(f"unprofitable (margin {pm:+.0f}%)"); score -= 0.05
        # ESTIMATE-REVISION MOMENTUM: where the analyst consensus is HEADING.
        # Upward forward-EPS revisions lead price (post-revision drift); the
        # crowd of analysts is revising toward a reality the price hasn't fully
        # caught. A real leading signal, weighted by analyst coverage.
        rev90 = g.get("eps_revision_90d_pct")
        na = g.get("num_analysts")
        covered = isinstance(na, (int, float)) and na >= 4
        if isinstance(rev90, (int, float)) and covered:
            if rev90 >= 5:
                pos.append(f"forward-EPS estimates REVISED UP {rev90:+.0f}% over 90 "
                           "days - analysts catching up to improving fundamentals "
                           "(a leading signal)"); score += 0.14
            elif rev90 <= -5:
                con.append(f"forward-EPS estimates CUT {rev90:+.0f}% over 90 days - "
                           "deteriorating consensus, price tends to drift down after");
                score -= 0.14
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
