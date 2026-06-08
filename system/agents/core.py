"""The core deliberation trio (master §6/§8).

Hypothesis -> Skeptic (blinded to conviction) -> Portfolio Manager (default PASS).
Deterministic implementations encode the design's decision discipline so the full
pipeline runs offline; with a real client the same prompts/schemas drive an LLM.
"""

from __future__ import annotations

from system.agents.base import Agent
from system.agents.prompts import HYPOTHESIS, PORTFOLIO_MANAGER, REBUTTAL, SKEPTIC
from system.schemas import Critique, Objection, RiskDecision, TradeHypothesis


def _f(raw: dict, key: str, default=0.0) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _mechanism_from_evidence(ev: dict, n_fam: int) -> str:
    """Build a readable, evidence-grounded thesis (deterministic stand-in)."""
    bits = []
    t = ev.get("technicals", {})
    if t.get("pct_below_52wk_high") is not None and t["pct_below_52wk_high"] > -6:
        bits.append(f"price within {abs(t['pct_below_52wk_high']):.0f}% of its 52-week high")
    if t.get("momentum_6mo_pct"):
        bits.append(f"{t['momentum_6mo_pct']:+.0f}% 6-month momentum")
    fl = ev.get("filings", {})
    if fl.get("text_change_vs_prior_pct"):
        bits.append(f"latest {fl.get('latest_periodic', {}).get('form', 'filing')} text "
                    f"changed {fl['text_change_vs_prior_pct']:+.0f}% vs the prior")
    if fl.get("recent_8k_count"):
        bits.append(f"{fl['recent_8k_count']} recent 8-K event(s)")
    ins = ev.get("insider", {}).get("recent_purchases", [])
    if ins:
        bits.append(f"{len(ins)} insider purchase(s)")
    if not bits:
        return (f"Cross-family confluence ({n_fam} families) signals under-reaction; "
                "expect drift over the swing window.")
    return ("Confluence of " + "; ".join(bits)
            + " suggests under-reaction; expect continuation/drift over the 2-20 day window.")


class HypothesisAgent(Agent):
    name = "hypothesis"
    schema_name = "TradeHypothesis"

    def __init__(self, client, model):
        super().__init__(client, model, HYPOTHESIS)

    def deterministic(self, inputs: dict) -> TradeHypothesis:
        symbol = inputs["symbol"]
        conf = inputs["confluence"]
        n_fam = int(conf.get("n_families", 0))
        strong = bool(conf.get("strong_single", False))
        combined = float(conf.get("combined_score", 0.0))

        if n_fam >= 2 or strong:
            mechanism = _mechanism_from_evidence(inputs.get("evidence", {}), n_fam)
            conviction = min(0.85, 0.45 + 0.12 * n_fam + 0.2 * combined)
            return TradeHypothesis(
                symbol=symbol, decision="propose", mechanism=mechanism,
                evidence_refs=list(inputs.get("evidence_refs", [])),
                expected_hold_days=10, invalidation="thesis catalyst reverses or "
                "price closes below the protective stop",
                raw_conviction=float(conviction), direction="long")
        return TradeHypothesis(
            symbol=symbol, decision="decline", mechanism="", evidence_refs=[],
            expected_hold_days=10, invalidation="", raw_conviction=0.3)

    def rebut(self, inputs: dict) -> dict:
        """One rebuttal to the skeptic's strongest objection (consensus step)."""
        if self.client.deterministic:
            return {"rebuttal": "The objection is a general base-rate caution, not "
                    "specific to this name; the concrete signals in the evidence "
                    "still support a bounded-risk entry."}
        raw = self.client.complete(REBUTTAL, inputs, "Rebuttal", model=self.model,
                                   max_tokens=200, temperature=self.temperature)
        return {"rebuttal": str(raw.get("rebuttal", ""))}

    def parse(self, raw: dict, inputs: dict) -> TradeHypothesis:
        return TradeHypothesis(
            symbol=inputs["symbol"],
            decision="propose" if raw.get("decision") == "propose" else "decline",
            mechanism=str(raw.get("mechanism", "")),
            evidence_refs=list(raw.get("evidence_refs", [])),
            expected_hold_days=int(raw.get("expected_hold_days", 10) or 10),
            invalidation=str(raw.get("invalidation", "")),
            raw_conviction=_f(raw, "raw_conviction", 0.3),
            direction=raw.get("direction", "long"))


class SkepticAgent(Agent):
    name = "skeptic"
    schema_name = "Critique"

    def __init__(self, client, model):
        super().__init__(client, model, SKEPTIC)

    def deterministic(self, inputs: dict) -> Critique:
        # Blinded: receives the candidate's evidence and open book, NOT the
        # proposer's conviction.
        objs = [Objection("base_rate",
                          "documented edges decay; most signals do not pay net of costs",
                          0.45)]
        corr = float(inputs.get("max_corr_to_book", 0.0))
        if corr > 0.6:
            objs.append(Objection("crowding",
                                  f"correlation {corr:.2f} to an open position", 0.7))
        min_conf = float(inputs.get("min_read_confidence", 1.0))
        if min_conf < 0.4:
            objs.append(Objection("data_quality",
                                  f"thin/low-confidence reads ({min_conf:.2f})", 0.6))
        priced = float(inputs.get("priced_in", 0.0))
        if priced > 0.6:
            objs.append(Objection("priced_in", "catalyst likely already discounted", 0.5))

        crit = Critique(objections=objs, strongest=max(objs, key=lambda o: o.severity).detail,
                        verdict="clean")
        m = crit.max_severity()
        crit.verdict = "kill" if m >= 0.7 else "caution" if m >= 0.45 else "clean"
        return crit

    def parse(self, raw: dict, inputs: dict) -> Critique:
        objs = [Objection(str(o.get("kind", "objection")), str(o.get("detail", "")),
                          _f(o, "severity", 0.3))
                for o in raw.get("objections", [])] or \
            [Objection("base_rate", "edges decay", 0.4)]
        verdict = raw.get("verdict", "caution")
        if verdict not in {"kill", "caution", "clean"}:
            verdict = "caution"
        return Critique(objections=objs,
                        strongest=str(raw.get("strongest", objs[0].detail)),
                        verdict=verdict)


class PortfolioManagerAgent(Agent):
    name = "portfolio_manager"
    schema_name = "RiskDecision"

    def __init__(self, client, model):
        super().__init__(client, model, PORTFOLIO_MANAGER)

    def deterministic(self, inputs: dict) -> RiskDecision:
        symbol = inputs["symbol"]
        hyp = inputs["hypothesis"]          # dict-like
        crit = inputs["critique"]
        price = float(inputs["price"])
        atr = float(inputs["atr"])

        decision_pass = RiskDecision(symbol=symbol, action="pass",
                                     final_conviction=0.0,
                                     decisive_factor="default PASS")
        if hyp["decision"] != "propose" or crit["verdict"] == "kill" or price <= 0 or atr <= 0:
            decision_pass.decisive_factor = (
                "declined" if hyp["decision"] != "propose"
                else "skeptic kill" if crit["verdict"] == "kill" else "bad price/atr")
            return decision_pass

        max_sev = float(crit.get("max_severity", 0.0))
        # A rebuttal that addressed the objection softens (not erases) its weight.
        penalty = 0.35 if inputs.get("rebuttal") else 0.5
        # final_conviction is LOWER than the proposer's whenever objections stand.
        final = float(hyp["raw_conviction"]) * (1.0 - penalty * max_sev)
        if final < 0.55:
            decision_pass.final_conviction = final
            decision_pass.decisive_factor = "conviction below entry threshold after critique"
            return decision_pass

        entry = price
        stop = price - 2.0 * atr            # request only; Risk Governor recomputes
        target = entry + 2.0 * (entry - stop)
        return RiskDecision(symbol=symbol, action="enter", final_conviction=final,
                            entry=entry, stop=stop, target=target, constraints_ack=True,
                            decisive_factor="edge survives the bear case")

    def parse(self, raw: dict, inputs: dict) -> RiskDecision:
        action = raw.get("action", "pass")
        if action not in {"enter", "adjust", "pass"}:
            action = "pass"
        price = _f(inputs, "price", 0.0)
        atr = _f(inputs, "atr", 0.0)
        # Treat the model's entry/stop/target as requests; fall back to ATR if absent.
        entry = _f(raw, "entry", price) if action != "pass" else None
        stop = _f(raw, "stop", price - 2 * atr) if action != "pass" else None
        target = _f(raw, "target", (entry + 2 * (entry - stop))
                    if (entry and stop) else 0.0) if action != "pass" else None
        return RiskDecision(symbol=inputs["symbol"], action=action,
                            final_conviction=_f(raw, "final_conviction", 0.0),
                            entry=entry, stop=stop, target=target,
                            constraints_ack=bool(raw.get("constraints_ack", True)),
                            decisive_factor=str(raw.get("decisive_factor", "")))
