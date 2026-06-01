"""The core deliberation trio (master §6/§8).

Hypothesis -> Skeptic (blinded to conviction) -> Portfolio Manager (default PASS).
Deterministic implementations encode the design's decision discipline so the full
pipeline runs offline; with a real client the same prompts/schemas drive an LLM.
"""

from __future__ import annotations

from system.agents.base import Agent
from system.agents.prompts import HYPOTHESIS, PORTFOLIO_MANAGER, SKEPTIC
from system.schemas import Critique, Objection, RiskDecision, TradeHypothesis


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
        edges = inputs.get("edge_ids", [])

        if n_fam >= 2 or strong:
            conviction = min(0.85, 0.45 + 0.12 * n_fam + 0.2 * combined)
            return TradeHypothesis(
                symbol=symbol, decision="propose",
                mechanism=f"Cross-family confluence ({n_fam} families: {edges}) "
                          "signals under-reaction; expect drift over the swing window.",
                evidence_refs=list(inputs.get("evidence_refs", [])),
                expected_hold_days=10, invalidation="thesis catalyst reverses or "
                "price closes below the protective stop",
                raw_conviction=float(conviction), direction="long")
        return TradeHypothesis(
            symbol=symbol, decision="decline", mechanism="", evidence_refs=[],
            expected_hold_days=10, invalidation="", raw_conviction=0.3)


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
        # final_conviction is LOWER than the proposer's whenever objections stand.
        final = float(hyp["raw_conviction"]) * (1.0 - 0.5 * max_sev)
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
