"""Agent plane: deterministic reads, the trio's decision discipline, fail-to-PASS."""

import pytest

from system.agents.core import HypothesisAgent, PortfolioManagerAgent, SkepticAgent
from system.agents.llm_client import MockLLMClient
from system.config import SystemConfig

CFG = SystemConfig()
M = CFG.models
CLIENT = MockLLMClient()


def _hyp(n_families, strong=False, combined=0.8):
    agent = HypothesisAgent(CLIENT, M.synthesis)
    return agent.run({"symbol": "AAA",
                      "confluence": {"n_families": n_families, "combined_score": combined,
                                     "strong_single": strong},
                      "edge_ids": ["edge01"], "evidence_refs": []})


def test_hypothesis_proposes_on_confluence_declines_otherwise():
    assert _hyp(2).decision == "propose"
    assert _hyp(1).decision == "decline"
    assert _hyp(1, strong=True).decision == "propose"


def test_swing_hold_is_within_bounds():
    assert 2 <= _hyp(2).expected_hold_days <= 20


def test_skeptic_is_adversarial_and_flags_crowding():
    crit = SkepticAgent(CLIENT, M.adversarial).run(
        {"symbol": "AAA", "max_corr_to_book": 0.8,
         "min_read_confidence": 0.9, "priced_in": 0.0})
    assert crit.objections                      # always finds a base-rate objection
    assert crit.verdict in {"kill", "caution"}  # crowding severity 0.7 -> kill


def test_pm_defaults_to_pass_on_skeptic_kill():
    pm = PortfolioManagerAgent(CLIENT, M.adversarial)
    d = pm.run({"symbol": "AAA",
                "hypothesis": {"decision": "propose", "raw_conviction": 0.8},
                "critique": {"verdict": "kill", "max_severity": 0.8},
                "price": 100.0, "atr": 2.0})
    assert d.action == "pass"


def test_pm_final_conviction_below_proposer_when_objections_stand():
    pm = PortfolioManagerAgent(CLIENT, M.adversarial)
    d = pm.run({"symbol": "AAA",
                "hypothesis": {"decision": "propose", "raw_conviction": 0.9},
                "critique": {"verdict": "caution", "max_severity": 0.5},
                "price": 100.0, "atr": 2.0})
    assert d.final_conviction < 0.9
    if d.action == "enter":
        assert d.stop < d.entry                 # long stop below entry


def test_pm_enters_on_clean_high_conviction():
    pm = PortfolioManagerAgent(CLIENT, M.adversarial)
    d = pm.run({"symbol": "AAA",
                "hypothesis": {"decision": "propose", "raw_conviction": 0.8},
                "critique": {"verdict": "clean", "max_severity": 0.3},
                "price": 100.0, "atr": 2.0})
    assert d.action == "enter" and d.entry == 100.0 and d.stop < 100.0


def test_analysts_read_fundamentals_deterministically():
    from system.agents.analysts import (
        FundamentalAnalyst, GrowthAnalyst, TechnicalAnalyst, ValuationAnalyst)

    ev = {"technicals": {"available": True, "price": 100.0, "above_200dma": True,
                         "momentum_6mo_pct": 20.0, "pct_below_52wk_high": -4.0, "rsi14": 60},
          "fundamentals": {"available": True,
                           "valuation": {"forward_pe": 12.0, "peg_ratio": 0.8,
                                         "price_to_sales": 3.0, "analyst_target_price": 120.0},
                           "growth": {"revenue_growth_pct": 22.0, "earnings_growth_pct": 30.0,
                                      "implied_fwd_eps_growth_pct": 18.0,
                                      "return_on_equity_pct": 25.0, "profit_margin_pct": 20.0}},
          "filings": {"available": False}, "insider": {"recent_purchases": []}}
    inp = {"symbol": "AAA", "evidence": ev}

    tech = TechnicalAnalyst(CLIENT, M.framing).run(inp)
    val = ValuationAnalyst(CLIENT, M.framing).run(inp)
    grow = GrowthAnalyst(CLIENT, M.framing).run(inp)
    fund = FundamentalAnalyst(CLIENT, M.framing).run(inp)

    assert tech.stance == "bullish" and 0 <= tech.score <= 1
    assert val.stance == "bullish"          # cheap P/E + PEG<1 + upside to target
    assert grow.stance == "bullish"         # strong growth + guided up
    assert fund.domain == "fundamental"
    # Missing fundamentals -> neutral, never crashes.
    empty = ValuationAnalyst(CLIENT, M.framing).run(
        {"symbol": "AAA", "evidence": {"fundamentals": {"available": False}}})
    assert empty.stance == "neutral"


def test_rebuttal_fallback_is_substantive_not_dismissive():
    agent = HypothesisAgent(CLIENT, M.synthesis)
    out = agent.rebut({"thesis": {"mechanism": "Uptrend continuation with cheap valuation",
                                  "invalidation": "close below 200-DMA"},
                       "objection": "edges decay"})
    text = out["rebuttal"].lower()
    assert "uptrend continuation" in text          # engages the actual mechanism
    assert "general base-rate caution" not in text  # not the old dismissive canned line


def test_orchestrator_fails_to_pass(synth_store):
    # An agent that always raises must yield a PASS decision, never a trade.
    from system.orchestrator import Orchestrator
    from harness.signals import Edge01Filing, Edge06Insider, Edge08Momentum
    from system.agents.specialists import EdgeSpecialist
    from harness.data.loader import available_at_for_session
    import pandas as pd

    store, sector_map = synth_store

    class Boom(HypothesisAgent):
        def deterministic(self, inputs):
            raise RuntimeError("boom")

    specs = [EdgeSpecialist(E(), CLIENT, M.framing)
             for E in (Edge01Filing, Edge06Insider, Edge08Momentum)]
    orch = Orchestrator(store, specs, Boom(CLIENT, M.synthesis),
                        SkepticAgent(CLIENT, M.adversarial),
                        PortfolioManagerAgent(CLIENT, M.adversarial), CFG)

    # Sweep the back half of the history to encounter confluence candidates;
    # no matter what, a crashing agent must never produce enter/adjust.
    sessions = pd.to_datetime(sorted(store.read_table("prices")["date"].unique()))
    saw_candidate = False
    for d in sessions[len(sessions) // 2:]:
        res = orch.run_cycle(available_at_for_session(d))
        saw_candidate = saw_candidate or bool(res.candidates)
        assert all(dec.action == "pass" for dec in res.decisions)
    assert saw_candidate, "expected at least one confluence candidate to exercise the fail-safe"
