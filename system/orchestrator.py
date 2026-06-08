"""The deterministic orchestrator (master §7/§8).

Not an LLM. Drives each cycle: sequencing, idempotent cycle_id, per-agent retry,
and the fail-safe. Specialists run independently; the Skeptic is blinded to the
proposer's conviction; any agent error after one retry defaults that candidate to
PASS (the system never fails *into* a trade). Produces RiskDecisions (proposals)
and a per-candidate Deliberation Record; the deterministic plane disposes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from system.agents.prompts import REBUTTAL
from system.agents.specialists import EdgeSpecialist
from system.config import SystemConfig
from system.confluence import Candidate, run_confluence
from system.data_plane.evidence import assemble_evidence
from system.data_plane.indicators import last_atr
from system.schemas import Envelope, RiskDecision, SpecialistRead


@dataclass
class CycleResult:
    cycle_id: str
    session_date: pd.Timestamp
    candidates: list[Candidate]
    decisions: list[RiskDecision]
    deliberation: dict[str, dict] = field(default_factory=dict)  # symbol -> transcript


class Orchestrator:
    def __init__(self, store, specialists: list[EdgeSpecialist], hypothesis,
                 skeptic, portfolio_manager, config: SystemConfig,
                 price_lookup=None):
        self.store = store
        self.specialists = specialists
        self.hypothesis = hypothesis
        self.skeptic = skeptic
        self.pm = portfolio_manager
        self.cfg = config
        # price_lookup(symbol, view) -> adjusted OHLCV df; defaults to view.prices.
        self.price_lookup = price_lookup or (lambda sym, view: view.prices(sym))

    def run_cycle(self, T) -> CycleResult:
        view = self.store.as_of(T)
        session = view.asof_date
        cycle_id = Envelope.hash_inputs({"date": str(session),
                                         "cfg": id(self.cfg.limits)})

        reads_by_symbol = self._gather_reads(view, session)
        candidates = run_confluence(reads_by_symbol, self.cfg)

        decisions, deliberation = [], {}
        for cand in candidates:
            decision, record = self._deliberate(view, cand)
            decisions.append(decision)
            deliberation[cand.symbol] = record
        return CycleResult(cycle_id, session, candidates, decisions, deliberation)

    def deliberate_symbol(self, T, symbol: str):
        """Force a full deliberation on ONE symbol (ignores confluence gating),
        returning (candidate, decision, transcript). For single-ticker analysis."""
        view = self.store.as_of(T)
        session = view.asof_date
        reads = []
        for spec in self.specialists:
            try:
                reads.append(spec.read(view, symbol, session))
            except Exception:
                continue
        cands = run_confluence({symbol: reads}, self.cfg)
        if cands:
            cand = cands[0]
        else:
            fams = sorted({r.family for r in reads if r.direction == "long"})
            cand = Candidate(symbol=symbol, combined_score=0.0, n_families=len(fams),
                             strong_single=False, families=fams,
                             edge_ids=[r.edge_id for r in reads],
                             evidence_refs=[e for r in reads for e in r.evidence_refs],
                             min_confidence=min((r.confidence for r in reads), default=0.0))
        decision, transcript = self._deliberate(view, cand)
        return cand, decision, transcript

    # -- specialists (independent, parallel-safe) --------------------------
    def _gather_reads(self, view, session) -> dict[str, list[SpecialistRead]]:
        reads: dict[str, list[SpecialistRead]] = {}
        for spec in self.specialists:
            try:
                triggered = spec.triggers(view, session)
            except Exception:
                triggered = []
            for sym in triggered:
                try:
                    read = spec.read(view, sym, session)
                except Exception:
                    continue
                reads.setdefault(sym, []).append(read)
        return reads

    # -- core trio with fail-to-PASS --------------------------------------
    def _step(self, agent, system_prompt, inputs, output) -> dict:
        """One transparent transcript step: which agent, its prompt, inputs, output."""
        inp = {k: v for k, v in inputs.items() if k != "evidence"}   # shown once
        return {"agent": agent.name, "model": agent.model,
                "system_prompt": system_prompt, "inputs": inp, "output": output}

    def _deliberate(self, view, cand: Candidate) -> tuple[RiskDecision, dict]:
        passcard = RiskDecision(cand.symbol, "pass", 0.0, decisive_factor="fail-safe PASS")

        # Give the agents the real, domain-specific evidence their expertise needs.
        evidence = assemble_evidence(view, cand.symbol)
        confluence = {"n_families": cand.n_families,
                      "combined_score": cand.combined_score,
                      "strong_single": cand.strong_single,
                      "edges": cand.edge_ids}
        transcript: dict = {"evidence": evidence, "steps": []}

        try:
            hyp_in = {"symbol": cand.symbol, "confluence": confluence,
                      "edge_ids": cand.edge_ids, "evidence_refs": cand.evidence_refs,
                      "evidence": evidence}
            hyp = self._with_retry(self.hypothesis, hyp_in)
            transcript["steps"].append(
                self._step(self.hypothesis, self.hypothesis.system_prompt, hyp_in, asdict(hyp)))

            crit_in = {"symbol": cand.symbol, "evidence": evidence, "thesis": asdict(hyp),
                       "max_corr_to_book": 0.0, "min_read_confidence": cand.min_confidence,
                       "priced_in": 0.0}
            crit = self._with_retry(self.skeptic, crit_in)
            transcript["steps"].append(
                self._step(self.skeptic, self.skeptic.system_prompt, crit_in, asdict(crit)))

            # Consensus step: on a survivable objection (caution), the proposer
            # gets one rebuttal before the PM arbitrates (master §8 flow).
            rebuttal = ""
            if crit.verdict == "caution" and hyp.decision == "propose":
                reb_in = {"symbol": cand.symbol, "thesis": asdict(hyp),
                          "objection": crit.strongest, "evidence": evidence}
                try:
                    reb = self.hypothesis.rebut(reb_in)
                    rebuttal = reb.get("rebuttal", "") if isinstance(reb, dict) else str(reb)
                    transcript["steps"].append({
                        "agent": "hypothesis_rebuttal", "model": self.hypothesis.model,
                        "system_prompt": REBUTTAL,
                        "inputs": {"objection": crit.strongest},
                        "output": {"rebuttal": rebuttal}})
                except Exception:
                    rebuttal = ""

            px = self.price_lookup(cand.symbol, view)
            price = float(px["close"].iloc[-1]) if len(px) else 0.0
            atr = last_atr(px, self.cfg.sizing.atr_len) if len(px) else 0.0

            pm_in = {"symbol": cand.symbol, "hypothesis": asdict(hyp),
                     "critique": {"verdict": crit.verdict, "max_severity": crit.max_severity(),
                                  "strongest": crit.strongest},
                     "rebuttal": rebuttal, "evidence": evidence,
                     "price": price, "atr": atr if atr == atr else 0.0}
            decision = self._with_retry(self.pm, pm_in)
            transcript["steps"].append(
                self._step(self.pm, self.pm.system_prompt, pm_in, asdict(decision)))
            return decision, transcript
        except Exception as exc:                      # fail to PASS, never into a trade
            transcript["steps"].append({"agent": "fail-safe", "model": "-",
                                        "system_prompt": "", "inputs": {},
                                        "output": {"error": f"{type(exc).__name__}: {exc}"}})
            passcard.decisive_factor = f"fail-safe PASS ({type(exc).__name__})"
            return passcard, transcript

    @staticmethod
    def _with_retry(agent, inputs, retries: int = 1):
        last = None
        for _ in range(retries + 1):
            try:
                return agent.run(inputs)
            except Exception as exc:
                last = exc
        raise last

    def _env(self, candidate_id, agent, payload) -> dict:
        return Envelope(candidate_id=candidate_id, agent=agent.name,
                        model_id=agent.model, prompt_version=agent.prompt_version,
                        payload=payload).to_dict()
