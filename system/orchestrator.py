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

from system.agents.specialists import EdgeSpecialist
from system.config import SystemConfig
from system.confluence import Candidate, run_confluence
from system.data_plane.indicators import last_atr
from system.schemas import Envelope, RiskDecision, SpecialistRead


@dataclass
class CycleResult:
    cycle_id: str
    session_date: pd.Timestamp
    candidates: list[Candidate]
    decisions: list[RiskDecision]
    deliberation: dict[str, list[dict]] = field(default_factory=dict)


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
    def _deliberate(self, view, cand: Candidate) -> tuple[RiskDecision, list[dict]]:
        record: list[dict] = []
        passcard = RiskDecision(cand.symbol, "pass", 0.0, decisive_factor="fail-safe PASS")

        try:
            hyp = self._with_retry(self.hypothesis, {
                "symbol": cand.symbol,
                "confluence": {"n_families": cand.n_families,
                               "combined_score": cand.combined_score,
                               "strong_single": cand.strong_single},
                "edge_ids": cand.edge_ids, "evidence_refs": cand.evidence_refs})
            record.append(self._env(cand.symbol, self.hypothesis, asdict(hyp)))

            crit = self._with_retry(self.skeptic, {
                "symbol": cand.symbol,
                "max_corr_to_book": 0.0,
                "min_read_confidence": cand.min_confidence,
                "priced_in": 0.0})
            record.append(self._env(cand.symbol, self.skeptic, asdict(crit)))

            px = self.price_lookup(cand.symbol, view)
            price = float(px["close"].iloc[-1]) if len(px) else 0.0
            atr = last_atr(px, self.cfg.sizing.atr_len) if len(px) else 0.0

            decision = self._with_retry(self.pm, {
                "symbol": cand.symbol,
                "hypothesis": asdict(hyp),
                "critique": {"verdict": crit.verdict, "max_severity": crit.max_severity()},
                "price": price, "atr": atr if atr == atr else 0.0})  # NaN guard
            record.append(self._env(cand.symbol, self.pm, asdict(decision)))
            return decision, record
        except Exception as exc:                      # fail to PASS, never into a trade
            passcard.decisive_factor = f"fail-safe PASS ({type(exc).__name__})"
            record.append(self._env(cand.symbol, self.pm, asdict(passcard)))
            return passcard, record

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
