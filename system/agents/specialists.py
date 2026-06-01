"""Specialist analysts (master §6).

A specialist = a deterministic edge (trigger + validated baseline score) wrapped
to emit a `SpecialistRead`. In deterministic mode the read IS the validated
signal — exactly the harness contract (master §5/§13: the harness and live system
share one interface). With a real LLM client, the same agent would instead reason
over least-context evidence to produce the read.
"""

from __future__ import annotations

import pandas as pd

from system.agents.base import Agent
from system.agents.prompts import (
    CATALYST,
    ECONOMIC_LINKS,
    FILINGS,
    INSIDER,
    MARKET_STRUCTURE,
    TRANSCRIPT_TONE,
)
from system.schemas import SpecialistRead

_AGENT_PROMPT = {
    "A": FILINGS, "C": INSIDER, "D": ECONOMIC_LINKS, "E": MARKET_STRUCTURE,
    "B": TRANSCRIPT_TONE,
}


class EdgeSpecialist(Agent):
    """Wraps one EdgeSignal as a specialist that emits a SpecialistRead."""

    def __init__(self, edge, client, model):
        super().__init__(client, model, _AGENT_PROMPT.get(edge.family, CATALYST))
        self.edge = edge
        self.name = f"specialist::{edge.edge_id}"
        self.schema_name = "SpecialistRead"

    def triggers(self, view, date) -> list[str]:
        return self.edge.triggers(view, date)

    def read(self, view, symbol, date) -> SpecialistRead:
        """Convenience: produce the read for a (symbol, date) via the run path."""
        return self.run({"view": view, "symbol": symbol, "date": date})

    def deterministic(self, inputs: dict) -> SpecialistRead:
        view, symbol, date = inputs["view"], inputs["symbol"], inputs["date"]
        sc = self.edge.score(view, symbol, date)
        return SpecialistRead(
            edge_id=self.edge.edge_id,
            family=self.edge.family,
            raw_score=float(sc["raw_score"]),
            confidence=float(sc.get("confidence", 0.5)),
            direction=self.edge.direction,
            evidence_refs=[f"{k}={v}" for k, v in sc.get("evidence", {}).items()],
        )

    def parse(self, raw: dict, inputs: dict) -> SpecialistRead:
        return SpecialistRead(
            edge_id=self.edge.edge_id, family=self.edge.family,
            raw_score=float(raw.get("raw_score", 0.0)),
            confidence=float(raw.get("confidence", 0.0)),
            direction=raw.get("direction", self.edge.direction),
            evidence_refs=list(raw.get("evidence_refs", [])),
        )
