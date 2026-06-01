"""Champion/challenger promotion pipeline (master §16, invariant 3).

Candidate -> validation gates -> single holdout look -> shadow/paper (min 2-3
months) -> tiny-capital challenger -> HUMAN promotion gate -> champion.

Asymmetric: a winning challenger does NOT auto-promote (that adds risk and
requires a human); a degrading champion DOES auto-demote (that reduces risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"          # passed deflated-Sharpe / PBO / SPA gates
    HOLDOUT_CHECKED = "holdout_checked"
    SHADOW = "shadow"                # paper, >= 2-3 months
    CHALLENGER = "challenger"        # tiny capital, head-to-head
    CHAMPION = "champion"
    ARCHIVED = "archived"


_FORWARD = [Stage.PROPOSED, Stage.VALIDATED, Stage.HOLDOUT_CHECKED,
            Stage.SHADOW, Stage.CHALLENGER, Stage.CHAMPION]


@dataclass
class Candidate:
    edge_id: str
    stage: Stage = Stage.PROPOSED
    history: list[str] = field(default_factory=list)


class PromotionPipeline:
    """Advances candidates one gate at a time. Promotion to CHAMPION requires an
    explicit human act; demotion is automatic."""

    def __init__(self):
        self.candidates: dict[str, Candidate] = {}

    def submit(self, edge_id: str) -> Candidate:
        c = Candidate(edge_id)
        self.candidates[edge_id] = c
        c.history.append("submitted")
        return c

    def advance(self, edge_id: str, gate_passed: bool, *, human_approval: bool = False) -> Stage:
        c = self.candidates[edge_id]
        if not gate_passed:
            return c.stage
        idx = _FORWARD.index(c.stage)
        if idx + 1 >= len(_FORWARD):
            return c.stage
        nxt = _FORWARD[idx + 1]
        # The final gate into CHAMPION is the human promotion gate.
        if nxt == Stage.CHAMPION and not human_approval:
            c.history.append("awaiting human promotion gate")
            return c.stage
        c.stage = nxt
        c.history.append(f"-> {nxt.value}")
        return c.stage

    def demote(self, edge_id: str, reason: str) -> Stage:
        """Automatic, reduce-only: a degrading champion steps down without a human."""
        c = self.candidates[edge_id]
        c.stage = Stage.ARCHIVED
        c.history.append(f"auto-demoted: {reason}")
        return c.stage
