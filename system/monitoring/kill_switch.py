"""Kill switch / fast-loop de-risking (master §15/§17, invariant 2).

Pre-committed, reduce-only actions. This module can recommend halting new entries
or flattening; it can NEVER add risk, raise a limit, or scale capital. Stopping
is a first-class, legitimate outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.config import KillCriteria
from system.monitoring.scorecard import HealthScorecard

OK = "ok"
HALT_NEW = "halt_new_entries"
FLATTEN = "flatten"


@dataclass
class KillVerdict:
    action: str            # OK | HALT_NEW | FLATTEN
    reasons: list[str]

    @property
    def reduce_only(self) -> bool:
        return self.action in {OK, HALT_NEW, FLATTEN}   # always true by construction


class KillSwitch:
    def __init__(self, criteria: KillCriteria):
        self.criteria = criteria

    def evaluate(self, scorecard: HealthScorecard, *,
                 daily_pnl_pct: float = 0.0, weekly_pnl_pct: float = 0.0,
                 slippage_ratio: float = 1.0,
                 reconciliation_ok: bool = True) -> KillVerdict:
        reasons, action = [], OK
        dd = scorecard.max_drawdown()

        if dd <= self.criteria.system_drawdown:
            reasons.append(f"system drawdown {dd:.2%}")
            action = FLATTEN
        elif dd <= self.criteria.strategy_drawdown:
            reasons.append(f"strategy drawdown {dd:.2%}")
            action = _escalate(action, HALT_NEW)

        if not reconciliation_ok:
            reasons.append("unresolved reconciliation drift")
            action = FLATTEN

        if slippage_ratio >= self.criteria.max_slippage_ratio:
            reasons.append(f"slippage {slippage_ratio:.1f}x model")
            action = _escalate(action, HALT_NEW)

        brier = scorecard.brier_score()
        if brier == brier and brier >= self.criteria.min_calibration_brier:
            reasons.append(f"calibration Brier {brier:.2f}")
            action = _escalate(action, HALT_NEW)

        return KillVerdict(action, reasons)


def _escalate(current: str, proposed: str) -> str:
    rank = {OK: 0, HALT_NEW: 1, FLATTEN: 2}
    return proposed if rank[proposed] > rank[current] else current
