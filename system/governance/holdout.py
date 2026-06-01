"""Holdout data vault (master invariant 4, §12/§16).

A segregated period the optimizer never sees during search; consulted at most
once per candidate; every access audited. Enforced in code: a second look at the
same candidate raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class HoldoutVault:
    start: pd.Timestamp
    end: pd.Timestamp
    _accessed: dict[str, int] = field(default_factory=dict)
    audit_log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.start = pd.Timestamp(self.start)
        self.end = pd.Timestamp(self.end)

    def contains(self, date) -> bool:
        d = pd.Timestamp(date)
        return self.start <= d <= self.end

    def consult(self, candidate_id: str, evaluator) -> object:
        """Run `evaluator()` against the holdout exactly once per candidate."""
        if self._accessed.get(candidate_id, 0) >= 1:
            raise RuntimeError(
                f"holdout already consulted for {candidate_id!r}; a second look "
                "is forbidden (invariant 4)")
        self._accessed[candidate_id] = 1
        result = evaluator()
        self.audit_log.append({"candidate": candidate_id,
                               "at": pd.Timestamp.now("UTC").isoformat()})
        return result
