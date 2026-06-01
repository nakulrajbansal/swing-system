"""Edge 6 — insider cluster buying (Family C, long). EDGAR Form 4, free.

Mechanism: multiple senior insiders buying in a short window is an informed,
non-routine signal. Trigger: a cluster (>=2 distinct senior roles, txn_code 'P')
arrived today. Score: cluster breadth x total dollar value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.signals.base import filed_today

SENIOR = {"CEO", "CFO", "COO", "President", "Chairman", "Director"}


class Edge06Insider:
    edge_id = "edge06_insider"
    family = "C"
    direction = "long"

    def _clusters_today(self, view, date) -> pd.DataFrame:
        f4 = view.form4()
        if f4.empty:
            return f4
        buys = f4[f4["txn_code"] == "P"]
        today = filed_today(view, buys, date)
        return today

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        today = self._clusters_today(view, date)
        if today.empty:
            return []
        out = []
        for sym, grp in today.groupby("symbol"):
            senior_roles = set(grp["insider_role"]) & SENIOR
            if len(senior_roles) >= 2:
                out.append(sym)
        return sorted(out)

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        today = self._clusters_today(view, date)
        grp = today[today["symbol"] == symbol]
        if grp.empty:
            return {"raw_score": 0.0, "confidence": 0.0, "evidence": {}}
        breadth = len(set(grp["insider_role"]) & SENIOR)
        value = float(grp["value"].sum())
        raw = float(breadth * np.log1p(value))
        return {"raw_score": raw, "confidence": float(min(1.0, 0.3 + 0.2 * breadth)),
                "evidence": {"breadth": breadth, "total_value": value}}
