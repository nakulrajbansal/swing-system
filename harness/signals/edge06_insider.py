"""Edge 6 — insider cluster buying (Family C, long). EDGAR Form 4, free.

Mechanism: multiple senior insiders buying in a short window is an informed,
non-routine signal. Trigger: a new purchase filed today brings the count of
senior-insider purchases in the trailing window to >= 2 for that symbol (i.e. a
cluster has formed). Score: cluster size x total dollar value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.signals.base import filed_today

SENIOR = {"CEO", "CFO", "COO", "President", "Chairman", "Director"}
WINDOW_DAYS = 7          # "short window" over which a cluster is counted
MIN_CLUSTER = 2          # >= this many senior buyers => cluster


class Edge06Insider:
    edge_id = "edge06_insider"
    family = "C"
    direction = "long"

    def _recent_senior_buys(self, view) -> pd.DataFrame:
        """Senior-insider PURCHASES filed in the trailing window (PIT)."""
        f4 = view.form4()
        if f4.empty:
            return f4
        buys = f4[(f4["txn_code"] == "P") & f4["insider_role"].isin(SENIOR)]
        if buys.empty:
            return buys
        lo = view.T - pd.Timedelta(days=WINDOW_DAYS)
        aa = pd.to_datetime(buys["available_at"])
        return buys[aa > lo]

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        f4 = view.form4()
        if f4.empty:
            return []
        today = filed_today(view, f4[f4["txn_code"] == "P"], date)
        if today.empty:
            return []
        recent = self._recent_senior_buys(view)
        out = []
        for sym in today["symbol"].dropna().unique():
            if (recent["symbol"] == sym).sum() >= MIN_CLUSTER:
                out.append(sym)
        return sorted(out)

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        recent = self._recent_senior_buys(view)
        grp = recent[recent["symbol"] == symbol]
        if grp.empty:
            return {"raw_score": 0.0, "confidence": 0.0, "evidence": {}}
        size = int(len(grp))                       # senior buyers in the window
        value = float(grp["value"].sum())
        raw = float(size * np.log1p(value))
        return {"raw_score": raw,
                "confidence": float(min(1.0, 0.3 + 0.15 * size)),
                "evidence": {"cluster_size": size, "total_value": value}}
