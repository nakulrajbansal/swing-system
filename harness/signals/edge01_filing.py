"""Edge 1 — filing-text change (Family A, long). EDGAR, free, leak-free.

Mechanism: material year-over-year changes in 10-K/10-Q risk factors and MD&A
are under-reacted to. Trigger: a periodic filing became available today. Score:
how much the text changed vs the comparable prior filing (token-set distance).
"""

from __future__ import annotations

import pandas as pd

from harness.signals.base import filed_today

PERIODIC = {"10-K", "10-Q"}


def _tokens(text) -> set[str]:
    return set(str(text).lower().split()) if text is not None else set()


def _jaccard_distance(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return 1.0 - inter / union if union else 0.0


class Edge01Filing:
    edge_id = "edge01_filing"
    family = "A"
    direction = "long"

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        filings = view.filings()
        if filings.empty:
            return []
        filings = filings[filings["form_type"].isin(PERIODIC)]
        today = filed_today(view, filings, date)
        return sorted(today["symbol"].dropna().unique().tolist())

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        filings = view.filings(symbol)
        filings = filings[filings["form_type"].isin(PERIODIC)].sort_values("available_at")
        if len(filings) < 1:
            return {"raw_score": 0.0, "confidence": 0.2,
                    "evidence": {"reason": "no filing"}}
        cur = filings.iloc[-1]
        # Compare like-for-like: prior filing of the SAME form type (10-Q vs prior
        # 10-Q), not 10-Q vs 10-K which is a length artifact, not a real change.
        same = filings[filings["form_type"] == cur["form_type"]]
        if len(same) < 2:
            return {"raw_score": 0.0, "confidence": 0.2,
                    "evidence": {"reason": "no prior same-type filing to diff"}}
        prev = same.iloc[-2]

        def _len(row) -> int:
            return len(_tokens(row["section_text_riskfactors"])) + \
                len(str(row["section_text_mdna"]).split())

        cur_len, prev_len = _len(cur), _len(prev)
        abs_change = abs(cur_len - prev_len)      # magnitude of YoY filing-text change
        d_risk = _jaccard_distance(cur["section_text_riskfactors"],
                                   prev["section_text_riskfactors"])
        raw = float(abs_change)
        return {
            "raw_score": raw,
            "confidence": float(min(1.0, 0.4 + abs_change / 100.0)),
            "evidence": {"abs_len_change": abs_change, "d_risk": d_risk,
                         "accession": cur["accession"]},
        }
