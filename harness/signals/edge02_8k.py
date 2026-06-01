"""Edge 2 — 8-K material event (Family A, long; deterministic baseline).

Trigger: an 8-K became available today. Score: a crude materiality proxy from
the item count / body length (the LLM Filings Analyst later replaces this with a
real classification). On data without 8-Ks this edge simply produces no events
and is KILLed — an honest outcome.
"""

from __future__ import annotations

import pandas as pd

from harness.signals.base import filed_today


class Edge02EightK:
    edge_id = "edge02_8k"
    family = "A"
    direction = "long"

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        filings = view.filings()
        if filings.empty:
            return []
        eightk = filings[filings["form_type"].astype(str).str.startswith("8-K")]
        today = filed_today(view, eightk, date)
        return sorted(today["symbol"].dropna().unique().tolist())

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        filings = view.filings(symbol)
        eightk = filings[filings["form_type"].astype(str).str.startswith("8-K")]
        today = filed_today(view, eightk, date)
        if today.empty:
            return {"raw_score": 0.0, "confidence": 0.0, "evidence": {}}
        body = str(today.iloc[-1].get("section_text_mdna", ""))
        raw = min(1.0, len(body.split()) / 200.0)
        return {"raw_score": float(raw), "confidence": 0.4,
                "evidence": {"n_8k": int(len(today))}}
