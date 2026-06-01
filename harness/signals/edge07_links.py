"""Edge 7 — economic-link read-through (Family D, long; deterministic baseline).

Mechanism: a catalyst at a focal firm propagates to disclosed economic links
(suppliers/customers) that have not yet moved. Trigger: a linked firm of the
focal had news today. Score: link strength proxy. The LLM Economic-Links Analyst
later replaces the scoring with real read-through inference.
"""

from __future__ import annotations

import pandas as pd

from harness.signals.base import filed_today

_STRENGTH = {"supplier": 0.8, "customer": 0.7, "competitor": 0.4}


class Edge07Links:
    edge_id = "edge07_links"
    family = "D"
    direction = "long"

    def _read_through_today(self, view, date) -> pd.DataFrame:
        links = view.links()
        news = view.news()
        if links.empty or news.empty:
            return pd.DataFrame(columns=["focal_symbol", "linked_symbol", "link_type"])
        news_today = filed_today(view, news, date)
        moved = set(news_today["symbol"].dropna())
        # focal triggers when its LINKED name had news today (focal not yet moved).
        hit = links[links["linked_symbol"].isin(moved) &
                    ~links["focal_symbol"].isin(moved)]
        return hit

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        hit = self._read_through_today(view, date)
        return sorted(hit["focal_symbol"].dropna().unique().tolist())

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        hit = self._read_through_today(view, date)
        grp = hit[hit["focal_symbol"] == symbol]
        if grp.empty:
            return {"raw_score": 0.0, "confidence": 0.0, "evidence": {}}
        raw = float(grp["link_type"].map(lambda t: _STRENGTH.get(t, 0.3)).max())
        return {"raw_score": raw, "confidence": 0.35,
                "evidence": {"links": grp["link_type"].tolist()}}
