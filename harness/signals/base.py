"""The edge-signal plug-in interface (harness spec §5).

Every edge implements this so one study engine — and later the live specialist
agents — serve all of them. ``triggers`` and ``score`` must compute using ONLY
the point-in-time view passed in (``store.as_of(date)``); never the raw store.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class EdgeSignal(Protocol):
    edge_id: str
    family: str           # "A".."E" (master §4)
    direction: str        # "long" | "short" | "avoid"

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        """Symbols whose trigger fired as of `date` (e.g. a new filing today)."""
        ...

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        """{'raw_score': float, 'confidence': float, 'evidence': dict}.

        Higher raw_score = stronger signal. Computed using only `view`.
        """
        ...


def _et_date(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert("America/New_York")
    return pd.Timestamp(ts.date())


def filed_today(view, table_df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Rows from an event frame whose availability falls on session `date`."""
    if table_df.empty:
        return table_df
    d = pd.Timestamp(date).normalize()
    et = table_df["available_at"].map(_et_date)
    return table_df[et == d]
