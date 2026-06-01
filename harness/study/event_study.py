"""The shared event-study engine (harness spec §6, master §13).

For each edge: find trigger sessions via the signal (point-in-time), score the
triggered names, enter at t+1 open with gap-aware costs, measure sector-adjusted
abnormal returns over [t+1, t+1+w], bucket by score into quintiles, and report
the top-minus-bottom spread with a Newey-West t-stat.

The PIT discipline applies to the SIGNAL's view of the world (triggers/score use
``store.as_of(decision_instant)``); realized forward returns are an *outcome*
measured from fully back-adjusted prices, which is correct and not leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from harness.data.loader import available_at_for_session
from harness.data.pit_store import PITStore
from harness.study import stats
from harness.study.costs import CostModel


@dataclass
class EventStudyResult:
    edge_id: str
    windows: tuple[int, ...]
    events: pd.DataFrame                       # one row per scored trigger
    quintiles: dict[int, pd.DataFrame]         # window -> quintile abnormal-return table
    spreads: dict[int, dict] = field(default_factory=dict)  # window -> {spread, tstat, n}
    oos_spreads: dict[int, dict] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            s = self.spreads.get(w, {})
            o = self.oos_spreads.get(w, {})
            rows.append({
                "window": w,
                "n": s.get("n", 0),
                "top_minus_bottom": s.get("spread", np.nan),
                "tstat": s.get("tstat", np.nan),
                "oos_spread": o.get("spread", np.nan),
                "oos_tstat": o.get("tstat", np.nan),
            })
        return pd.DataFrame(rows)


def _adjusted_panels(store: PITStore, symbols, asof) -> dict[str, pd.DataFrame]:
    view = store.as_of(asof)
    panels = {}
    for sym in symbols:
        df = view.prices(sym, adjust=True)
        if not df.empty:
            panels[sym] = df.set_index("date")
    return panels


def run_event_study(
    store: PITStore,
    signal,
    sector_map: dict[str, str],
    windows: tuple[int, ...] = (5, 10, 20),
    costs: CostModel | None = None,
    oos_start: str | pd.Timestamp | None = None,
) -> EventStudyResult:
    costs = costs or CostModel()
    cost_frac = costs.round_trip_bps() / 10_000.0

    universe = store.as_of(_last_instant(store)).universe()
    last = _last_instant(store)
    panels = _adjusted_panels(store, set(universe) | set(sector_map.values()), last)

    # Trigger sessions: ask the signal, per session, using a PIT view.
    sessions = _all_sessions(panels, universe)
    records = []
    for sess in sessions:
        view = store.as_of(available_at_for_session(sess))
        try:
            triggered = signal.triggers(view, sess)
        except Exception:
            triggered = []
        for sym in triggered:
            if sym not in panels:
                continue
            sc = signal.score(view, sym, sess)
            rec = {"symbol": sym, "date": pd.Timestamp(sess),
                   "raw_score": float(sc["raw_score"])}
            for w in windows:
                rec[f"abn_{w}"] = _abnormal_return(
                    panels, sym, sector_map.get(sym), sess, w, cost_frac)
            records.append(rec)

    events = pd.DataFrame(records)
    result = EventStudyResult(edge_id=getattr(signal, "edge_id", "?"),
                              windows=windows, events=events, quintiles={})
    if events.empty:
        return result

    oos_start = pd.Timestamp(oos_start) if oos_start is not None else None
    for w in windows:
        col = f"abn_{w}"
        sub = events.dropna(subset=[col, "raw_score"]).copy()
        if sub.empty:
            continue
        sub["quintile"] = stats.quintile_buckets(sub["raw_score"])
        result.quintiles[w] = sub.groupby("quintile")[col].agg(["mean", "count"])
        result.spreads[w] = _spread(sub, col, w)
        if oos_start is not None:
            oos = sub[sub["date"] >= oos_start]
            if not oos.empty:
                oos = oos.copy()
                oos["quintile"] = stats.quintile_buckets(oos["raw_score"])
                result.oos_spreads[w] = _spread(oos, col, w)
    return result


def _spread(sub: pd.DataFrame, col: str, window: int) -> dict:
    qmax, qmin = sub["quintile"].max(), sub["quintile"].min()
    top = sub.loc[sub["quintile"] == qmax, col].to_numpy()
    bot = sub.loc[sub["quintile"] == qmin, col].to_numpy()
    spread = float(np.nanmean(top) - np.nanmean(bot)) if len(top) and len(bot) else np.nan
    # Per-event spread proxy for the t-stat: top names minus the bottom-quintile mean.
    series = sub[col].to_numpy() - np.nanmean(bot) if len(bot) else sub[col].to_numpy()
    _, tstat = stats.newey_west_tstat(series, lag=window)
    return {"spread": spread, "tstat": float(tstat), "n": int(len(sub))}


def _abnormal_return(panels, sym, sector, sess, window, cost_frac) -> float:
    sp = panels.get(sym)
    if sp is None or sess not in sp.index:
        return np.nan
    idx = sp.index.get_loc(sess)
    entry_i, exit_i = idx + 1, idx + 1 + window
    if exit_i >= len(sp.index):
        return np.nan
    entry, exit_ = sp["open"].iloc[entry_i], sp["open"].iloc[exit_i]
    if entry <= 0:
        return np.nan
    stock_ret = exit_ / entry - 1.0
    bench_ret = 0.0
    bp = panels.get(sector)
    if bp is not None:
        edates = sp.index[entry_i], sp.index[exit_i]
        if edates[0] in bp.index and edates[1] in bp.index:
            b0, b1 = bp["open"].loc[edates[0]], bp["open"].loc[edates[1]]
            if b0 > 0:
                bench_ret = b1 / b0 - 1.0
    return stock_ret - bench_ret - cost_frac


def _all_sessions(panels, universe) -> pd.DatetimeIndex:
    idx = None
    for sym in universe:
        if sym in panels:
            idx = panels[sym].index if idx is None else idx.union(panels[sym].index)
    return idx if idx is not None else pd.DatetimeIndex([])


def _last_instant(store: PITStore) -> pd.Timestamp:
    prices = store.read_table("prices")
    if prices.empty:
        return pd.Timestamp.now(tz="UTC")
    last_date = pd.to_datetime(prices["date"]).max()
    return available_at_for_session(last_date) + pd.Timedelta(days=1)
