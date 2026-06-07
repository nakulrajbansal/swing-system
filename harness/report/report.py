"""Scorecards and PASS/KILL verdicts (harness spec §7/§8).

Pass bar (tunable, conservative):
  * quintile abnormal returns increase monotonically with raw_score,
  * the 20-day top-minus-bottom spread is positive and clearly above round-trip
    costs, with a Newey-West t-stat > 2.5,
  * it holds in the reserved out-of-sample window.
Anything else is a KILL — the default, honest outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.study import stats
from harness.study.costs import CostModel
from harness.study.event_study import EventStudyResult

DECISION_WINDOW = 20
TSTAT_BAR = 2.5
MIN_EVENTS = 30          # a PASS needs enough events to not be small-sample noise


def _monotonic(quintile_table: pd.DataFrame) -> bool:
    if quintile_table is None or len(quintile_table) < 2:
        return False
    means = quintile_table.sort_index()["mean"].to_numpy()
    idx = np.arange(len(means))
    if np.std(means) == 0:
        return False
    return float(np.corrcoef(idx, means)[0, 1]) > 0.5


def edge_scorecard(result: EventStudyResult, costs: CostModel | None = None) -> dict:
    costs = costs or CostModel()
    cost_frac = costs.round_trip_bps() / 10_000.0
    w = DECISION_WINDOW if DECISION_WINDOW in result.windows else result.windows[-1]

    spread = result.spreads.get(w, {})
    oos = result.oos_spreads.get(w, {})
    qtable = result.quintiles.get(w)

    monotonic = _monotonic(qtable)
    spr = spread.get("spread", float("nan"))
    tstat = spread.get("tstat", float("nan"))
    oos_spr = oos.get("spread", float("nan"))
    n_events = 0 if result.events is None else int(len(result.events))

    enough_events = n_events >= MIN_EVENTS
    passes = bool(
        enough_events                                  # small samples never pass
        and np.isfinite(spr) and spr > cost_frac
        and np.isfinite(tstat) and tstat > TSTAT_BAR
        and monotonic
        and (not np.isfinite(oos_spr) or oos_spr > 0)
    )
    return {
        "edge_id": result.edge_id,
        "n_events": n_events,
        "enough_events": enough_events,
        "decision_window": w,
        "monotonic_quintiles": monotonic,
        "spread": spr,
        "spread_vs_cost": spr - cost_frac if np.isfinite(spr) else float("nan"),
        "tstat": tstat,
        "oos_spread": oos_spr,
        "verdict": "PASS" if passes else "KILL",
        "quintiles": None if qtable is None else qtable.to_dict("index"),
    }


def portfolio_summary(scorecards: list[dict], results: list[EventStudyResult]) -> dict:
    """Aggregate verdicts and a deflated-Sharpe correction for multiple testing."""
    passed = [s["edge_id"] for s in scorecards if s["verdict"] == "PASS"]
    n_trials = max(1, len(scorecards))

    # Pool the decision-window abnormal returns of passing edges for a
    # portfolio-level deflated Sharpe across the edges tested.
    pooled = []
    for s, r in zip(scorecards, results):
        if s["verdict"] != "PASS":
            continue
        w = s["decision_window"]
        col = f"abn_{w}"
        if r.events is not None and col in r.events:
            pooled.extend(r.events[col].dropna().tolist())
    pooled = np.asarray(pooled, dtype=float)
    sr = stats.sharpe_ratio(pooled) if len(pooled) else 0.0
    dsr = stats.deflated_sharpe_ratio(sr, pooled, n_trials) if len(pooled) else 0.0

    return {
        "edges_tested": [s["edge_id"] for s in scorecards],
        "edges_passed": passed,
        "n_trials": n_trials,
        "pooled_sharpe": sr,
        "deflated_sharpe_prob": dsr,
        "portfolio_verdict": "PROCEED" if passed and dsr > 0.95 else "INSUFFICIENT",
    }


def format_scorecard(card: dict) -> str:
    enough = card.get("enough_events", card["n_events"] >= MIN_EVENTS)
    flag = "" if enough else f"  (< {MIN_EVENTS} events: too small to trust)"
    lines = [f"=== {card['edge_id']} : {card['verdict']} ===",
             f"  events={card['n_events']}{flag}  window={card['decision_window']}d",
             f"  spread={card['spread']:.4f}  vs_cost={card['spread_vs_cost']:.4f}"
             f"  tstat={card['tstat']:.2f}  oos_spread={card['oos_spread']:.4f}",
             f"  monotonic_quintiles={card['monotonic_quintiles']}"]
    return "\n".join(lines)
