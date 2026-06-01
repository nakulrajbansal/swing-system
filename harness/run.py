"""Phase 1 harness orchestrator: load -> signal -> study -> report.

Runs the free edges (1, 2, 6, 7, 8) through the shared event-study engine on the
deterministic synthetic universe (offline). The set of passing edges is the
project's first real evidence and gates everything downstream.

    python -m harness.run                 # synthetic, offline
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from harness.data.loader import SyntheticConfig, SyntheticLoader
from harness.data.pit_store import PITStore
from harness.report.report import edge_scorecard, format_scorecard, portfolio_summary
from harness.signals import ALL_FREE_EDGES
from harness.study.costs import CostModel
from harness.study.event_study import run_event_study


def build_synthetic_store(root: Path | None = None) -> tuple[PITStore, dict]:
    root = root or Path(tempfile.mkdtemp(prefix="pit_"))
    store = PITStore(root)
    loader = SyntheticLoader(store, SyntheticConfig())
    loader.load_all()
    return store, loader.sector_map()


def run_all_edges(store: PITStore, sector_map: dict, oos_start="2022-01-01") -> tuple[list, list]:
    costs = CostModel()
    results, cards = [], []
    for EdgeCls in ALL_FREE_EDGES:
        signal = EdgeCls()
        result = run_event_study(store, signal, sector_map, costs=costs, oos_start=oos_start)
        card = edge_scorecard(result, costs)
        results.append(result)
        cards.append(card)
    return cards, results


def main() -> None:
    store, sector_map = build_synthetic_store()
    cards, results = run_all_edges(store, sector_map)
    print("\n" + "=" * 60)
    print("PHASE 1 — VALIDATION HARNESS (synthetic universe)")
    print("=" * 60)
    for card in cards:
        print(format_scorecard(card))
    summary = portfolio_summary(cards, results)
    print("\n--- Portfolio summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
