"""Event study: the harness recovers a planted edge and kills noise."""

import pytest

from harness.data.loader import SyntheticConfig, SyntheticLoader
from harness.data.pit_store import PITStore
from harness.report.report import edge_scorecard
from harness.signals import Edge01Filing, Edge07Links
from harness.study.event_study import run_event_study


@pytest.fixture(scope="module")
def big_store(tmp_path_factory):
    """A larger universe so the planted edge clears the full institutional bar.

    Edge 1 is filing-triggered (sparse), so even a multi-year universe studies fast.
    """
    root = tmp_path_factory.mktemp("big")
    st = PITStore(root)
    SyntheticLoader(st, SyntheticConfig(n_symbols=10, start="2018-01-02",
                                        end="2023-12-29", seed=11)).load_all()
    return st, {f"SYN{i:02d}": ["XLK", "XLF", "XLE", "XLV", "XLY"][i % 5] for i in range(10)}


def test_planted_edge01_passes(big_store):
    store, sector_map = big_store
    result = run_event_study(store, Edge01Filing(), sector_map, oos_start="2022-01-01")
    card = edge_scorecard(result)
    assert card["n_events"] > 0
    assert card["spread"] > 0
    assert card["monotonic_quintiles"] is True
    assert card["tstat"] > 2.5
    assert card["verdict"] == "PASS"


def test_noise_edge07_does_not_pass(synth_store):
    store, sector_map = synth_store
    result = run_event_study(store, Edge07Links(), sector_map, oos_start="2022-01-01")
    card = edge_scorecard(result)
    assert card["verdict"] == "KILL"
