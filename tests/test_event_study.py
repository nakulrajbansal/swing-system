"""Event study: the harness recovers a planted edge and kills noise."""

import pandas as pd
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


def test_small_sample_never_passes():
    """A great-looking spread on too few events must KILL (anti-overfitting)."""
    from harness.study.event_study import EventStudyResult

    n = 13
    events = pd.DataFrame({"abn_20": [0.1] * n, "raw_score": list(range(n))})
    qt = pd.DataFrame({"mean": [0.0, 0.05, 0.1, 0.15, 0.2], "count": [3, 2, 3, 2, 3]},
                      index=[1, 2, 3, 4, 5])
    res = EventStudyResult(edge_id="x", windows=(20,), events=events, quintiles={20: qt},
                           spreads={20: {"spread": 0.2, "tstat": 14.0, "n": n}},
                           oos_spreads={20: {"spread": 0.1, "tstat": 5.0, "n": n}})
    card = edge_scorecard(res)
    assert card["n_events"] == 13 and card["enough_events"] is False
    assert card["verdict"] == "KILL"
