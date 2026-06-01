"""End-to-end paper-trading engine: it runs, respects caps, and stays paper-only."""

import pytest

from system.config import SystemConfig
from system.execution.broker import AlpacaBroker, PaperBroker
from system.run_live import PaperTradingEngine


def test_engine_runs_end_to_end(synth_store):
    store, sector_map = synth_store
    engine = PaperTradingEngine(store, sector_map, starting_equity=100_000.0)
    result = engine.run(start="2022-06-01", end="2022-12-30")
    assert result.n_cycles > 0
    assert result.final_equity > 0
    assert "trades" in result.scorecard
    # Hard cap respected: never more than max_open_positions held at the end.
    assert len(engine.broker.positions()) <= SystemConfig().limits.max_open_positions


def test_engine_is_paper_only_by_default():
    assert SystemConfig().paper_only is True


def test_engine_uses_paper_broker(synth_store):
    store, sector_map = synth_store
    engine = PaperTradingEngine(store, sector_map)
    assert isinstance(engine.broker, PaperBroker)


def test_live_broker_is_gated():
    # Asymmetric autonomy: live trading cannot be turned on without explicit opt-in.
    with pytest.raises(RuntimeError, match="disabled"):
        AlpacaBroker("key", "secret")
    with pytest.raises(RuntimeError, match="credentials"):
        AlpacaBroker(None, None, enable_live=True)


def test_closed_trades_have_protective_exit_reasons(synth_store):
    store, sector_map = synth_store
    engine = PaperTradingEngine(store, sector_map)
    result = engine.run(start="2022-06-01", end="2022-12-30")
    for t in result.closed_trades:
        assert t["reason"] in {"stop", "target", "time", "guardian"}
