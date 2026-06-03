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
    # Paper constructs fine (no network on construction).
    paper = AlpacaBroker("key", "secret")
    assert paper.env == "paper" and "paper-api" in paper.base
    # Asymmetric autonomy: LIVE (real money) requires explicit confirmation.
    with pytest.raises(RuntimeError, match="confirm_live|real-money"):
        AlpacaBroker("key", "secret", env="live")
    # Live with confirmation points at the live endpoint.
    assert "//api.alpaca" in AlpacaBroker("k", "s", env="live", confirm_live=True).base
    # Missing credentials always rejected.
    with pytest.raises(RuntimeError, match="credentials"):
        AlpacaBroker(None, None)


def test_closed_trades_have_protective_exit_reasons(synth_store):
    store, sector_map = synth_store
    engine = PaperTradingEngine(store, sector_map)
    result = engine.run(start="2022-06-01", end="2022-12-30")
    for t in result.closed_trades:
        assert t["reason"] in {"stop", "target", "time", "guardian"}
