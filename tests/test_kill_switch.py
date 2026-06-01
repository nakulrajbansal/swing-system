"""Kill switch: reduce-only escalation on drawdown / reconciliation / slippage."""

from system.config import KillCriteria
from system.monitoring.kill_switch import FLATTEN, HALT_NEW, OK, KillSwitch
from system.monitoring.scorecard import HealthScorecard

KS = KillSwitch(KillCriteria())


def _scorecard_with_drawdown(dd: float) -> HealthScorecard:
    sc = HealthScorecard()
    sc.equity_curve = [100_000.0, 100_000.0 * (1 + dd)]
    return sc


def test_clean_state_is_ok():
    assert KS.evaluate(HealthScorecard()).action == OK


def test_strategy_drawdown_halts_new_entries():
    v = KS.evaluate(_scorecard_with_drawdown(-0.16))
    assert v.action in {HALT_NEW, FLATTEN}


def test_system_drawdown_flattens():
    v = KS.evaluate(_scorecard_with_drawdown(-0.25))
    assert v.action == FLATTEN


def test_reconciliation_failure_flattens():
    v = KS.evaluate(HealthScorecard(), reconciliation_ok=False)
    assert v.action == FLATTEN


def test_actions_are_always_reduce_only():
    for dd in (-0.01, -0.16, -0.25):
        assert KS.evaluate(_scorecard_with_drawdown(dd)).reduce_only
