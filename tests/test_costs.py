"""Costs: gap-aware fills and spread/slippage direction."""

import pytest

from harness.study.costs import CostModel, gap_aware_exit, net_pnl


def test_gap_aware_stop_fills_at_open_on_gap_down():
    # Long stop at 100; session opens at 96 (gapped through) -> fill 96, not 100.
    assert gap_aware_exit(100.0, 96.0, "stop") == 96.0
    # No gap (open above stop) -> resting stop price.
    assert gap_aware_exit(100.0, 101.0, "stop") == 100.0


def test_gap_aware_target_fills_at_open_on_gap_up():
    assert gap_aware_exit(110.0, 113.0, "target") == 113.0
    assert gap_aware_exit(110.0, 109.0, "target") == 110.0


def test_fill_price_pays_spread_in_adverse_direction():
    c = CostModel(half_spread_bps=5, slippage_bps=5)
    assert c.fill_price(100.0, "buy") > 100.0
    assert c.fill_price(100.0, "sell") < 100.0


def test_net_pnl_subtracts_two_commissions():
    c = CostModel(commission_per_share=0.01, min_commission=1.0,
                  half_spread_bps=0, slippage_bps=0)
    # 100 shares, +1.00 move -> gross 100, minus 2 * max(1, 100*0.01)=2*1=2.
    assert net_pnl(10.0, 11.0, 100, c) == pytest.approx(100.0 - 2.0)
