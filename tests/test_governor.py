"""Risk Governor: hard caps, loss halts, and the asymmetric-autonomy invariant.

These tests guard load-bearing invariants (master §3/§10). They must not be
weakened.
"""

from system.config import SystemConfig
from system.risk.governor import GovernorContext, Position, RiskGovernor
from system.risk.sizing import position_size

CFG = SystemConfig()
GOV = RiskGovernor(CFG)


def ctx(**kw) -> GovernorContext:
    base = dict(equity=100_000.0, reference_price=100.0, atr_value=2.0, sector="XLK")
    base.update(kw)
    return GovernorContext(**base)


def test_pass_action_is_never_approved():
    assert not GOV.evaluate("AAA", "pass", ctx()).approved


def test_risk_based_sizing_when_no_cap_binds():
    # Wide stop (atr 20) makes the risk cap, not notional, the binding constraint.
    t = GOV.evaluate("AAA", "enter", ctx(reference_price=100.0, atr_value=20.0))
    assert t.approved
    expected = position_size(100_000, 100.0, 100.0 - 2 * 20.0, CFG.limits.risk_per_trade)
    assert t.shares == expected
    assert t.binding_cap == "risk_per_trade"
    assert t.stop == 100.0 - 2 * 20.0


def test_single_name_notional_cap_trims():
    t = GOV.evaluate("AAA", "enter", ctx(reference_price=100.0, atr_value=2.0))
    # risk sizing would be 250; 15% notional cap = 150 -> trimmed.
    assert t.shares == 150
    assert t.binding_cap == "max_single_name"


def test_governor_never_adds_beyond_sizing():
    # Asymmetric autonomy: approved shares <= risk-sized proposal, never more.
    for atr in (1.0, 5.0, 20.0, 50.0):
        t = GOV.evaluate("AAA", "enter", ctx(atr_value=atr))
        proposed = position_size(100_000, 100.0, 100.0 - 2 * atr, CFG.limits.risk_per_trade)
        assert t.shares <= proposed


def test_daily_loss_halt_blocks_new_entries():
    t = GOV.evaluate("AAA", "enter", ctx(daily_pnl_pct=-0.03))
    assert not t.approved and t.binding_cap == "daily_loss_halt"


def test_weekly_loss_halt_blocks_new_entries():
    t = GOV.evaluate("AAA", "enter", ctx(weekly_pnl_pct=-0.08))
    assert not t.approved and t.binding_cap == "weekly_loss_halt"


def test_max_open_positions_blocks_new_name():
    positions = [Position(f"S{i}", 10, 100, 96, "XLF", 100) for i in range(8)]
    t = GOV.evaluate("AAA", "enter", ctx(positions=positions))
    assert not t.approved and t.binding_cap == "max_open_positions"


def test_portfolio_heat_cap_trims():
    # Existing open risk 5,800 of a 6,000 (6%) budget -> only 200 headroom.
    heavy = Position("X", 145, 100, 60, "XLF", 100)   # open_risk = 145*40 = 5800
    t = GOV.evaluate("AAA", "enter", ctx(reference_price=100.0, atr_value=20.0,
                                         positions=[heavy]))
    # per-share risk 40, headroom 200 -> 5 shares.
    assert t.shares == 5
    assert t.binding_cap == "max_portfolio_heat"


def test_below_minimum_viable_is_rejected():
    t = GOV.evaluate("AAA", "enter", ctx(equity=50.0, reference_price=100.0, atr_value=20.0))
    assert not t.approved
