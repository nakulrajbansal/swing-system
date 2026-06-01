"""Unit tests for the corporate-action factor math (no store involved)."""

from __future__ import annotations

import pandas as pd
import pytest

from harness.data import corp_actions as ca

DATES = pd.bdate_range("2026-03-09", "2026-03-20")  # 10 sessions
EX = pd.Timestamp("2026-03-16")  # 6th session


def _prices(closes=None) -> pd.DataFrame:
    closes = closes if closes is not None else [100.0 + i for i in range(len(DATES))]
    return pd.DataFrame({"date": DATES, "close": closes,
                         "open": closes, "high": closes, "low": closes,
                         "volume": [1000] * len(DATES)})


def test_no_actions_is_identity():
    prices = _prices()
    empty = pd.DataFrame(columns=["symbol", "ex_date", "type", "ratio_or_amount"])
    f = ca.adjustment_factors(empty, prices, pd.Timestamp("2026-03-20"))
    assert (f == 1.0).all()


def test_split_back_adjusts_pre_ex_only():
    prices = _prices()
    actions = pd.DataFrame({"ex_date": [EX], "type": ["split"], "ratio_or_amount": [2.0]})
    f = ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
    assert (f[f.index < EX] == 0.5).all()
    assert (f[f.index >= EX] == 1.0).all()


def test_split_not_applied_when_ex_after_asof():
    prices = _prices()
    actions = pd.DataFrame({"ex_date": [EX], "type": ["split"], "ratio_or_amount": [2.0]})
    f = ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-13"))  # before ex
    assert (f == 1.0).all()


def test_reverse_split():
    prices = _prices()
    actions = pd.DataFrame({"ex_date": [EX], "type": ["split"], "ratio_or_amount": [0.5]})
    f = ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
    assert (f[f.index < EX] == 2.0).all()


def test_dividend_uses_prior_close():
    closes = [100.0 + i for i in range(len(DATES))]
    prices = _prices(closes)
    prior_close = prices.loc[prices["date"] < EX, "close"].iloc[-1]
    actions = pd.DataFrame({"ex_date": [EX], "type": ["dividend"], "ratio_or_amount": [2.0]})
    f = ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
    expected = 1.0 - 2.0 / prior_close
    assert f[f.index < EX].values == pytest.approx(expected)
    assert (f[f.index >= EX] == 1.0).all()


def test_combined_split_then_dividend():
    prices = _prices()
    div_ex = pd.Timestamp("2026-03-19")
    actions = pd.DataFrame(
        {
            "ex_date": [EX, div_ex],
            "type": ["split", "dividend"],
            "ratio_or_amount": [2.0, 1.0],
        }
    )
    f = ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
    prior_div_close = prices.loc[prices["date"] < div_ex, "close"].iloc[-1]
    dc = 1.0 - 1.0 / prior_div_close
    assert f[f.index < EX].values == pytest.approx(0.5 * dc)
    assert f[(f.index >= EX) & (f.index < div_ex)].values == pytest.approx(dc)
    assert (f[f.index >= div_ex] == 1.0).all()


def test_volume_factor_is_split_only():
    prices = _prices()
    actions = pd.DataFrame(
        {
            "ex_date": [EX, pd.Timestamp("2026-03-19")],
            "type": ["split", "dividend"],
            "ratio_or_amount": [2.0, 1.0],
        }
    )
    vf = ca.volume_adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
    assert (vf[vf.index < EX] == 2.0).all()   # dividend does not touch volume
    assert (vf[vf.index >= EX] == 1.0).all()


def test_unknown_type_raises():
    prices = _prices()
    actions = pd.DataFrame({"ex_date": [EX], "type": ["bogus"], "ratio_or_amount": [1.0]})
    with pytest.raises(ValueError, match="unknown corporate-action type"):
        ca.adjustment_factors(actions, prices, pd.Timestamp("2026-03-20"))
