"""The leakage CI test — the heart of build step 1.

Asserts the point-in-time store never reveals information from after the
decision time T. This test must never be weakened (CLAUDE.md non-negotiables).
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from harness.data import assert_no_leakage
from harness.data.pit_store import EVENT_TABLES
from tests.conftest import DIV_AMOUNT, DIV_EX, SPLIT_EX, SPLIT_RATIO


def post_close(date: str) -> pd.Timestamp:
    """A post-close decision instant on the given trading date (UTC)."""
    return pd.Timestamp(f"{date} 21:00", tz="UTC")


FIXED_PROBES = [
    post_close(d)
    for d in [
        "2026-01-05", "2026-02-10", "2026-03-13", "2026-03-16",
        "2026-04-01", "2026-04-14", "2026-04-15", "2026-05-01", "2026-05-29",
    ]
]


def _random_probes(n: int = 50, seed: int = 1234) -> list[pd.Timestamp]:
    rng = random.Random(seed)
    span = (pd.Timestamp("2026-05-29") - pd.Timestamp("2026-01-02")).days
    out = []
    for _ in range(n):
        day = pd.Timestamp("2026-01-02") + pd.Timedelta(days=rng.randint(0, span))
        hour = rng.randint(0, 23)
        out.append(pd.Timestamp(f"{day.date()} {hour:02d}:00", tz="UTC"))
    return out


ALL_PROBES = FIXED_PROBES + _random_probes()


# --- 1. Universal canary --------------------------------------------------
@pytest.mark.parametrize("T", ALL_PROBES, ids=lambda t: str(t))
def test_no_leakage_canary(store, T):
    view = store.as_of(T)
    assert assert_no_leakage(view)

    # Belt-and-suspenders: re-check the time columns directly per table.
    for table, col in EVENT_TABLES.items():
        df = getattr(view, table)()
        if not df.empty:
            assert (pd.to_datetime(df[col]) <= T).all(), f"{table} leaked"
    for symbol in ("AAA", "BBB"):
        df = view.prices(symbol, adjust=False)
        if not df.empty:
            assert (df["date"] <= view.asof_date).all()


def test_as_of_rejects_naive_timestamp(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        store.as_of(pd.Timestamp("2026-03-01"))


# --- 2. Corp-action non-leakage ------------------------------------------
def _factor_by_date(view) -> pd.Series:
    raw = view.prices("AAA", adjust=False).set_index("date")["close"]
    adj = view.prices("AAA", adjust=True).set_index("date")["close"]
    return (adj / raw).reindex(raw.index)


def test_split_not_applied_before_ex(store):
    # Day before the split ex-date: the split is in the future, must not apply.
    view = store.as_of(post_close("2026-03-13"))
    factors = _factor_by_date(view)
    assert factors.values == pytest.approx(1.0)


def test_split_applied_on_and_after_ex(store):
    # On the ex-date itself the action IS applied (ex_date == asof_date).
    view = store.as_of(post_close("2026-04-01"))
    factors = _factor_by_date(view)
    pre = factors[factors.index < SPLIT_EX]
    post = factors[factors.index >= SPLIT_EX]
    assert pre.eq(1.0 / SPLIT_RATIO).all()      # pre-ex history halved
    assert post.eq(1.0).all()                   # current bars unchanged


def test_dividend_back_adjustment(store):
    view = store.as_of(post_close("2026-05-01"))
    raw = view.prices("AAA", adjust=False).set_index("date")["close"]
    factors = _factor_by_date(view)

    close_prior = raw[raw.index < DIV_EX].iloc[-1]
    div_contrib = 1.0 - DIV_AMOUNT / close_prior

    pre_split = factors[factors.index < SPLIT_EX]
    between = factors[(factors.index >= SPLIT_EX) & (factors.index < DIV_EX)]
    after_div = factors[factors.index >= DIV_EX]

    assert pre_split.values == pytest.approx(div_contrib / SPLIT_RATIO)
    assert between.values == pytest.approx(div_contrib)
    assert after_div.values == pytest.approx(1.0)


def test_future_corp_action_invisible_changes_history(store):
    # The SAME historical bar must read differently before vs after the split
    # becomes visible — proving adjustment is genuinely as-of, not future-baked.
    hist_date = pd.Timestamp("2026-02-02")
    before = store.as_of(post_close("2026-03-13")).prices("AAA").set_index("date")
    after = store.as_of(post_close("2026-04-01")).prices("AAA").set_index("date")
    assert after.loc[hist_date, "close"] == pytest.approx(
        before.loc[hist_date, "close"] / SPLIT_RATIO
    )


# --- 3. Monotonicity of information --------------------------------------
def test_information_only_grows(store):
    ordered = sorted(ALL_PROBES)
    for table, col in EVENT_TABLES.items():
        seen: set = set()
        for T in ordered:
            df = getattr(store.as_of(T), table)()
            now = set(pd.to_datetime(df[col])) if not df.empty else set()
            assert seen <= now, f"{table}: visible set shrank as T advanced"
            seen = now


# --- 4. Universe PIT (survivorship-free membership) ----------------------
def test_universe_membership_is_point_in_time(store):
    assert store.as_of(post_close("2026-03-15")).universe() == ["AAA", "BBB"]
    # BBB end_date 2026-04-01 is exclusive: gone on and after that session.
    assert store.as_of(post_close("2026-04-01")).universe() == ["AAA"]
    assert store.as_of(post_close("2026-04-02")).universe() == ["AAA"]
    # CCC not listed until 2026-05-01.
    assert store.as_of(post_close("2026-04-30")).universe() == ["AAA"]
    assert store.as_of(post_close("2026-05-01")).universe() == ["AAA", "CCC"]


# --- 5. Boundary equality (<= semantics) ---------------------------------
def test_available_at_equal_to_T_is_visible(store):
    exact = pd.Timestamp("2026-02-10 14:00", tz="UTC")  # filing acc-1 time
    filings = store.as_of(exact).filings("AAA")
    assert "acc-1" in set(filings["accession"])
    # One microsecond earlier it must be invisible.
    just_before = exact - pd.Timedelta(microseconds=1)
    assert "acc-1" not in set(store.as_of(just_before).filings("AAA")["accession"])


def test_price_bar_on_decision_date_is_visible(store):
    view = store.as_of(post_close("2026-02-10"))
    prices = view.prices("AAA", adjust=False)
    assert prices["date"].max() == pd.Timestamp("2026-02-10")
