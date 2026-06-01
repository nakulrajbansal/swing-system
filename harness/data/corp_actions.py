"""Corporate-action as-of adjustment (splits + cash dividends).

Point-in-time discipline (master design §9, harness spec §4):

  * Raw prices are immutable ground truth. We NEVER store future-adjusted
    prices. The adjustment is a pure function of (raw prices, corp_actions,
    as-of trading date) computed at query time.
  * An `as_of(T)` query applies ONLY corporate actions with `ex_date <= T`.
    An action with `ex_date > T` is in the future and must never touch the
    prices returned — that would be leakage.

Back-adjustment convention (CRSP-style total return). For a query as of trading
date `T`, the multiplicative factor applied to a bar on date `d` is the
cumulative product of the contributions of every action whose ex-date falls in
`(d, T]`:

  * Split with ratio `r` (2-for-1 -> r=2.0, reverse 1-for-2 -> r=0.5):
    pre-ex prices are divided by `r`           -> price contribution 1/r
    pre-ex volume is multiplied by `r`         -> volume contribution r
  * Cash dividend of amount `D` (ex_date):
    pre-ex prices are multiplied by (1 - D / close_{prior}), where
    `close_{prior}` is the raw close on the last session strictly before the
    ex-date. Dividends do not adjust volume.

The most recent visible bar (date == latest session <= T) always has factor
1.0, i.e. it carries the actual traded price. All functions operate on a single
symbol's data; the caller filters by symbol.
"""

from __future__ import annotations

import pandas as pd

SPLIT = "split"
DIVIDEND = "dividend"
VALID_TYPES = frozenset({SPLIT, DIVIDEND})


def _as_date(value) -> pd.Timestamp:
    """Normalise a date-like value to a tz-naive midnight Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _applicable(corp_actions: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    """Actions visible as of `asof_date`, sorted by ex_date (PIT filter)."""
    if corp_actions.empty:
        return corp_actions
    ca = corp_actions.copy()
    ca["ex_date"] = ca["ex_date"].map(_as_date)
    ca = ca[ca["ex_date"] <= asof_date]
    return ca.sort_values("ex_date").reset_index(drop=True)


def adjustment_factors(
    corp_actions: pd.DataFrame,
    prices: pd.DataFrame,
    asof_date,
) -> pd.Series:
    """Price back-adjustment factor per trading date, as of `asof_date`.

    Multiply raw open/high/low/close by this factor to get the as-of-T adjusted
    series. Only actions with ``ex_date <= asof_date`` are applied.

    Parameters
    ----------
    corp_actions : columns [ex_date, type, ratio_or_amount] for ONE symbol.
    prices : columns [date, close, ...] for the same symbol (raw, unadjusted).
    asof_date : the decision trading date (date-like).
    """
    asof_date = _as_date(asof_date)
    dates = prices["date"].map(_as_date)
    factor = pd.Series(1.0, index=dates.values)

    ca = _applicable(corp_actions, asof_date)
    if ca.empty:
        return factor

    raw_close = pd.Series(prices["close"].to_numpy(), index=dates.values)

    for action in ca.itertuples(index=False):
        ex_date = action.ex_date
        atype = action.type
        if atype not in VALID_TYPES:
            raise ValueError(f"unknown corporate-action type: {atype!r}")
        amount = float(action.ratio_or_amount)
        pre_ex = factor.index < ex_date
        if not pre_ex.any():
            continue  # nothing before the ex-date to back-adjust
        if atype == SPLIT:
            if amount <= 0:
                raise ValueError(f"split ratio must be > 0, got {amount}")
            contribution = 1.0 / amount
        else:  # DIVIDEND
            prior = raw_close.index[raw_close.index < ex_date]
            if len(prior) == 0:
                continue  # no prior close to anchor the dividend; no effect
            close_prior = float(raw_close.loc[prior[-1]])
            if close_prior <= 0:
                continue
            contribution = 1.0 - amount / close_prior
        factor.loc[pre_ex] *= contribution

    return factor


def volume_adjustment_factors(
    corp_actions: pd.DataFrame,
    prices: pd.DataFrame,
    asof_date,
) -> pd.Series:
    """Volume back-adjustment factor per trading date (splits only)."""
    asof_date = _as_date(asof_date)
    dates = prices["date"].map(_as_date)
    factor = pd.Series(1.0, index=dates.values)

    ca = _applicable(corp_actions, asof_date)
    if ca.empty:
        return factor

    for action in ca.itertuples(index=False):
        if action.type != SPLIT:
            continue
        ratio = float(action.ratio_or_amount)
        if ratio <= 0:
            raise ValueError(f"split ratio must be > 0, got {ratio}")
        pre_ex = factor.index < action.ex_date
        factor.loc[pre_ex] *= ratio

    return factor


def apply_adjustment(
    prices: pd.DataFrame,
    price_factors: pd.Series,
    volume_factors: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copy of `prices` with OHLC (and volume) back-adjusted.

    Factors are aligned to `prices['date']` positionally; the caller produced
    both from the same price frame.
    """
    out = prices.copy()
    pf = price_factors.to_numpy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].to_numpy() * pf
    if volume_factors is not None and "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy() * volume_factors.to_numpy()
    return out
