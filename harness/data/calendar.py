"""Minimal US equity exchange calendar (deterministic, no network).

Master design §9: an exchange calendar (holidays/half-days) with UTC internally;
post-close information maps to the T+1 entry window. We model trading sessions as
US business days minus a fixed federal/NYSE holiday set. Good enough for the
harness and paper engine; a vendor calendar can replace this behind the same API.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# NYSE-observed holidays are computed by rule so the calendar extends to any year.
_FIXED = {  # (month, day) holidays, shifted to nearest weekday when on a weekend
    (1, 1): "New Year's Day",
    (6, 19): "Juneteenth",
    (7, 4): "Independence Day",
    (12, 25): "Christmas",
}


def _observed(y: int, m: int, d: int) -> date:
    dt = date(y, m, d)
    wd = dt.weekday()
    if wd == 5:            # Saturday -> observed Friday
        return dt - timedelta(days=1)
    if wd == 6:            # Sunday -> observed Monday
        return dt + timedelta(days=1)
    return dt


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    offset = (weekday - d.weekday()) % 7
    return date(y, m, 1 + offset + (n - 1) * 7)


def _last_weekday(y: int, m: int, weekday: int) -> date:
    d = _nth_weekday(y, m, weekday, 4)
    nxt = d + timedelta(days=7)
    return d if nxt.month != m else nxt


def _good_friday(y: int) -> date:
    # Anonymous Gregorian (Meeus/Jones/Butcher) algorithm for Easter, minus 2 days.
    a = y % 19
    b, c = divmod(y, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    mth = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * mth + 114) // 31
    day = ((h + l - 7 * mth + 114) % 31) + 1
    return (pd.Timestamp(date(y, month, day)) - pd.Timedelta(days=2)).date()


def holidays(year: int) -> set[date]:
    h = {_observed(year, m, d) for (m, d) in _FIXED}
    h.add(_nth_weekday(year, 1, 0, 3))    # MLK day (3rd Mon Jan)
    h.add(_nth_weekday(year, 2, 0, 3))    # Presidents day (3rd Mon Feb)
    h.add(_last_weekday(year, 5, 0))      # Memorial day (last Mon May)
    h.add(_nth_weekday(year, 9, 0, 1))    # Labor day (1st Mon Sep)
    h.add(_nth_weekday(year, 11, 3, 4))   # Thanksgiving (4th Thu Nov)
    h.add(_good_friday(year))
    return h


def sessions(start, end) -> pd.DatetimeIndex:
    """Trading sessions (tz-naive dates) in [start, end] inclusive."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    bdays = pd.bdate_range(start, end)
    hol: set[date] = set()
    for y in range(start.year, end.year + 1):
        hol |= holidays(y)
    return pd.DatetimeIndex([d for d in bdays if d.date() not in hol])


def is_session(d) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.weekday() < 5 and d.date() not in holidays(d.year)


def next_session(d) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_session(d):
        d += pd.Timedelta(days=1)
    return d
