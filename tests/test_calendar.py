"""Exchange calendar: holidays and session navigation."""

import pandas as pd

from harness.data import calendar as cal


def test_known_2023_holidays_are_not_sessions():
    holidays = {
        "2023-01-02",  # New Year (observed Mon)
        "2023-01-16",  # MLK
        "2023-02-20",  # Presidents
        "2023-04-07",  # Good Friday
        "2023-05-29",  # Memorial
        "2023-06-19",  # Juneteenth
        "2023-07-04",  # Independence
        "2023-09-04",  # Labor
        "2023-11-23",  # Thanksgiving
        "2023-12-25",  # Christmas
    }
    for h in holidays:
        assert not cal.is_session(h), h


def test_regular_weekday_is_session():
    assert cal.is_session("2023-03-15")          # a plain Wednesday
    assert not cal.is_session("2023-03-18")      # Saturday


def test_sessions_exclude_weekends_and_holidays():
    s = cal.sessions("2023-07-01", "2023-07-07")
    days = {d.date().isoformat() for d in s}
    assert "2023-07-04" not in days              # holiday
    assert "2023-07-01" not in days              # Saturday
    assert "2023-07-03" in days and "2023-07-05" in days


def test_next_session_skips_holiday():
    # Day before Independence Day 2023 (Mon 07-03) -> next session is 07-05.
    assert cal.next_session("2023-07-03") == pd.Timestamp("2023-07-05")
