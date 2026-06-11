"""Shared Wikipedia table fetcher for the screening universes.

Wikipedia returns HTTP 403 to pandas' default urllib User-Agent, which made
every universe module silently fall back to its static list. Fetching with a
descriptive User-Agent via requests fixes that; parsing stays in pandas.
"""

from __future__ import annotations

from io import StringIO

_UA = "swing-system-screener/1.0 (local desktop research app)"


def read_tables(url: str) -> list:
    """All HTML tables on a Wikipedia page (raises on any failure)."""
    import pandas as pd
    import requests  # lazy

    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def constituents_table(url: str, min_rows: int):
    """The constituents table on `url`: the first table that has a ticker
    column and at least `min_rows` rows. Returns (DataFrame, ticker_column)
    or (None, None) if no table qualifies."""
    for t in read_tables(url):
        col = next((c for c in ("Symbol", "Ticker", "Ticker symbol")
                    if c in t.columns), None)
        if col is not None and len(t) >= min_rows:
            return t, col
    return None, None
