"""Nasdaq-100 (QQQ) screening universe.

Tries the current constituents from Wikipedia; falls back to a broad static list.
Sectors reuse the S&P 500 GICS map (the two universes overlap heavily); names not
found there simply get no sector boost. Live screening only - never used by the
validation harness.
"""

from __future__ import annotations

import functools

# Broad static fallback of Nasdaq-100 constituents (QQQ holdings).
_STATIC: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "AVGO", "META", "TSLA", "GOOGL", "GOOG", "COST",
    "NFLX", "TMUS", "CSCO", "PEP", "ADBE", "AMD", "LIN", "QCOM", "TXN", "INTU",
    "AMGN", "ISRG", "AMAT", "BKNG", "HON", "CMCSA", "ADP", "VRTX", "GILD", "ADI",
    "MU", "LRCX", "REGN", "PANW", "KLAC", "SBUX", "MELI", "SNPS", "CDNS", "CRWD",
    "MAR", "PYPL", "ORLY", "ABNB", "FTNT", "ASML", "CTAS", "MRVL", "NXPI", "PCAR",
    "MNST", "ADSK", "WDAY", "ROP", "CPRT", "PAYX", "DASH", "KDP", "ODFL", "CEG",
    "FAST", "ROST", "BKR", "KHC", "EA", "DDOG", "EXC", "VRSK", "CCEP", "CSGP",
    "XEL", "CTSH", "TTWO", "IDXX", "ANSS", "ON", "DXCM", "BIIB", "GEHC", "CSX",
    "WBD", "ZS", "ILMN", "MDB", "TEAM", "GFS", "ARM", "SMCI", "LULU", "MRNA",
    "PDD", "TTD", "APP", "PLTR", "MSTR", "AEP", "PAYC", "WBA",
]


@functools.lru_cache(maxsize=1)
def nasdaq100_symbols() -> list[str]:
    """Current Nasdaq-100 tickers (live if possible, else the static fallback)."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            col = next((c for c in ("Ticker", "Symbol") if c in t.columns), None)
            if col is None:
                continue
            syms = [str(s).strip().upper().replace(".", "-") for s in t[col] if str(s).strip()]
            if len(syms) >= 90:
                return sorted(dict.fromkeys(syms))
    except Exception:
        pass
    return sorted(dict.fromkeys(_STATIC))


def screen_universe(limit: int | None = None) -> list[str]:
    syms = nasdaq100_symbols()
    return syms[:limit] if limit else syms
