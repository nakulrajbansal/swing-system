"""Deterministic technical indicators (master §9 feature engine).

Pure functions over an OHLCV frame (already point-in-time / corp-action adjusted
by the store). No look-ahead: every value at row i uses only rows <= i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rolling_high(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).max()


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return close.pct_change().rolling(n).std()


def last_atr(df: pd.DataFrame, n: int = 14) -> float:
    """Most recent ATR value (NaN-safe)."""
    a = atr(df, n).dropna()
    return float(a.iloc[-1]) if len(a) else float("nan")
