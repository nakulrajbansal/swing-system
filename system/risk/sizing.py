"""Risk-based position sizing (master §10).

shares = floor( equity * risk_per_trade / (entry - stop) ), stop from ATR.
Sizing only ever PROPOSES shares; the Risk Governor trims to the binding cap.
"""

from __future__ import annotations

import math


def atr_stop(entry: float, atr_value: float, mult: float) -> float:
    """Long protective stop a multiple of ATR below entry."""
    return entry - mult * atr_value


def position_size(equity: float, entry: float, stop: float, risk_per_trade: float) -> int:
    """Shares such that (entry - stop) * shares ~= risk_per_trade * equity."""
    per_share_risk = entry - stop
    if per_share_risk <= 0 or equity <= 0:
        return 0
    return int(math.floor(equity * risk_per_trade / per_share_risk))


def r_multiple(entry: float, stop: float, price: float) -> float:
    """How many R (entry-to-stop units) `price` is from entry."""
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return (price - entry) / risk
