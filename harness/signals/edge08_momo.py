"""Edge 8 — price / 52-week-high momentum (Family E, long). Prices only, free.

Mechanism: stocks near their 52-week high with strong trailing momentum tend to
continue (under-reaction to good news). Trigger: price within 5% of its 52-week
high. Score: proximity to high blended with 126-session momentum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_52W = 252
MOMENTUM_LOOKBACK = 126
PROXIMITY = 0.05


class Edge08Momentum:
    edge_id = "edge08_momo"
    family = "E"
    direction = "long"

    def triggers(self, view, date: pd.Timestamp) -> list[str]:
        out = []
        for sym in view.universe():
            px = view.prices(sym, adjust=True)
            if len(px) < LOOKBACK_52W:
                continue
            window = px["close"].iloc[-LOOKBACK_52W:]
            last = window.iloc[-1]
            high = window.max()
            if high > 0 and last >= high * (1 - PROXIMITY):
                out.append(sym)
        return sorted(out)

    def score(self, view, symbol: str, date: pd.Timestamp) -> dict:
        px = view.prices(symbol, adjust=True)
        if len(px) < LOOKBACK_52W:
            return {"raw_score": 0.0, "confidence": 0.0, "evidence": {}}
        close = px["close"]
        window = close.iloc[-LOOKBACK_52W:]
        last, high = window.iloc[-1], window.max()
        proximity = float(last / high) if high > 0 else 0.0
        mom = float(last / close.iloc[-MOMENTUM_LOOKBACK] - 1.0) \
            if len(close) > MOMENTUM_LOOKBACK else 0.0
        raw = 0.5 * proximity + 0.5 * float(np.tanh(mom * 3))
        return {"raw_score": raw, "confidence": float(min(1.0, 0.4 + proximity / 2)),
                "evidence": {"proximity_to_52wk_high": proximity, "momentum_126d": mom}}
