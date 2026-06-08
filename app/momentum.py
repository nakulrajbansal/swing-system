"""Local record of open momentum swing trades, so the time-based exit can fire.

Alpaca positions don't carry an entry date, so we track entry/exit dates here,
keyed by environment (paper vs live kept separate). Stored next to the config.
"""

from __future__ import annotations

import json

from app.config import CONFIG_DIR

MOMENTUM_PATH = CONFIG_DIR / "momentum_positions.json"


def load_positions() -> dict:
    """{env: {symbol: {entry_date, exit_on, shares, entry_price}}}"""
    if MOMENTUM_PATH.exists():
        try:
            return json.loads(MOMENTUM_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_positions(data: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MOMENTUM_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
