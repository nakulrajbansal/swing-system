"""Validation gate: which edges are allowed to trade live.

Master design principle: an edge joins the live system ONLY after passing
validation. We persist the edges that PASSED the most recent validation run, and
the live deliberation can be restricted to that set. Stored next to the config.
"""

from __future__ import annotations

import json
import time

from app.config import CONFIG_DIR

VALIDATED_PATH = CONFIG_DIR / "validated_edges.json"


def save_validated(passed: list[str], data_source: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATED_PATH.write_text(json.dumps({
        "passed": list(passed),
        "data_source": data_source,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


def load_validated() -> dict:
    if VALIDATED_PATH.exists():
        try:
            return json.loads(VALIDATED_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"passed": [], "data_source": None, "saved_at": None}
