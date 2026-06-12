"""The watchlist: names whose TIME hasn't come yet.

Screens produce two kinds of almost-trades: WATCH-tier names (real interest,
not enough conviction today) and PM 'adjust' calls (sound thesis, extended
entry — wait for the pullback). Both used to evaporate when the run ended.
Now they persist here with concrete trigger levels, and the watch check
(--watch / 'Check watchlist') fires an alert the moment price actually enters
the entry window or breaks out on unusual volume — "right time", delivered.

Stored at ~/.swing_system/watchlist.json. Entries expire (~3 weeks) so the
list stays a watchlist, not a graveyard.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd

from app.config import CONFIG_DIR

WATCHLIST_PATH = CONFIG_DIR / "watchlist.json"
EXPIRE_DAYS = 21


def load(path=None) -> list[dict]:
    p = path or WATCHLIST_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save(items: list[dict], path=None) -> None:
    p = path or WATCHLIST_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


def add(new_items: list[dict], today: str, path=None) -> int:
    """Add/refresh watch entries (deduped per symbol; newest wins). Each item:
    symbol, reason, pullback_target?, breakout_level?, sector?."""
    items = {it["symbol"]: it for it in load(path)}
    added = 0
    expires = (datetime.date.fromisoformat(today)
               + datetime.timedelta(days=EXPIRE_DAYS)).isoformat()
    for it in new_items:
        sym = it.get("symbol")
        if not sym or not (it.get("pullback_target") or it.get("breakout_level")):
            continue
        items[sym] = {**it, "added": today, "expires": expires,
                      "rvol_trigger": it.get("rvol_trigger", 1.5)}
        added += 1
    save(sorted(items.values(), key=lambda x: x["symbol"]), path)
    return added


def active(today: str, path=None) -> list[dict]:
    """Live entries; expired ones are pruned (and persisted away)."""
    items = load(path)
    live = [it for it in items if str(it.get("expires", "")) >= today]
    if len(live) != len(items):
        save(live, path)
    return live


def remove(symbols: set[str], path=None) -> None:
    save([it for it in load(path) if it["symbol"] not in symbols], path)


def watch_hits(items: list[dict], closes: pd.DataFrame,
               volumes: pd.DataFrame | None = None) -> list[dict]:
    """Which watch entries have TRIGGERED: price entered the pullback window
    (<= target +1%), or took out the breakout level on unusual volume. Pure."""
    hits = []
    for it in items:
        sym = it.get("symbol")
        if sym not in getattr(closes, "columns", []):
            continue
        c = closes[sym].dropna()
        if c.empty:
            continue
        last = float(c.iloc[-1])
        tgt = it.get("pullback_target")
        brk = it.get("breakout_level")
        if tgt and last <= float(tgt) * 1.01:
            hits.append({**it, "kind": "pullback", "price": round(last, 2)})
            continue
        if brk and last >= float(brk) * 0.999:
            rvol = None
            if volumes is not None and sym in volumes.columns:
                v = volumes[sym].reindex(c.index).dropna()
                base = float(v.iloc[-21:-1].mean()) if len(v) > 21 else 0.0
                rvol = float(v.iloc[-1]) / base if base > 0 else None
            if rvol is None or rvol >= float(it.get("rvol_trigger", 1.5)):
                hits.append({**it, "kind": "breakout", "price": round(last, 2),
                             "rvol": round(rvol, 1) if rvol else None})
    return hits
