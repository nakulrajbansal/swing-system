"""Recommendation ledger: persist every BUY/ADJUST the desk recommends and later
score how it actually performed - whether or not the trade was executed.

This closes the learning loop for advice (not just executed trades): each
recommendation is saved with its entry/stop/target and hold window, and once the
window elapses its forward return is measured from price history and fed into the
shared LessonMemory (so the agents learn from their own calls). Stored at:

    ~/.swing_system/recommendations.json
"""

from __future__ import annotations

import json

import pandas as pd

from app.config import CONFIG_DIR
from system.reflection.memory import TradeOutcome
from system.schemas import Lesson

LEDGER_PATH = CONFIG_DIR / "recommendations.json"


def load(path=None) -> list[dict]:
    p = path or LEDGER_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save(records: list[dict], path=None) -> None:
    p = path or LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def record(recs: list[dict], source: str, as_of: str, path=None) -> int:
    """Append BUY/ADJUST recommendations as open ledger entries (idempotent per
    day+symbol+source). Returns how many new entries were added."""
    led = load(path)
    have = {(r["date"], r["symbol"], r.get("source")) for r in led}
    added = 0
    for r in recs:
        key = (as_of, r["symbol"], source)
        if key in have:
            continue
        led.append({
            "id": f"{as_of}:{r['symbol']}:{source}",
            "date": as_of, "symbol": r["symbol"], "action": "buy",
            "sector": r.get("sector", "?"),
            "entry": r.get("entry"), "stop": r.get("stop"), "target": r.get("target"),
            "conviction": r.get("conviction"), "hold_days": r.get("hold_days", 10),
            "exit_by": r.get("exit_by"), "thesis": (r.get("thesis") or "")[:300],
            "setup_type": "confluence_swing", "source": source, "status": "open",
            # Cohort tags: which lens produced the pick, so the desk can learn
            # whether hidden-gem and moat-bullish picks actually outperform.
            "hidden_gem": bool(r.get("hidden_gem")),
            "moat_stance": r.get("moat_stance"),
        })
        have.add(key)
        added += 1
    if added:
        save(led, path)
    return added


def _price_on_or_after(series: pd.Series, when: str):
    idx = series.index[series.index >= pd.Timestamp(when)]
    if len(idx):
        v = float(series.loc[idx[0]])
        return v if v == v else None
    return None


def evaluate(closes: pd.DataFrame, today: str, memory=None, path=None) -> dict:
    """Score matured open recommendations from price history; feed outcomes into
    LessonMemory. Returns a summary. `closes` is the wide adjusted-close frame the
    screen already downloads."""
    led = load(path)
    evaluated = wins = 0
    rets = []
    changed = False
    for r in led:
        if r.get("status") != "open" or not r.get("exit_by"):
            continue
        if r["exit_by"] > today:                      # window not elapsed yet
            continue
        sym = r["symbol"]
        if sym not in closes.columns:
            continue
        s = closes[sym].dropna()
        entry = _price_on_or_after(s, r["date"])
        exit_px = _price_on_or_after(s, r["exit_by"])
        if exit_px is None and len(s):                # matured past data end -> last
            exit_px = float(s.iloc[-1])
        if entry is None or exit_px is None or entry <= 0:
            continue
        pnl_pct = (exit_px / entry - 1) * 100.0
        # Close-path reason (no intraday): stop checked first (conservative).
        path_s = s[(s.index >= pd.Timestamp(r["date"])) & (s.index <= pd.Timestamp(r["exit_by"]))]
        reason = "time"
        if r.get("stop") and len(path_s) and float(path_s.min()) <= float(r["stop"]):
            reason, pnl_pct = "stop", (float(r["stop"]) / entry - 1) * 100.0
        elif r.get("target") and len(path_s) and float(path_s.max()) >= float(r["target"]):
            reason, pnl_pct = "target", (float(r["target"]) / entry - 1) * 100.0
        r["status"] = "evaluated"
        r["return_pct"] = round(pnl_pct, 2)
        r["outcome"] = reason
        r["evaluated_on"] = today
        evaluated += 1
        wins += 1 if pnl_pct > 0 else 0
        rets.append(pnl_pct)
        changed = True
        if memory is not None:
            conv = float(r.get("conviction") or 0.0)
            memory.record_outcome(TradeOutcome("confluence_swing", sym, conv,
                                               round(pnl_pct, 2), reason, r["exit_by"]))
            verb = "paid" if pnl_pct > 0 else "did not pay"
            memory.add(Lesson("confluence_swing",
                              f"recommended {sym} {verb} {pnl_pct:+.1f}% (exit: {reason}).",
                              pnl_pct > 0, "clean" if reason in {"target", "time"} else "stopped",
                              symbol=sym, as_of=r["exit_by"], pnl_pct=round(pnl_pct, 2),
                              conviction=conv), human_reviewed=True)
    if changed:
        save(led, path)
    avg = round(sum(rets) / len(rets), 1) if rets else 0.0
    return {"evaluated": evaluated, "win_rate_pct": round(100 * wins / evaluated, 0) if evaluated else 0,
            "avg_return_pct": avg, "open": sum(1 for r in led if r.get("status") == "open"),
            "cohorts": cohort_stats(led)}


def _cohort(rows: list[dict]) -> dict:
    rets = [r["return_pct"] for r in rows if isinstance(r.get("return_pct"), (int, float))]
    if not rets:
        return {"n": 0}
    wins = sum(1 for x in rets if x > 0)
    return {"n": len(rets), "win_rate_pct": round(100 * wins / len(rets), 0),
            "avg_return_pct": round(sum(rets) / len(rets), 1)}


def cohort_stats(led: list[dict] | None = None, path=None) -> dict:
    """Performance split by the lens that produced each pick, over ALL scored
    recommendations: hidden-gem vs core, and moat-bullish vs the rest. This is
    how the desk learns whether the discovery lenses actually pay."""
    rows = [r for r in (led if led is not None else load(path))
            if r.get("status") == "evaluated"]
    return {
        "hidden_gem": _cohort([r for r in rows if r.get("hidden_gem")]),
        "core": _cohort([r for r in rows if not r.get("hidden_gem")]),
        "moat_bullish": _cohort([r for r in rows if r.get("moat_stance") == "bullish"]),
        "moat_other": _cohort([r for r in rows if r.get("moat_stance") not in (None, "bullish")]),
    }


def summarize(path=None) -> str:
    led = load(path)
    if not led:
        return "No recommendations recorded yet."
    done = [r for r in led if r.get("status") == "evaluated"]
    openr = [r for r in led if r.get("status") == "open"]
    lines = [f"Recommendation ledger: {len(led)} total - {len(openr)} open, "
             f"{len(done)} scored."]
    if done:
        wins = sum(1 for r in done if (r.get("return_pct") or 0) > 0)
        avg = sum(r.get("return_pct") or 0 for r in done) / len(done)
        lines.append(f"  scored hit rate {100 * wins / len(done):.0f}%  |  "
                     f"avg forward return {avg:+.1f}%")
        for r in sorted(done, key=lambda x: x.get("return_pct") or 0, reverse=True)[:10]:
            lines.append(f"   {r['date']}  {r['symbol']:<6} {r.get('return_pct', 0):+6.1f}%  "
                         f"({r.get('outcome', '?')})")
    if openr:
        lines.append("  open (awaiting their exit date):")
        for r in openr[-8:]:
            lines.append(f"   {r['date']}  {r['symbol']:<6} entry {r.get('entry')}  "
                         f"-> exit by {r.get('exit_by')}")
    return "\n".join(lines)
