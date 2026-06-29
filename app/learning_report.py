"""Learning report: the synthesis behind the Learning tab.

Turns the raw ledger + lesson memory into a structured, human-readable picture of
how the desk is learning:

  * LEARNINGS — the active lessons and what the realized record says by lens.
  * PARAMETERS — every knob the desk adapts from its own results, WHICH AGENTS
    each knob moves, and how much evidence is behind it.
  * STRATEGY — the current configuration in plain English, plus how it has
    drifted over time (a dated strategy journal).
  * READINESS — an evidence-gated recommendation (see `readiness.py`) for when
    the desk has earned a tiny real-capital allocation. Recommendation only:
    enabling real money stays a human act.

IO lives here (app layer); the math is the pure `system.reflection.readiness`.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from app import reco_ledger
from app.config import CONFIG_DIR
from app.learning import load_memory
from system.reflection import readiness
from system.reflection.calibration import (calibration_table, desk_throttle,
                                           MIN_CALIBRATED)

JOURNAL_PATH = CONFIG_DIR / "strategy_journal.json"

# Which agents each adaptive knob actually moves (traced through the engine).
_AFFECTS = {
    "calibration": "Portfolio Manager — win-probability shown on every plan and "
                   "the conviction it will size from",
    "throttle": "Entry gate + Risk sizing — raises the conviction bar and scales "
                "gross exposure down (risk-reducing only)",
    "weights": "Screener / pre-filter — which names even reach the analyst agents",
    "lessons": "All deliberation agents — macro, technical, fundamental, "
               "valuation, growth, moat, strategist, skeptic and the PM read the "
               "recalled lessons + base rates",
}


def _equity_curve(scored: list[dict]) -> list[float]:
    """Compounded curve of the scored forward returns, oldest→newest."""
    rows = sorted(scored, key=lambda r: str(r.get("evaluated_on") or ""))
    eq, v = [1.0], 1.0
    for r in rows:
        v *= 1.0 + float(r["return_pct"]) / 100.0
        eq.append(round(v, 4))
    return eq


def _shadow_days(led: list[dict]) -> int | None:
    dates = [str(r.get("date") or "") for r in led if r.get("date")]
    if not dates:
        return None
    try:
        first = datetime.date.fromisoformat(min(dates))
        return (datetime.date.today() - first).days
    except ValueError:
        return None


def _latest_preset() -> dict | None:
    """The most recently tuned weight preset (the nightly walk-forward winner),
    read from the screen's cache. None if self-tuning has not run yet."""
    cache = CONFIG_DIR / "data_store"
    files = sorted(cache.glob("preset_*.json")) if cache.exists() else []
    for p in reversed(files):
        try:
            sel = json.loads(p.read_text(encoding="utf-8"))
            if sel.get("name"):
                return sel
        except Exception:
            continue
    return None


def _preset_tilt(name: str) -> str:
    """A plain-English description of how a preset tilts the screen vs base."""
    return {
        "base": "balanced trend + momentum + relative-strength weighting",
        "timing": "leans on entry timing and momentum acceleration (buy strength "
                  "resting on its 20-DMA, not extended)",
        "discovery": "leans on acceleration and relative strength to surface "
                     "earlier-stage leaders before the crowd",
        "defensive": "leans hard on relative strength and the 200-DMA trend, "
                     "fades raw 3-month chasing — a risk-off tilt",
    }.get(name, "custom weighting")


def build_report(led: list[dict] | None = None, mem=None) -> dict:
    """Assemble the full structured learning report."""
    led = reco_ledger.load() if led is None else led
    mem = load_memory() if mem is None else mem
    scored = [r for r in led if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))]
    n = len(scored)
    wins = sum(1 for r in scored if r["return_pct"] > 0)
    avg = (sum(r["return_pct"] for r in scored) / n) if n else 0.0
    excess = [r["excess_pct"] for r in scored if isinstance(r.get("excess_pct"), (int, float))]

    table = calibration_table(led)
    bands = table.get("bands", [])
    throttle = desk_throttle(led)
    preset = _latest_preset()
    cohorts = reco_ledger.cohort_stats(led)

    rd = readiness.assess(scored, equity_curve=_equity_curve(scored),
                          calibration_bands=bands, shadow_days=_shadow_days(led))

    # -- PARAMETERS the desk adapts from its own record ----------------------
    cal_conf = (f"calibrated on {table.get('n_total', 0)} scored calls"
                if table.get("n_total", 0) >= MIN_CALIBRATED
                else f"not yet calibrated (only {table.get('n_total', 0)} scored)")
    band_txt = "  ".join(f"{b['lo']:.2f}-{b['hi']:.2f}→{b['win_rate_pct']:.0f}%"
                         for b in bands if b.get("n") and b.get("win_rate_pct") is not None)
    parameters = [
        {"param": "Conviction calibration",
         "value": band_txt or "—",
         "affects": _AFFECTS["calibration"], "confidence": cal_conf},
        {"param": "Desk throttle",
         "value": (f"ACTIVE — +{throttle['conviction_bump']:.2f} conviction bar, "
                   f"gross ×{throttle['gross_scale']:.2f} ({throttle['reason']})"
                   if throttle["active"] else "dormant (record not poor enough to de-risk)"),
         "affects": _AFFECTS["throttle"],
         "confidence": f"keyed off {throttle['n']} scored calls"},
        {"param": "Factor weights (nightly preset + regime)",
         "value": (f"'{preset['name']}' — {_preset_tilt(preset['name'])}"
                   if preset else "base (self-tuner has not run yet today)"),
         "affects": _AFFECTS["weights"],
         "confidence": "in-sample walk-forward selection — advisory tilt only"},
        {"param": "Lesson memory (advisory recall)",
         "value": f"{sum(1 for e in mem.entries if e.human_reviewed)} active "
                  f"lesson(s); base rate {n} trades, {(100*wins/n if n else 0):.0f}% win",
         "affects": _AFFECTS["lessons"],
         "confidence": "advisory only — never changes a risk limit"},
    ]

    # -- LEARNINGS (active lessons + cohort performance) ---------------------
    active = [e.lesson for e in mem.entries if e.human_reviewed]
    pattern = [l for l in active if l.kind]
    anecdotes = [l for l in active if not l.kind]
    learnings = {
        "patterns": [{"kind": l.kind, "text": l.lesson, "worked": l.thesis_correct}
                     for l in pattern],
        "recent_anecdotes": [{"text": l.lesson, "as_of": l.as_of}
                             for l in anecdotes[-8:]],
        "pending": sum(1 for e in mem.entries if not e.human_reviewed),
        "cohorts": cohorts,
    }

    # -- STRATEGY (current configuration, one paragraph) ---------------------
    strat_bits = []
    strat_bits.append(f"Screening with the "
                      f"{('‘' + preset['name'] + '’ preset — ' + _preset_tilt(preset['name'])) if preset else 'base weighting'}")
    strat_bits.append("conviction is shrunk toward the realized base rate before "
                      "it sizes anything" if table.get("n_total", 0) >= MIN_CALIBRATED
                      else "conviction is still taken at face value (too few scored calls to calibrate)")
    strat_bits.append("the throttle is HOLDING SIZE DOWN — " + throttle["reason"]
                      if throttle["active"] else "the throttle is dormant (full plan size)")
    strategy = {
        "preset": preset["name"] if preset else "base",
        "throttle_active": throttle["active"],
        "active_lesson_count": len(active),
        "summary": "; ".join(strat_bits) + ".",
        "top_lessons": [l.lesson for l in (pattern + anecdotes[::-1])[:5]],
    }

    # -- EVOLUTION (monthly cohorts + the dated strategy journal) ------------
    monthly = _monthly(scored)
    evolution = {"monthly": monthly, "journal": load_journal()}

    return {
        "headline": {
            "n_scored": n,
            "hit_rate": round(100 * wins / n) if n else 0,
            "avg_return": round(avg, 2),
            "excess_avg": round(sum(excess) / len(excess), 2) if excess else None,
            "open": sum(1 for r in led if r.get("status") == "open"),
            "readiness_score": rd["score"], "stage": rd["stage"],
            "verdict": rd["verdict"],
        },
        "readiness": rd,
        "parameters": parameters,
        "learnings": learnings,
        "strategy": strategy,
        "evolution": evolution,
    }


def _monthly(scored: list[dict]) -> list[dict]:
    # Grouped by ENTRY month (the regime the calls were made in), so the record
    # spreads across the period it actually traded rather than the recent batch
    # in which everything happened to be scored.
    by: dict[str, list[float]] = {}
    for r in scored:
        m = str(r.get("date") or r.get("evaluated_on") or "")[:7]
        if m:
            by.setdefault(m, []).append(float(r["return_pct"]))
    out = []
    for m in sorted(by):
        rets = by[m]
        wins = sum(1 for x in rets if x > 0)
        out.append({"month": m, "n": len(rets),
                    "hit": round(100 * wins / len(rets)),
                    "avg": round(sum(rets) / len(rets), 1)})
    return out


# -- strategy journal (how the configuration drifts over time) --------------
def load_journal(path: Path | None = None) -> list[dict]:
    p = path or JOURNAL_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def append_snapshot(report: dict | None = None, path: Path | None = None) -> dict:
    """Record a dated snapshot of the desk's state so the Learning tab can show
    how the strategy and its readiness have moved. Idempotent per day (the latest
    snapshot for a date replaces an earlier one). Called by the curator."""
    p = path or JOURNAL_PATH
    rep = report or build_report()
    h, rd = rep["headline"], rep["readiness"]
    snap = {
        "date": datetime.date.today().isoformat(),
        "n_scored": h["n_scored"], "hit_rate": h["hit_rate"],
        "avg_return": h["avg_return"], "readiness_score": h["readiness_score"],
        "stage": h["stage"], "psr": rd["psr"], "brier": rd["brier"],
        "preset": rep["strategy"]["preset"],
        "throttle_active": rep["strategy"]["throttle_active"],
        "active_lessons": rep["strategy"]["active_lesson_count"],
    }
    journal = [s for s in load_journal(p) if s.get("date") != snap["date"]]
    journal.append(snap)
    journal.sort(key=lambda s: s["date"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    return snap


# -- text rendering (desktop GUI / console) ----------------------------------
def _gate_mark(status: str) -> str:
    return {"pass": "✓", "progress": "◐", "fail": "✗"}.get(status, "·")


def render_text(report: dict) -> str:
    h = report["headline"]
    rd = report["readiness"]
    L = []
    L.append("=" * 66)
    L.append(f"  DEPLOYMENT READINESS: {rd['score']}/100   →   {rd['verdict']}")
    L.append(f"  recommended stage: {rd['stage'].upper()}")
    L.append("=" * 66)
    L.append(rd["human_gate"])
    L.append("")
    L.append("READINESS GATES")
    L.append("-" * 66)
    for g in rd["gates"]:
        L.append(f"  {_gate_mark(g['status'])} {g['name']:<22} {g['detail']}")
        L.append(f"      ↳ {g['why']}")
    L.append("")
    L.append(f"TRACK RECORD: {h['n_scored']} scored calls · hit {h['hit_rate']}% · "
             f"avg {h['avg_return']:+.1f}%"
             + (f" · vs market {h['excess_avg']:+.1f}%" if h['excess_avg'] is not None else "")
             + f" · {h['open']} open")
    L.append(f"  PSR {rd['psr'] * 100:.0f}% (need {int(readiness.PSR_TARGET*100)}%)  ·  "
             f"per-trade Sharpe {rd['per_trade_sharpe']:.2f}  ·  "
             f"more trades to significance: "
             f"{'reached' if rd['min_track_record'] is None and rd['psr'] >= readiness.PSR_TARGET else (str(rd['min_track_record']) if rd['min_track_record'] else 'n/a (edge not positive)')}")
    if rd["brier"] is not None:
        L.append(f"  calibration Brier {rd['brier']:.3f}  ·  max drawdown "
                 f"{rd['max_drawdown'] * 100:.0f}%")
    L.append("")
    L.append("WHAT THE DESK IS ADJUSTING (and which agents it moves)")
    L.append("-" * 66)
    for p in report["parameters"]:
        L.append(f"  • {p['param']}: {p['value']}")
        L.append(f"      affects → {p['affects']}")
        L.append(f"      evidence → {p['confidence']}")
    L.append("")
    L.append("CURRENT STRATEGY")
    L.append("-" * 66)
    L.append(f"  {report['strategy']['summary']}")
    if report["strategy"]["top_lessons"]:
        L.append("  active lessons in play:")
        for t in report["strategy"]["top_lessons"]:
            L.append(f"    - {t}")
    L.append("")
    L.append("LENS PERFORMANCE (does each discovery lens pay?)")
    L.append("-" * 66)
    for key, label in (("hidden_gem", "hidden-gem"), ("core", "core"),
                       ("moat_bullish", "moat-bullish")):
        c = report["learnings"]["cohorts"].get(key, {})
        if c.get("n"):
            L.append(f"  {label:<14} n={c['n']:<3} hit {c['win_rate_pct']:.0f}%  "
                     f"avg {c['avg_return_pct']:+.1f}%")
    L.append("")
    L.append("STRATEGY EVOLUTION")
    L.append("-" * 66)
    monthly = report["evolution"]["monthly"]
    if monthly:
        L.append("  by month (scored calls):")
        for m in monthly[-12:]:
            L.append(f"    {m['month']}  n={m['n']:<3} hit {m['hit']}%  avg {m['avg']:+.1f}%")
    journal = report["evolution"]["journal"]
    if len(journal) >= 2:
        L.append("  readiness over time:")
        for s in journal[-8:]:
            L.append(f"    {s['date']}  score {s['readiness_score']:>3}/100  "
                     f"({s['stage']})  n={s['n_scored']}  preset={s.get('preset', '?')}")
    elif not monthly:
        L.append("  (history accumulates as recommendations mature and the curator runs)")
    L.append("")
    L.append(f"LESSONS: {len(report['learnings']['patterns'])} pattern + "
             f"{report['strategy']['active_lesson_count']} active total · "
             f"{report['learnings']['pending']} pending more evidence")
    return "\n".join(L)
