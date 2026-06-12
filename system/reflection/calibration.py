"""Conviction calibration: turn the desk's stated conviction into an
evidence-based probability.

The ledger records what the desk CLAIMED (conviction) and what HAPPENED
(realized return). This module maps the two: scored calls are bucketed into
conviction bands over a TRAILING window (regimes drift), and a new
recommendation's conviction is converted to a win probability by shrinking the
band's realized win rate toward the stated conviction with a pseudo-count —
with no evidence you get the desk's own claim back; with a seasoned record the
realized rate dominates. Honest by construction: the label always says how
much evidence is behind the number.
"""

from __future__ import annotations

BANDS = ((0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01))
TRAILING = 60        # most recent scored calls considered (regimes drift)
PSEUDO_N = 10        # shrinkage strength toward the stated conviction
MIN_CALIBRATED = 8   # below this many scored calls, the label says "uncalibrated"


def calibration_table(rows: list[dict], trailing: int = TRAILING) -> dict:
    """Per-conviction-band realized results over the trailing window."""
    scored = [r for r in rows
              if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))
              and isinstance(r.get("conviction"), (int, float))]
    scored.sort(key=lambda r: str(r.get("evaluated_on") or ""))
    scored = scored[-trailing:]
    bands = []
    for lo, hi in BANDS:
        rs = [r for r in scored if lo <= float(r["conviction"]) < hi]
        wins = sum(1 for r in rs if float(r["return_pct"]) > 0)
        bands.append({"lo": lo, "hi": hi, "n": len(rs), "wins": wins,
                      "win_rate_pct": round(100.0 * wins / len(rs), 0) if rs else None})
    return {"bands": bands, "n_total": len(scored)}


def calibrated_probability(conviction, table: dict, k: int = PSEUDO_N):
    """Empirical-Bayes win probability for a stated conviction.

    p = (band wins + conviction * k) / (band n + k): n=0 returns the stated
    conviction unchanged; a seasoned band pulls toward its realized rate."""
    try:
        c = min(max(float(conviction), 0.0), 1.0)
    except (TypeError, ValueError):
        return None
    band = next((b for b in table.get("bands", [])
                 if b["lo"] <= c < b["hi"]), None)
    if band is None:
        return round(c, 2)
    return round((band["wins"] + c * k) / (band["n"] + k), 2)


def describe(p, table: dict) -> str:
    """Human label for a calibrated probability — always says the evidence."""
    n = table.get("n_total", 0)
    if p is None:
        return "n/a"
    if n < MIN_CALIBRATED:
        return f"~{p * 100:.0f}% (uncalibrated - only {n} scored call(s) so far)"
    return f"{p * 100:.0f}% (calibrated on {n} scored calls)"
