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


THROTTLE_MIN_N = 8        # need this many scored calls before judging a streak


def desk_throttle(rows: list[dict], recent: int = 12) -> dict:
    """The 'shut down if no edge' control, keyed off the desk's own realized
    recommendations. When the recent record is poor OR convictions are running
    hot vs. realized odds, raise the entry bar and cut gross exposure. STRICTLY
    risk-reducing and DORMANT until enough trades are scored, so it can never
    misfire on thin data or make the desk more aggressive."""
    scored = [r for r in rows
              if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))]
    scored.sort(key=lambda r: str(r.get("evaluated_on") or ""))
    n = len(scored)
    off = {"active": False, "conviction_bump": 0.0, "gross_scale": 1.0,
           "n": n, "reason": ""}
    if n < THROTTLE_MIN_N:
        return off
    window = scored[-recent:]
    wins = sum(1 for r in window if r["return_pct"] > 0)
    hit = 100.0 * wins / len(window)
    convs = [r["conviction"] for r in window
             if isinstance(r.get("conviction"), (int, float))]
    avg_conv = (sum(convs) / len(convs) * 100) if convs else None
    cal_gap = (avg_conv - hit) if avg_conv is not None else 0.0
    reasons = []
    if hit < 40:
        reasons.append(f"recent hit rate {hit:.0f}% over {len(window)} calls")
    if cal_gap >= 25:
        reasons.append(f"convictions running ~{cal_gap:.0f}pp hot vs realized")
    if not reasons:
        return off
    # Graduated, capped de-risking — bigger when the evidence is worse.
    bump = 0.05 if hit < 40 else 0.0
    bump += 0.03 if cal_gap >= 25 else 0.0
    scale = 0.6 if hit < 30 else 0.75
    return {"active": True, "conviction_bump": round(min(bump, 0.10), 2),
            "gross_scale": scale, "n": n,
            "reason": "; ".join(reasons)}


def describe(p, table: dict) -> str:
    """Human label for a calibrated probability — always says the evidence."""
    n = table.get("n_total", 0)
    if p is None:
        return "n/a"
    if n < MIN_CALIBRATED:
        return f"~{p * 100:.0f}% (uncalibrated - only {n} scored call(s) so far)"
    return f"{p * 100:.0f}% (calibrated on {n} scored calls)"
