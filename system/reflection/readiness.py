"""Deployment-readiness assessment: is the desk's edge real enough to risk cash?

This is the rigorous, evidence-gated answer to "when can I deploy this with a
real account?" — grounded in the standard quantitative-finance toolkit rather
than a feeling about a few good trades:

  * Probabilistic Sharpe Ratio (PSR) and Minimum Track Record Length (MinTRL)
    — Bailey & Lopez de Prado (2012). PSR is the probability the TRUE per-trade
    Sharpe exceeds zero given the observed Sharpe, skew and kurtosis; MinTRL is
    how many scored trades you need before that probability clears a confidence
    bar. This is what turns "we have a few winners" into "the edge is
    statistically distinguishable from luck" — and tells you how many more
    trades to wait when it is not.
  * Brier score + reliability (calibration) — a forecaster is only trustworthy
    if its stated conviction matches realized odds; an accurate-but-overconfident
    book sizes itself into ruin.
  * Max drawdown, time/regime diversity, shadow-track duration, and edge-decay
    — the remaining champion/challenger gates (master §16). At swing frequency,
    distinguishing skill from luck takes months; these gates encode that the
    early numbers are mostly noise and force a conservative ramp.

Pure and deterministic (no IO); the app layer feeds it ledger rows and renders
the result. Crucially this module only ever *recommends* a stage — enabling real
money stays a human act (asymmetric-autonomy invariant): automation may move the
desk toward LESS risk on its own, never toward more.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORM = NormalDist()

# Confidence bar for the edge-significance gate (one-sided): the desk must be
# 95% sure the true per-trade Sharpe is positive before tiny-capital is advised.
PSR_TARGET = 0.95
_Z_TARGET = _NORM.inv_cdf(PSR_TARGET)          # ~1.645

# Gate thresholds — conservative on purpose (swing frequency = thin data).
MIN_SCORED_OK = 30          # scored calls for a trustworthy track record
MIN_SCORED_PROGRESS = 15
MIN_MONTHS_OK = 6           # distinct calendar months the record must span
MIN_SECTORS_OK = 4          # distinct sectors (no single-theme fluke)
MIN_SHADOW_DAYS_OK = 90     # paper/shadow track length (master: 2-3 months min)
MIN_SHADOW_DAYS_PROGRESS = 45
MAX_DD_OK = -0.15           # equity-curve max drawdown tolerated for "ready"
MAX_DD_FAIL = -0.25
BRIER_OK = 0.22             # below the 0.25 "always 50%" baseline = informative
BRIER_PROGRESS = 0.25
MIN_CALIBRATED = 8          # below this many calls, calibration is not judged


# -- core statistics ---------------------------------------------------------
def _moments(rets: list[float]) -> dict:
    n = len(rets)
    if n < 2:
        return {"n": n, "mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 3.0}
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return {"n": n, "mean": mean, "std": 0.0, "skew": 0.0, "kurt": 3.0}
    m3 = sum((r - mean) ** 3 for r in rets) / n
    m4 = sum((r - mean) ** 4 for r in rets) / n
    sd_pop = math.sqrt(sum((r - mean) ** 2 for r in rets) / n)
    skew = m3 / (sd_pop ** 3) if sd_pop else 0.0
    kurt = m4 / (sd_pop ** 4) if sd_pop else 3.0      # non-excess (normal = 3)
    return {"n": n, "mean": mean, "std": std, "skew": skew, "kurt": kurt}


def sharpe(rets: list[float]) -> float:
    """Per-trade Sharpe (not annualized): mean / std of the trade returns."""
    m = _moments(rets)
    return (m["mean"] / m["std"]) if m["std"] else 0.0


def probabilistic_sharpe(rets: list[float], sr_benchmark: float = 0.0) -> float:
    """PSR: P(true Sharpe > `sr_benchmark`) given the observed Sharpe, skew and
    kurtosis (Bailey & Lopez de Prado). Returns a probability in [0, 1]."""
    m = _moments(rets)
    n, sr = m["n"], (m["mean"] / m["std"] if m["std"] else 0.0)
    if n < 2 or m["std"] == 0:
        return 0.0
    denom = 1.0 - m["skew"] * sr + ((m["kurt"] - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        denom = 1e-9
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_NORM.cdf(z))


def min_track_record_length(rets: list[float], target: float = PSR_TARGET,
                            sr_benchmark: float = 0.0) -> float:
    """MinTRL: the number of scored trades required for the per-trade Sharpe to
    be significantly above `sr_benchmark` at `target` confidence. Infinite when
    the observed edge is non-positive (no track length proves a zero edge)."""
    m = _moments(rets)
    sr = (m["mean"] / m["std"]) if m["std"] else 0.0
    if sr <= sr_benchmark or m["std"] == 0:
        return math.inf
    z = _NORM.inv_cdf(target)
    denom = 1.0 - m["skew"] * sr + ((m["kurt"] - 1.0) / 4.0) * sr * sr
    return 1.0 + denom * (z / (sr - sr_benchmark)) ** 2


def brier_score(probs: list[float], wins: list[int]) -> float | None:
    """Mean squared error of the stated win-probabilities (conviction) vs the
    0/1 outcomes. 0.25 is the 'always 50%' baseline; lower is more informative."""
    pairs = [(p, y) for p, y in zip(probs, wins)
             if isinstance(p, (int, float))]
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def reliability(bands: list[dict]) -> dict:
    """How well conviction bands rank-order realized win rates. `bands` come from
    the calibration table (low→high conviction). Returns monotonicity and ECE
    (expected calibration error: mean |band win-rate − band mid-conviction|)."""
    used = [b for b in bands if b.get("n") and b.get("win_rate_pct") is not None]
    if len(used) < 2:
        return {"monotonic": None, "ece": None, "n_bands": len(used)}
    rates = [b["win_rate_pct"] for b in used]
    monotonic = all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    tot = sum(b["n"] for b in used)
    ece = sum(b["n"] * abs(b["win_rate_pct"] / 100.0
                           - (b["lo"] + b["hi"]) / 2.0) for b in used) / tot
    return {"monotonic": monotonic, "ece": round(ece, 3), "n_bands": len(used)}


def max_drawdown(equity_curve: list[float]) -> float:
    """Largest peak-to-trough fraction of a compounded equity curve (<= 0)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak)
    return worst


# -- gate helpers ------------------------------------------------------------
def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _gate(name: str, status: str, detail: str, why: str) -> dict:
    """status: 'pass' | 'progress' | 'fail'."""
    return {"name": name, "status": status, "detail": detail, "why": why}


_SCORE = {"pass": 1.0, "progress": 0.5, "fail": 0.0}


def assess(scored: list[dict], equity_curve: list[float] | None = None,
           calibration_bands: list[dict] | None = None,
           shadow_days: int | None = None) -> dict:
    """Full readiness assessment from the desk's scored recommendations.

    `scored` rows need: return_pct, conviction, evaluated_on (ISO date),
    sector. `equity_curve` is the compounded scored-return curve;
    `calibration_bands` the conviction→win-rate table; `shadow_days` the span of
    the paper track. Returns gates, a 0-100 readiness score, and a recommended
    promotion stage — advisory only; turning on real money stays a human act."""
    rets = [float(r["return_pct"]) for r in scored
            if isinstance(r.get("return_pct"), (int, float))]
    n = len(rets)
    gates: list[dict] = []

    # G1 — EVIDENCE: enough scored trades to distinguish skill from luck.
    if n >= MIN_SCORED_OK:
        gates.append(_gate("Sample size", "pass",
                           f"{n} scored calls (>= {MIN_SCORED_OK})",
                           "Few trades cannot separate skill from luck."))
    elif n >= MIN_SCORED_PROGRESS:
        gates.append(_gate("Sample size", "progress",
                           f"{n} scored calls (target {MIN_SCORED_OK})",
                           "Few trades cannot separate skill from luck."))
    else:
        gates.append(_gate("Sample size", "fail",
                           f"only {n} scored calls (need >= {MIN_SCORED_PROGRESS})",
                           "Few trades cannot separate skill from luck."))

    # G2 — EDGE SIGNIFICANCE: PSR / MinTRL (the statistical core).
    psr = probabilistic_sharpe(rets) if n >= 2 else 0.0
    sr = sharpe(rets)
    mintrl = min_track_record_length(rets)
    mintrl_txt = ("∞ (edge not positive)" if mintrl == math.inf
                  else f"{mintrl:.0f}")
    if n >= MIN_SCORED_PROGRESS and psr >= PSR_TARGET:
        gates.append(_gate("Edge is real (PSR)", "pass",
                           f"PSR {psr * 100:.0f}% with {n} trades "
                           f"(need ~{mintrl_txt}); per-trade Sharpe {sr:.2f}",
                           "Probability the true edge beats zero, skew/kurtosis "
                           "adjusted."))
    elif psr >= 0.80 and sr > 0:
        gates.append(_gate("Edge is real (PSR)", "progress",
                           f"PSR {psr * 100:.0f}% (target {PSR_TARGET * 100:.0f}%); "
                           f"~{mintrl_txt} trades needed, have {n}",
                           "Probability the true edge beats zero, skew/kurtosis "
                           "adjusted."))
    else:
        gates.append(_gate("Edge is real (PSR)", "fail",
                           f"PSR {psr * 100:.0f}% (target {PSR_TARGET * 100:.0f}%); "
                           f"per-trade Sharpe {sr:.2f}",
                           "Probability the true edge beats zero, skew/kurtosis "
                           "adjusted."))

    # G3 — CALIBRATION: does stated conviction match realized odds?
    probs = [float(r["conviction"]) for r in scored
             if isinstance(r.get("conviction"), (int, float))]
    wins = [1 if float(r["return_pct"]) > 0 else 0 for r in scored
            if isinstance(r.get("conviction"), (int, float))
            and isinstance(r.get("return_pct"), (int, float))]
    brier = brier_score(probs, wins)
    rel = reliability(calibration_bands or [])
    if brier is None or n < MIN_CALIBRATED:
        gates.append(_gate("Calibration", "progress",
                           f"not yet judged (need {MIN_CALIBRATED} scored, have {n})",
                           "Conviction must track real odds before it can size risk."))
    elif brier <= BRIER_OK and rel.get("monotonic") is not False:
        gates.append(_gate("Calibration", "pass",
                           f"Brier {brier:.3f} (<= {BRIER_OK}); bands rank-ordered",
                           "Conviction must track real odds before it can size risk."))
    elif brier <= BRIER_PROGRESS:
        gates.append(_gate("Calibration", "progress",
                           f"Brier {brier:.3f} (target <= {BRIER_OK})"
                           + ("" if rel.get("monotonic") is not False
                              else "; conviction bands are INVERTED"),
                           "Conviction must track real odds before it can size risk."))
    else:
        gates.append(_gate("Calibration", "fail",
                           f"Brier {brier:.3f} (> {BRIER_PROGRESS} baseline)",
                           "Conviction must track real odds before it can size risk."))

    # G4 — DRAWDOWN: survivable on the scored-return equity curve.
    mdd = max_drawdown(equity_curve or [])
    if equity_curve and len(equity_curve) >= 3:
        if mdd >= MAX_DD_OK:
            gates.append(_gate("Drawdown", "pass", f"max drawdown {mdd * 100:.0f}%",
                               "A real-money account has to survive the worst run."))
        elif mdd >= MAX_DD_FAIL:
            gates.append(_gate("Drawdown", "progress", f"max drawdown {mdd * 100:.0f}%",
                               "A real-money account has to survive the worst run."))
        else:
            gates.append(_gate("Drawdown", "fail", f"max drawdown {mdd * 100:.0f}%",
                               "A real-money account has to survive the worst run."))
    else:
        gates.append(_gate("Drawdown", "progress", "not enough scored calls yet",
                           "A real-money account has to survive the worst run."))

    # G5 — DIVERSITY: spread across time (regimes) and sectors, not one theme.
    # Keyed off ENTRY date (when the trade was live and exposed to a regime),
    # not when it was scored — calls often mature in one recent batch.
    months = {str(r.get("date") or r.get("evaluated_on") or "")[:7] for r in scored
              if r.get("date") or r.get("evaluated_on")}
    months.discard("")
    sectors = {r.get("sector") for r in scored if r.get("sector") not in (None, "?")}
    nm, ns = len(months), len(sectors)
    md, sd = _plural(nm, "month"), _plural(ns, "sector")
    if nm >= MIN_MONTHS_OK and ns >= MIN_SECTORS_OK:
        gates.append(_gate("Diversity", "pass", f"{md}, {sd}",
                           "An edge seen in one month or one theme may be a fluke."))
    elif nm >= 3 and ns >= 2:
        gates.append(_gate("Diversity", "progress",
                           f"{md} (need {MIN_MONTHS_OK}), {sd} (need {MIN_SECTORS_OK})",
                           "An edge seen in one month or one theme may be a fluke."))
    else:
        gates.append(_gate("Diversity", "fail", f"{md}, {sd}",
                           "An edge seen in one month or one theme may be a fluke."))

    # G6 — SHADOW DURATION: a real paper track, not a fast week of luck.
    if shadow_days is not None:
        if shadow_days >= MIN_SHADOW_DAYS_OK:
            gates.append(_gate("Shadow track", "pass",
                               f"{shadow_days} days of paper track",
                               "Skill at swing frequency only shows over months."))
        elif shadow_days >= MIN_SHADOW_DAYS_PROGRESS:
            gates.append(_gate("Shadow track", "progress",
                               f"{shadow_days} days (target {MIN_SHADOW_DAYS_OK})",
                               "Skill at swing frequency only shows over months."))
        else:
            gates.append(_gate("Shadow track", "fail",
                               f"{shadow_days} days (need >= {MIN_SHADOW_DAYS_PROGRESS})",
                               "Skill at swing frequency only shows over months."))

    # G7 — EDGE STABILITY: the recent half is not decaying vs the older half.
    stability = _stability(scored)
    gates.append(stability)

    # -- roll up -------------------------------------------------------------
    score = round(100.0 * sum(_SCORE[g["status"]] for g in gates) / len(gates))
    critical = {"Sample size", "Edge is real (PSR)"}
    crit_fail = any(g["status"] == "fail" and g["name"] in critical for g in gates)
    any_fail = any(g["status"] == "fail" for g in gates)
    all_pass = all(g["status"] == "pass" for g in gates)

    if crit_fail or n < MIN_SCORED_PROGRESS:
        stage, verdict = "shadow", "NOT READY — keep building the paper record"
    elif all_pass:
        stage, verdict = "challenger", ("READY for a TINY real-capital challenger "
                                        "(smallest size, head-to-head with paper)")
    elif not any_fail:
        stage, verdict = "shadow", ("ALMOST — edge is significant; clear the "
                                    "remaining gates before risking cash")
    else:
        stage, verdict = "shadow", "NOT READY — keep building the paper record"

    return {
        "n_scored": n, "psr": round(psr, 3), "per_trade_sharpe": round(sr, 3),
        "min_track_record": (None if mintrl == math.inf else round(mintrl, 1)),
        "brier": (None if brier is None else round(brier, 3)),
        "reliability": rel, "max_drawdown": round(mdd, 4),
        "months": nm, "sectors": ns, "shadow_days": shadow_days,
        "gates": gates, "score": score, "stage": stage, "verdict": verdict,
        "human_gate": ("Enabling real-money trading is a human action — the desk "
                       "can recommend this stage but never flips it on itself."),
    }


def _stability(scored: list[dict]) -> dict:
    """Compare the recent half of the record to the older half; a materially
    worse recent half flags a decaying edge."""
    rows = sorted((r for r in scored
                   if isinstance(r.get("return_pct"), (int, float))),
                  key=lambda r: str(r.get("evaluated_on") or ""))
    if len(rows) < 2 * MIN_SCORED_PROGRESS:
        return _gate("Edge stability", "progress",
                     "not enough scored calls to judge decay yet",
                     "A fading edge should stop a deployment.")
    half = len(rows) // 2
    old = sum(float(r["return_pct"]) for r in rows[:half]) / half
    new_rows = rows[half:]
    new = sum(float(r["return_pct"]) for r in new_rows) / len(new_rows)
    if new >= 0 and new >= 0.6 * old:
        return _gate("Edge stability", "pass",
                     f"recent avg {new:+.1f}% vs earlier {old:+.1f}%",
                     "A fading edge should stop a deployment.")
    if new >= 0:
        return _gate("Edge stability", "progress",
                     f"recent avg {new:+.1f}% below earlier {old:+.1f}%",
                     "A fading edge should stop a deployment.")
    return _gate("Edge stability", "fail",
                 f"recent avg {new:+.1f}% has turned negative (earlier {old:+.1f}%)",
                 "A fading edge should stop a deployment.")
