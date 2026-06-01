"""Statistics for the event study (harness spec §6/§7, master §12).

Quintile bucketing, Newey-West t-stats for overlapping returns, and the
Deflated Sharpe Ratio for multiple-testing discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sps


def quintile_buckets(scores: pd.Series, q: int = 5) -> pd.Series:
    """Rank scores into 1..q buckets (q = strongest). Ties broken by rank."""
    if len(scores) < q:
        # Too few names to bucket cleanly; rank into as many bins as possible.
        ranks = scores.rank(method="first")
        return ((ranks - 1) // max(1, len(scores) // q + 1) + 1).astype(int)
    ranks = scores.rank(method="first")
    return (np.ceil(ranks / len(scores) * q)).astype(int).clip(1, q)


def newey_west_tstat(returns: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """Mean and Newey-West t-stat of a return series (overlapping-sample safe).

    lag defaults to the window length implied by sample size; pass the holding
    window for overlapping forward returns.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return (float(np.mean(r)) if n else 0.0, 0.0)
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mean = r.mean()
    resid = r - mean
    gamma0 = resid @ resid / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)             # Bartlett kernel
        cov = resid[k:] @ resid[:-k] / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return mean, mean / se if se > 0 else 0.0


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)


def deflated_sharpe_ratio(
    observed_sr: float,
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> float:
    """Probability the true Sharpe > 0 after deflating for `n_trials` (Bailey/LdP).

    Accounts for multiple testing, sample length, and the higher moments of the
    return distribution. Returns a probability in [0, 1].
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4 or n_trials < 1:
        return 0.0
    sr = observed_sr / np.sqrt(periods_per_year)          # per-period SR
    skew = float(_sps.skew(r))
    kurt = float(_sps.kurtosis(r, fisher=False))

    # Expected max Sharpe across n_trials independent noise strategies.
    emc = 0.5772156649
    e_max = np.sqrt(2 * np.log(n_trials)) - (
        (np.log(n_trials) + 2 * np.log(np.log(max(n_trials, 2)) + 1e-12)) /
        (2 * np.sqrt(2 * np.log(max(n_trials, 2))))
    ) if n_trials > 1 else 0.0
    sr0 = e_max * (1.0 / np.sqrt(n))  # threshold SR from variance of estimator (approx)
    _ = emc  # documented; threshold approximated above

    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4.0 * sr**2)
    denom = denom if denom > 1e-9 else 1e-9
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(_sps.norm.cdf(z))
