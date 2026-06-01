"""Correlation-cluster heat (master §10, G14/G26).

A 60-day rolling-correlation cluster groups names whose pairwise return
correlation exceeds a threshold. Cluster heat (summed open risk within a
cluster) is capped in code, not left to PM judgment — this prevents a
correlated pile-on that defeats per-name limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_clusters(
    returns: pd.DataFrame,
    threshold: float = 0.6,
) -> list[set[str]]:
    """Single-linkage clusters from a returns matrix (columns = symbols).

    Two names join the same cluster if their pairwise correlation >= threshold;
    clustering is transitive (connected components of the threshold graph).
    """
    cols = list(returns.columns)
    if len(cols) <= 1:
        return [{c} for c in cols]
    corr = returns.corr().fillna(0.0).to_numpy()

    parent = {c: c for c in cols}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr[i, j] >= threshold:
                union(cols[i], cols[j])

    groups: dict[str, set[str]] = {}
    for c in cols:
        groups.setdefault(find(c), set()).add(c)
    return list(groups.values())


def cluster_of(symbol: str, clusters: list[set[str]]) -> set[str]:
    for cl in clusters:
        if symbol in cl:
            return cl
    return {symbol}


def returns_matrix(price_panels: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """Build an aligned daily-returns matrix from per-symbol OHLCV panels."""
    series = {}
    for sym, df in price_panels.items():
        if df is None or df.empty:
            continue
        s = df.set_index("date")["close"] if "date" in df.columns else df["close"]
        series[sym] = s.pct_change()
    if not series:
        return pd.DataFrame()
    mat = pd.DataFrame(series).dropna(how="all")
    return mat.iloc[-lookback:]
