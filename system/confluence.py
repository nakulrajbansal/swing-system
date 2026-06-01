"""The confluence engine (deterministic, master §5).

Turns specialist reads into the day's ranked top trades:
  1. each specialist emits a SpecialistRead;
  2. raw_score -> cross-sectional percentile per edge (so no edge dominates by scale);
  3. collapse within family (strongest, don't sum correlated signals);
  4. combine across families with independence-aware weights;
  5. confluence rule: high-confidence requires >= 2 independent families agreeing
     (or one exceptionally strong family);
  6. rank high-confidence names; top-K proceed to deliberation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from system.config import SystemConfig
from system.schemas import SpecialistRead

DEFAULT_FAMILY_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.8, "E": 0.8}
AGREE_PERCENTILE = 0.60


@dataclass
class Candidate:
    symbol: str
    combined_score: float
    n_families: int
    strong_single: bool
    families: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    min_confidence: float = 1.0


def _percentiles(score_by_symbol: dict[str, float]) -> dict[str, float]:
    syms = list(score_by_symbol)
    if len(syms) == 1:
        return {syms[0]: 1.0}
    vals = np.array([score_by_symbol[s] for s in syms], dtype=float)
    order = vals.argsort().argsort()                 # 0..n-1 ranks
    pct = order / (len(syms) - 1)
    return {s: float(p) for s, p in zip(syms, pct)}


def run_confluence(
    reads_by_symbol: dict[str, list[SpecialistRead]],
    config: SystemConfig,
    weights: dict[str, float] | None = None,
) -> list[Candidate]:
    weights = weights or DEFAULT_FAMILY_WEIGHTS

    # Step 2: per-edge cross-sectional percentile across triggered names.
    edge_scores: dict[str, dict[str, float]] = {}
    for sym, reads in reads_by_symbol.items():
        for r in reads:
            edge_scores.setdefault(r.edge_id, {})[sym] = r.raw_score
    edge_pct = {eid: _percentiles(sm) for eid, sm in edge_scores.items()}

    candidates = []
    for sym, reads in reads_by_symbol.items():
        # Step 3: collapse within family -> strongest percentile per family.
        fam_pct: dict[str, float] = {}
        fam_edges: dict[str, str] = {}
        evidence, confidences = [], []
        for r in reads:
            if r.direction != "long":
                continue
            p = edge_pct[r.edge_id][sym]
            if p > fam_pct.get(r.family, -1.0):
                fam_pct[r.family] = p
                fam_edges[r.family] = r.edge_id
            evidence.extend(r.evidence_refs)
            confidences.append(r.confidence)

        if not fam_pct:
            continue

        # Step 4: combine across families with independence-aware weights.
        num = sum(weights.get(f, 1.0) * p for f, p in fam_pct.items())
        den = sum(weights.get(f, 1.0) for f in fam_pct)
        combined = num / den if den else 0.0

        # Step 5: confluence rule.
        agreeing = [f for f, p in fam_pct.items() if p >= AGREE_PERCENTILE]
        strong_single = any(p >= config.confluence_strong_single for p in fam_pct.values())
        high_conf = len(agreeing) >= config.confluence_min_families or strong_single
        if not high_conf:
            continue

        candidates.append(Candidate(
            symbol=sym, combined_score=combined, n_families=len(agreeing),
            strong_single=strong_single, families=sorted(agreeing),
            edge_ids=[fam_edges[f] for f in sorted(agreeing)] or list(fam_edges.values()),
            evidence_refs=evidence,
            min_confidence=min(confidences) if confidences else 0.0))

    # Step 6: rank, top-K.
    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates[: config.top_k]
