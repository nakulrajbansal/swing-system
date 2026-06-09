"""Core JSON schemas / contracts (master Appendix B, §7).

Agents exchange only schema-validated JSON wrapped in an envelope. These
dataclasses are the single source of truth for those payloads; each validates
its own invariants so a malformed agent output is caught deterministically (and
the orchestrator can then fail that candidate to PASS).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Direction = Literal["long", "short", "avoid"]


def _clamp01(x: float, name: str) -> float:
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


# --------------------------------------------------------------------------
# Specialist read envelope (master §5/§6): the cross-agent contract.
# --------------------------------------------------------------------------
@dataclass
class SpecialistRead:
    edge_id: str
    family: str                       # "A".."E"
    raw_score: float
    confidence: float
    direction: Direction
    evidence_refs: list[str] = field(default_factory=list)

    def validate(self) -> "SpecialistRead":
        _clamp01(self.confidence, "confidence")
        if self.family not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"bad family {self.family!r}")
        if self.direction not in {"long", "short", "avoid"}:
            raise ValueError(f"bad direction {self.direction!r}")
        return self


# --------------------------------------------------------------------------
# Specialist reads (Appendix B)
# --------------------------------------------------------------------------
@dataclass
class TechRead:
    regime: str                       # "uptrend"|"downtrend"|"range"|"unclear"
    trend_quality: float
    volatility: str                   # "low"|"normal"|"high"
    relative_strength: float
    support: float | None
    resistance: float | None
    tradability: str                  # "good"|"fair"|"poor"

    def validate(self) -> "TechRead":
        _clamp01(self.trend_quality, "trend_quality")
        return self


@dataclass
class AnalystRead:
    """A written, swing-trade-oriented read from a domain analyst (technical or
    fundamental). Shown verbatim in the deliberation and fed to the trio."""
    domain: str                       # "technical" | "fundamental"
    stance: str                       # "bullish" | "neutral" | "bearish"
    score: float                      # 0..1 strength of the LONG case
    assessment: str
    positives: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    def validate(self) -> "AnalystRead":
        _clamp01(self.score, "score")
        if self.stance not in {"bullish", "neutral", "bearish"}:
            self.stance = "neutral"
        return self


@dataclass
class CatalystRead:
    items: list[dict]                 # [{type, bias, materiality, timing, status, record_id}]
    narrative: str
    priced_in: float                  # 0=not priced, 1=fully priced
    confidence: float

    def validate(self) -> "CatalystRead":
        _clamp01(self.priced_in, "priced_in")
        _clamp01(self.confidence, "confidence")
        return self


# --------------------------------------------------------------------------
# Core trio (Appendix B)
# --------------------------------------------------------------------------
@dataclass
class TradeHypothesis:
    symbol: str
    decision: Literal["propose", "decline"]
    mechanism: str
    evidence_refs: list[str]
    expected_hold_days: int
    invalidation: str
    raw_conviction: float             # 0.7 ~ 70% chance
    direction: Direction = "long"

    def validate(self) -> "TradeHypothesis":
        _clamp01(self.raw_conviction, "raw_conviction")
        if self.decision == "propose":
            if not self.mechanism or not self.invalidation:
                raise ValueError("a proposed thesis needs mechanism + invalidation")
            if not 2 <= self.expected_hold_days <= 20:
                raise ValueError("swing hold must be 2..20 days")
        return self


@dataclass
class Objection:
    kind: str
    detail: str
    severity: float                   # 0..1

    def validate(self) -> "Objection":
        _clamp01(self.severity, "severity")
        return self


@dataclass
class Critique:
    objections: list[Objection]
    strongest: str
    verdict: Literal["kill", "caution", "clean"]

    def validate(self) -> "Critique":
        for o in self.objections:
            o.validate()
        if self.verdict not in {"kill", "caution", "clean"}:
            raise ValueError(f"bad verdict {self.verdict!r}")
        return self

    def max_severity(self) -> float:
        return max((o.severity for o in self.objections), default=0.0)


@dataclass
class RiskDecision:
    symbol: str
    action: Literal["enter", "adjust", "pass"]
    final_conviction: float
    entry: float | None = None        # request only; Risk Governor recomputes
    stop: float | None = None
    target: float | None = None
    constraints_ack: bool = True
    decisive_factor: str = ""

    def validate(self) -> "RiskDecision":
        _clamp01(self.final_conviction, "final_conviction")
        if self.action not in {"enter", "adjust", "pass"}:
            raise ValueError(f"bad action {self.action!r}")
        if self.action in {"enter", "adjust"}:
            if self.entry is None or self.stop is None:
                raise ValueError("enter/adjust requires entry and stop requests")
            if self.stop >= self.entry:
                raise ValueError("long stop must be below entry")
        return self


@dataclass
class Lesson:
    setup_type: str
    lesson: str
    thesis_correct: bool
    execution_quality: str
    # Optional provenance (advisory; tagged with when the trade closed so recall
    # can be filtered point-in-time and never leak a future outcome).
    symbol: str = ""
    as_of: str = ""               # ISO date the originating trade closed
    pnl_pct: float = 0.0
    conviction: float = 0.0

    def validate(self) -> "Lesson":
        return self


# --------------------------------------------------------------------------
# Deliberation envelope (master §7)
# --------------------------------------------------------------------------
@dataclass
class Envelope:
    candidate_id: str
    agent: str
    model_id: str
    prompt_version: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    inputs_hash: str = ""

    @staticmethod
    def hash_inputs(inputs: Any) -> str:
        blob = json.dumps(inputs, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)


def to_json(obj) -> str:
    return json.dumps(asdict(obj), default=str, sort_keys=True)
