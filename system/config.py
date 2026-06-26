"""System configuration: hard invariants, risk limits, model tiering.

The risk limits and the asymmetric-autonomy rule are INVARIANTS (master §3):
their structure cannot be modified by any agent or optimizer at runtime.
Changing them is a deliberate, version-controlled human act. We enforce that by
freezing the dataclass; the Risk Governor reads these and nothing may raise a
limit programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    """Hard ceilings (master §10). Defaults are tunable by humans, invariant in code."""
    risk_per_trade: float = 0.01        # <= 1% equity at risk per trade
    max_open_positions: int = 8
    max_portfolio_heat: float = 0.06    # <= 6% total open risk
    max_single_name: float = 0.15       # <= 15% equity in one name
    max_gross_exposure: float = 1.0      # <= 1.0x equity gross notional (no leverage)
    max_sector_exposure: float = 0.30   # <= 30% in one sector
    max_cluster_heat: float = 0.03      # <= 3% heat per correlation cluster
    max_new_entries_per_cycle: int = 3
    daily_loss_halt: float = -0.03      # -3% -> no new entries
    weekly_loss_halt: float = -0.07     # -7% -> halt + human review


@dataclass(frozen=True)
class SizingParams:
    atr_len: int = 14
    atr_stop_mult: float = 2.0
    reward_risk: float = 2.0
    min_viable_shares: int = 1
    entry_band_bps: float = 50.0        # marketable-limit band around the reference


@dataclass(frozen=True)
class ModelTiering:
    """Pinned model IDs (master §6). Low temperature, structured outputs."""
    framing: str = "claude-haiku-4-5"        # high-volume specialist framing
    synthesis: str = "claude-sonnet-4-6"     # hypothesis / synthesis
    adversarial: str = "claude-opus-4-8"     # Skeptic + Portfolio Manager
    temperature: float = 0.0
    max_tokens: int = 1500


@dataclass(frozen=True)
class CalibrationPolicy:
    """Cold-start: flat sizing until enough closed trades (master §14)."""
    cold_start_trades: int = 50
    flat_multiplier: float = 1.0


@dataclass(frozen=True)
class KillCriteria:
    """Pre-committed kill thresholds (master §17)."""
    strategy_drawdown: float = -0.15
    system_drawdown: float = -0.20
    max_slippage_ratio: float = 2.0     # live slippage >> model
    min_calibration_brier: float = 0.30


@dataclass(frozen=True)
class SystemConfig:
    limits: RiskLimits = field(default_factory=RiskLimits)
    sizing: SizingParams = field(default_factory=SizingParams)
    models: ModelTiering = field(default_factory=ModelTiering)
    calibration: CalibrationPolicy = field(default_factory=CalibrationPolicy)
    kill: KillCriteria = field(default_factory=KillCriteria)

    # Confluence (master §5)
    confluence_min_families: int = 2
    confluence_strong_single: float = 0.95   # one exceptionally strong family
    top_k: int = 5

    # Contested-debate escalation: a second skeptic round fires when the
    # critique is severe or the analyst panel disagrees sharply.
    debate_severity_threshold: float = 0.60
    debate_analyst_spread: float = 0.35

    # Execution
    paper_only: bool = True                  # live trading requires explicit opt-in
    correlation_lookback: int = 60
    correlation_threshold: float = 0.6


DEFAULT_CONFIG = SystemConfig()
