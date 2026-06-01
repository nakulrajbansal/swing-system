"""Health scorecard (master §15/§18).

Calibration (Brier), drawdown, expectancy, and live-vs-backtest divergence. The
fast loop consumes this and may only de-risk. Distinguishing skill from luck
takes months to years at swing frequency; early numbers are mostly noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HealthScorecard:
    closed_pnls: list[float] = field(default_factory=list)
    predicted_probs: list[float] = field(default_factory=list)   # conviction at entry
    outcomes: list[int] = field(default_factory=list)            # 1 = win
    equity_curve: list[float] = field(default_factory=list)

    def record_trade(self, pnl: float, predicted_prob: float | None = None):
        self.closed_pnls.append(pnl)
        if predicted_prob is not None:
            self.predicted_probs.append(predicted_prob)
            self.outcomes.append(1 if pnl > 0 else 0)

    def mark_equity(self, equity: float):
        self.equity_curve.append(equity)

    def brier_score(self) -> float:
        if not self.predicted_probs:
            return float("nan")
        p = np.array(self.predicted_probs)
        y = np.array(self.outcomes)
        return float(np.mean((p - y) ** 2))

    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        c = np.array(self.equity_curve)
        peak = np.maximum.accumulate(c)
        return float(((c - peak) / peak).min())

    def expectancy(self) -> float:
        return float(np.mean(self.closed_pnls)) if self.closed_pnls else 0.0

    def win_rate(self) -> float:
        if not self.closed_pnls:
            return 0.0
        return float(np.mean([1 for p in self.closed_pnls if p > 0]) if self.closed_pnls else 0)

    def summary(self) -> dict:
        wins = [p for p in self.closed_pnls if p > 0]
        return {
            "trades": len(self.closed_pnls),
            "win_rate": round(len(wins) / len(self.closed_pnls), 3) if self.closed_pnls else 0.0,
            "expectancy": round(self.expectancy(), 2),
            "max_drawdown": round(self.max_drawdown(), 4),
            "brier": self.brier_score(),
        }
