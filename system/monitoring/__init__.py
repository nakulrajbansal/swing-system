"""Fast loop: health monitoring + kill switch. May only REDUCE risk (master §15)."""

from system.monitoring.kill_switch import KillSwitch
from system.monitoring.scorecard import HealthScorecard

__all__ = ["HealthScorecard", "KillSwitch"]
