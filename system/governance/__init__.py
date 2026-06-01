"""Slow loop: holdout vault + champion/challenger promotion (master §16, may only propose)."""

from system.governance.holdout import HoldoutVault
from system.governance.promotion import PromotionPipeline, Stage

__all__ = ["HoldoutVault", "PromotionPipeline", "Stage"]
