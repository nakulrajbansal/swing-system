"""Governance: holdout-once invariant and the human promotion gate."""

import pytest

from system.governance.holdout import HoldoutVault
from system.governance.promotion import PromotionPipeline, Stage


def test_holdout_consulted_at_most_once():
    vault = HoldoutVault("2024-01-01", "2024-12-31")
    assert vault.consult("edge01", lambda: 42) == 42
    with pytest.raises(RuntimeError, match="already consulted"):
        vault.consult("edge01", lambda: 99)
    assert len(vault.audit_log) == 1


def test_promotion_requires_human_gate_for_champion():
    p = PromotionPipeline()
    p.submit("edge01")
    for _ in range(6):                       # advance through gates with passes
        p.advance("edge01", gate_passed=True)
    # Stuck just below champion without human approval (winning != auto-promote).
    assert p.candidates["edge01"].stage == Stage.CHALLENGER
    p.advance("edge01", gate_passed=True, human_approval=True)
    assert p.candidates["edge01"].stage == Stage.CHAMPION


def test_degrading_champion_auto_demotes():
    p = PromotionPipeline()
    p.submit("edge01")
    p.demote("edge01", "drawdown breach")    # automatic, no human needed
    assert p.candidates["edge01"].stage == Stage.ARCHIVED
