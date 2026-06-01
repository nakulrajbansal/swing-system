"""Meta agents (master §6, §15-16).

Guardian: exit-only re-check of open positions (can never add risk).
Reflection: post-trade attribution + one durable lesson (advisory only).
Researcher: proposes falsifiable edge/parameter changes (no authority).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from system.agents.base import Agent
from system.agents.prompts import GUARDIAN, REFLECTION, RESEARCHER
from system.schemas import Lesson


@dataclass
class GuardianDecision:
    symbol: str
    action: str            # "exit" | "hold"  (never "add")
    reason: str = ""

    def validate(self) -> "GuardianDecision":
        if self.action not in {"exit", "hold"}:
            raise ValueError("Guardian may only exit or hold, never add")
        return self


@dataclass
class ResearchProposal:
    hypothesis: str
    mechanism: str
    pass_kill_test: str
    authority: str = "none"   # carries no authority; must pass the validation funnel

    def validate(self) -> "ResearchProposal":
        if self.authority != "none":
            raise ValueError("research proposals carry no authority")
        return self


class GuardianAgent(Agent):
    name = "guardian"
    schema_name = "GuardianDecision"

    def __init__(self, client, model):
        super().__init__(client, model, GUARDIAN)

    def deterministic(self, inputs: dict) -> GuardianDecision:
        symbol = inputs["symbol"]
        # Exit only on a thesis-breaking contemporaneous catalyst.
        if bool(inputs.get("thesis_broken", False)):
            return GuardianDecision(symbol, "exit", inputs.get("reason", "thesis broken"))
        return GuardianDecision(symbol, "hold")


class ReflectionAgent(Agent):
    name = "reflection"
    schema_name = "Lesson"

    def __init__(self, client, model):
        super().__init__(client, model, REFLECTION)

    def deterministic(self, inputs: dict) -> Lesson:
        trade = inputs["trade"]
        pnl = float(trade.get("pnl", 0.0))
        reason = trade.get("reason", "")
        thesis_correct = pnl > 0
        quality = "clean" if reason in {"target", "time"} else "stopped"
        return Lesson(
            setup_type=trade.get("setup_type", "confluence_swing"),
            lesson=("edge paid net of costs" if thesis_correct
                    else "edge did not pay; revisit trigger or sizing"),
            thesis_correct=thesis_correct, execution_quality=quality)


class ResearcherAgent(Agent):
    name = "researcher"
    schema_name = "ResearchProposal"

    def __init__(self, client, model):
        super().__init__(client, model, RESEARCHER)

    def deterministic(self, inputs: dict) -> ResearchProposal:
        return ResearchProposal(
            hypothesis=inputs.get("hypothesis", "a candidate edge"),
            mechanism=inputs.get("mechanism", "stated mechanism"),
            pass_kill_test="validate through the shared event-study harness; "
                           "PASS only on monotonic, cost-surviving, OOS spread")
