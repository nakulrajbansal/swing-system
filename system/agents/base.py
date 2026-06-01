"""Agent base: deterministic-or-LLM dispatch with schema validation.

Each agent has a single responsibility, a pinned model, a versioned system
prompt, and a strict output schema. Under deterministic (mock) mode the agent
runs its domain logic directly; under a real client it emits least-context JSON
and parses the structured response. Either way the result is schema-validated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from system.agents.llm_client import LLMClient
from system.agents.prompts import PROMPT_VERSION


class Agent(ABC):
    name: str = "agent"
    schema_name: str = "object"

    def __init__(self, client: LLMClient, model: str, system_prompt: str,
                 temperature: float = 0.0, max_tokens: int = 1500):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = PROMPT_VERSION

    def run(self, inputs: dict):
        if self.client.deterministic:
            obj = self.deterministic(inputs)
        else:
            raw = self.client.complete(
                self.system_prompt, inputs, self.schema_name,
                model=self.model, max_tokens=self.max_tokens,
                temperature=self.temperature)
            obj = self.parse(raw, inputs)
        return obj.validate()

    @abstractmethod
    def deterministic(self, inputs: dict):
        """Domain-logic read used in offline/CI mode."""

    def parse(self, raw: dict, inputs: dict):
        """Default real-mode parse: subclasses override to map dict -> schema."""
        raise NotImplementedError(f"{self.name}: real-mode parse not wired")
