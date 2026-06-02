"""LLM client interface (master §6).

`MockLLMClient` is the default and marks DETERMINISTIC mode: agents compute their
reads with deterministic domain logic (no tokens, no network) so the whole system
runs offline and in CI. `AnthropicClient` is the real adapter; it activates only
when an API key is supplied. It uses pinned model IDs, temperature 0, structured
JSON output via a tool schema, and prompt caching on the (static) system prompt.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    deterministic: bool = False

    @abstractmethod
    def complete(self, system: str, payload: dict, schema_hint: str,
                 model: str, max_tokens: int = 1500, temperature: float = 0.0) -> dict:
        ...


class MockLLMClient(LLMClient):
    """Deterministic mode marker. Agents use their own domain logic instead of
    calling out. ``complete`` is never invoked in this mode."""

    deterministic = True

    def complete(self, *a, **k) -> dict:  # pragma: no cover - not used in mock mode
        raise RuntimeError(
            "MockLLMClient.complete should not be called; agents run their "
            "deterministic path under mock mode."
        )


class CallBudgetExceeded(RuntimeError):
    """Raised when an AnthropicClient passes its per-run API call cap."""


class AnthropicClient(LLMClient):
    """Real Anthropic adapter (gated by ANTHROPIC_API_KEY). Lazy SDK import.

    A hard per-run call budget protects against runaway spend: each `complete`
    increments a counter and raises once the cap is hit. The cap defaults to
    ANTHROPIC_MAX_CALLS (or 200). Callers that swallow agent errors (the
    orchestrator) will then simply stop using the LLM for the rest of the run.
    """

    deterministic = False

    def __init__(self, api_key: str | None = None, max_calls: int | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No ANTHROPIC_API_KEY. Use MockLLMClient for offline/deterministic "
                "runs, or supply a key to enable real specialist agents."
            )
        self.max_calls = int(max_calls if max_calls is not None
                             else os.environ.get("ANTHROPIC_MAX_CALLS", 200))
        self.calls = 0
        from anthropic import Anthropic  # lazy
        self._client = Anthropic(api_key=self.api_key)

    def complete(self, system: str, payload: dict, schema_hint: str,
                 model: str, max_tokens: int = 1500, temperature: float = 0.0) -> dict:
        if self.calls >= self.max_calls:
            raise CallBudgetExceeded(
                f"LLM call budget ({self.max_calls}) reached; stopping to cap cost. "
                "Raise ANTHROPIC_MAX_CALLS to allow more.")
        self.calls += 1
        # Structured output via a single tool the model must call.
        tool = {
            "name": "emit",
            "description": f"Return the {schema_hint} JSON object.",
            "input_schema": {"type": "object", "additionalProperties": True},
        }
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            # Prompt caching on the static system prompt cuts cost on repeated calls.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user",
                       "content": json.dumps(payload, default=str, sort_keys=True)}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("model did not return the structured tool output")


def default_client() -> LLMClient:
    """Anthropic if a key is present, else the deterministic mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except Exception:
            pass
    return MockLLMClient()
