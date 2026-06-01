"""The live multi-agent trading system (master design Parts II-VII).

Three planes: Data (point-in-time, reuses `harness.data`), Agent (probabilistic,
LLM specialists + core trio), Deterministic (authoritative: sizing, Risk
Governor, execution, position management). Capital decisions exit only through
the deterministic plane. Paper-only by default; the live broker is gated behind
explicit credentials + a human-enable flag (asymmetric-autonomy invariant).
"""
