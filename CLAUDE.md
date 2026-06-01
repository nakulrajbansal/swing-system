# Swing Trading System

## What this is
An AI-native multi-agent swing trading system. The complete design is in
docs/design/ai_native_MASTER_design_v1.0.md. READ IT before proposing changes.

## Current phase
Phase 1 ONLY: build the validation harness per
docs/design/ai_native_validation_harness_spec_v1.0.md.
Do NOT build agents, LLM calls, execution, or live data yet (out of scope, see spec section 9).

## Non-negotiable rules (from the design)
- Point-in-time integrity: nothing timestamped after T is ever visible at decision time T.
  Add a CI leakage test and never weaken it.
- Deterministic-first: this harness uses NO LLM calls.
- Gap-aware fills: never assume a fill at the stop price on a gap.
- Realistic costs on every simulated trade.

## Stack
Python 3.11+, pandas, numpy. Local parquet/SQLite store. No network except the data loader.

## Build order (from the harness spec section 10)
1. pit_store.py + corp_actions.py + leakage CI test
2. data/loader.py (EDGAR + free prices)
3. study/ (event_study, stats, costs)
4. signals/edge01_filing.py, validate end to end
5. remaining free edges (2, 6, 7, 8)
6. report/