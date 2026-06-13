# Swing Trading System — contributor & agent guide

## What this is
An AI-native multi-agent swing-trading desk: a desktop app + engine that screens
the market, runs LLM analyst agents over the best names, manages the full trade
lifecycle on a broker (paper by default), and learns from its own results.

**Start here:** [`README.md`](README.md) for the overview, then
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (operating the app),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (modules & flows), and
[`docs/SIGNALS.md`](docs/SIGNALS.md) (the signal catalogue). The original design
is in `docs/design/`.

## Status
All phases are built and in active use: the validation harness (`harness/`), the
agent/risk/execution engine (`system/`), and the desktop application + live
workflow (`app/`). It runs offline + deterministic by default; real data, real
LLM agents, and a real broker are opt-in and gated.

> The original design called for a phased build (harness first). That phasing is
> complete — this is no longer "Phase 1 only". Treat the invariants below as
> binding; the phase restriction is historical.

## Non-negotiable invariants (enforced in code; never weaken)
- **Point-in-time integrity** — nothing timestamped after decision time T is ever
  visible at T. Guarded by `tests/test_leakage.py`.
- **Deterministic-first** — every agent/signal has a deterministic path; the
  whole system runs free and offline without an LLM.
- **Two keys** — a trade opens only if the PM chooses ENTER *and* the Risk
  Governor approves+sizes it. No agent ever sizes or places an order.
- **Asymmetric autonomy** — automation may only *reduce* risk; raising a limit,
  scaling capital, or enabling live trading requires a human.
- **Fail to PASS** — any agent error defaults the candidate to PASS.
- **Gap-aware fills & realistic costs** on every simulated trade.

## Stack
Python 3.11+, pandas, numpy, Tkinter (stdlib GUI). Optional: yfinance + requests
(real data), anthropic (real LLM), pyinstaller (packaging). Local parquet store;
no network except the data loaders and the broker.

## Working conventions
- New behaviour ships with a test; keep `pytest` green on every commit.
- After changing app/engine code, rebuild the exe (`build\build_windows.ps1`) and
  verify with `--selftest`; the running app locks the exe, so close it first.
- Agent reasoning is rendered in full (wrapped, never truncated) in the console
  and the saved logs.
