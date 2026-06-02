# AI-Native Multi-Agent Swing Trading System

A complete, end-to-end implementation of the design in
`docs/design/ai_native_MASTER_design_v1.0.md` and
`docs/design/ai_native_validation_harness_spec_v1.0.md`.

> **Honest framing.** This is a *system built to discover and exploit edges with
> discipline, and to shut itself down if none exist* — not a proven money-maker.
> It runs **paper-only by default**; live trading is gated behind explicit human
> opt-in. Synthetic data is used for offline demos and tests. Not investment advice.

## Layout

```
harness/                    # Phase 1: the validation harness (deterministic, no LLM)
  data/      pit_store, corp_actions, calendar, loader     # point-in-time store
  study/     costs (gap-aware), stats (Newey-West, deflated Sharpe), event_study
  signals/   base + free edges 1,2,6,7,8
  report/    per-edge scorecard + PASS/KILL + portfolio summary
  run.py     load -> signal -> study -> report

system/                     # Phases 2-5: the live multi-agent system (paper-default)
  config.py                 # hard risk limits + invariants, model tiering
  schemas.py                # JSON contracts (Appendix B) + deliberation envelope
  data_plane/indicators.py
  agents/                   # LLM specialists + core trio + meta (deterministic mock default)
  confluence.py             # cross-family confluence engine (deterministic)
  orchestrator.py           # deterministic cycle driver, fail-to-PASS
  risk/                     # sizing, correlation clusters, Risk Governor (authoritative)
  execution/                # PaperBroker (gap-aware) + gated AlpacaBroker, two-stage entry
  monitoring/               # health scorecard + kill switch (reduce-only)
  governance/               # holdout vault + champion/challenger promotion (human gate)
  reflection/memory.py      # decayed, human-reviewed lessons
  run_live.py               # the full daily cycle, wired end to end
```

## Load-bearing invariants (enforced in code)

- **Point-in-time integrity** — at decision time T, nothing timestamped after T is
  visible; no corporate action with `ex_date > T` touches returned prices. Guarded
  by `tests/test_leakage.py` (the leakage CI test — never weaken it).
- **Two keys** — a trade opens only if the PM chooses ENTER *and* the Risk Governor
  approves+sizes it. No agent ever sizes or places an order.
- **Asymmetric autonomy** — automation may only *reduce* risk; raising a limit,
  scaling capital, or enabling live trading requires a human.
- **Fail to PASS** — any agent error defaults the candidate to PASS; the system
  never fails *into* a trade.
- **Gap-aware fills, realistic costs** on every simulated trade.

## Run it (offline, deterministic — no network, no API keys)

```bash
pip install -e ".[dev]"

pytest                       # full suite incl. the leakage CI test
python -m harness.run        # Phase-1 validation: edges 1,2,6,7,8 -> scorecards
python -m system.run_live    # end-to-end paper-trading cycle (MockLLMClient)
```

`harness.run` recovers the planted Edge-1 signal (PASS) and KILLs the noise edges,
applying a deflated-Sharpe multiple-testing correction at the portfolio level.

## Desktop app (GUI)

A Tkinter GUI to configure API keys/parameters and run the harness or paper
engine with live output, packaged as a standalone executable.

```bash
pip install -e ".[dev,gui]"
python -m app.main                 # launch the GUI
.\build\build_windows.ps1          # -> dist\SwingSystem.exe (Windows)
bash build/build_macos.sh          # -> dist/SwingSystem.app (run on a Mac)
```

See `build/README.md` for details. Keys are stored locally in
`~/.swing_system/config.json` (never committed or bundled). Live-data, real-LLM,
and live-broker paths are configurable but gated; one-click runs stay paper-only.

## Enabling real components (opt-in)

- **Real data (wired):** `pip install -e ".[live-data]"`. The app's `data_source="live"`
  pulls **real free data via yfinance** (`harness.data.loader.LiveLoader` /
  `fetch_prices_yahoo`) — raw prices + splits/dividends, so the point-in-time
  as-of-T adjustment holds on real data. Cached under `~/.swing_system/data_store`.
  `fetch_edgar_submissions` pulls EDGAR filing metadata (SEC requires a `User-Agent`).
  Filing-text / Form-4-detail / news tables for live data are a further step, so on
  real data today only the price-based momentum edge (and the paper engine) have inputs.
- **Real LLM agents:** `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY`.
  `system.agents.llm_client.default_client()` then returns the Anthropic adapter
  (pinned models, temperature 0, structured output, prompt caching). Without a key
  everything runs deterministically via `MockLLMClient`.
- **Live broker:** `AlpacaBroker` refuses to construct unless `enable_live=True`
  and credentials are supplied; order wiring is intentionally left to a human.
