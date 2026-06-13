# Architecture — how the Swing System fits together

This is the developer's map: the three layers, the end-to-end flows, the
point-in-time data model, and a module-by-module reference of every file and its
key functions. Individual functions are documented in their docstrings; this
document gives you the structure to find and understand them.

---

## Contents

- [Three layers](#three-layers)
- [End-to-end flows](#end-to-end-flows)
- [The point-in-time data model](#the-point-in-time-data-model)
- [The agent model](#the-agent-model)
- [Module reference](#module-reference)
  - [`app/` — application & live workflow](#app--application--live-workflow)
  - [`system/` — agent, risk & execution engine](#system--agent-risk--execution-engine)
  - [`harness/` — Phase-1 validation](#harness--phase-1-validation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Conventions](#conventions)

---

## Three layers

| Layer | Package | Role | LLM? |
|---|---|---|---|
| **Application** | `app/` | The desktop GUI and every operator flow (screen, deep-dive, review, backtest, momentum, watch, curate). Talks to the engine. | drives it |
| **Engine** | `system/` | The agents, the deliberation orchestrator, the Risk Governor, execution, and the reflection/learning subsystem. | yes |
| **Harness** | `harness/` | Point-in-time data store and the deterministic Phase-1 validation that proves an edge before it's allowed to trade. | no |

The dependency direction is `app → system → harness`. The harness is fully
deterministic and never imports the others.

---

## End-to-end flows

### A. The screen (the main flow)
`app.runner.run_screen`:
1. Resolve the universe (`_screen_universe`) and download closes+volumes once
   (`fetch_closes_volumes_batch`), cached per day.
2. Score matured ledger recommendations off the fresh prices and run the curator
   (`reco_ledger.evaluate` → `curator.curate`).
3. Read the market regime and **self-tune** the factor weights
   (`market_regime`, `strategy.select_preset`, `strategy.factor_weights`).
4. Pre-filter & rank the whole universe (`screen.prescreen` →
   `strategy.composite_score`), dropping corrupt and illiquid names.
5. Build the **shortlist** (`strategy.select_shortlist`): sector- and
   correlation-diversified top names + reserved hidden-gem slots.
6. Compute the **self-throttle** (`calibration.desk_throttle`).
7. For each shortlisted name, `_analyze_symbol` runs the full deep-dive (flow B).
8. Apply the throttle's raised bar, build the suggested portfolio
   (`strategy.construct_portfolio`), record to the ledger, populate the watchlist.

### B. The deep-dive (one name)
`app.runner._analyze_symbol` → `system.run_live.PaperTradingEngine` →
`system.orchestrator.Orchestrator.deliberate_symbol`:
1. Build the PIT evidence packet (`evidence.assemble_evidence`).
2. Run the six analysts (`system.agents.analysts.*`) — each an `AnalystRead`.
3. Strategist proposes (`HypothesisAgent`), Skeptic attacks (`SkepticAgent`,
   blind to conviction), Strategist rebuts; if the call is *contested*
   (high severity or split analyst panel) the Skeptic gets a round-2
   `rejoin`; the Portfolio Manager decides (`PortfolioManagerAgent`).
4. Any exception → the candidate defaults to PASS (fail-to-PASS).
5. The Risk Governor sizes any ENTER/ADJUST (`risk.governor.RiskGovernor`),
   off the real account equity; calibration attaches a P(win).

### C. Review exits (the open book)
`app.runner.run_position_review`: read broker positions → arm missing protective
orders → time-exit past-due plans → Guardian hold/exit on the rest → breakeven
coaching → score broker-side closes and matured recs → run the curator.

### D. Learning
Every scored outcome (from C, or from a screen's `reco_ledger.evaluate`) updates
the ledger and lesson memory, which feed: calibration (P win), the curator
(activate/retire/write lessons), the gem-slot count, and the self-throttle. The
loop is closed and runs automatically whenever outcomes are scored.

### E. Phase-1 validation (offline, no LLM)
`harness.run`: load a PIT store → for each edge, run an event study with gap-aware
costs → score with Newey-West / deflated-Sharpe stats → PASS/KILL per edge and a
portfolio verdict. The app's "Only trade validated edges" gate reads this.

---

## The point-in-time data model

The harness `PITStore` (`harness/data/pit_store.py`) is the spine. It stores one
parquet per table and answers **as-of** queries: `store.as_of(T)` returns an
`AsOfView` where every accessor only sees rows with `available_at <= T` (and no
corporate action with `ex_date > T` adjusts prices). `T` must be timezone-aware —
a tz-naive instant is rejected, because it would be ambiguous and could leak.

Tables (`TABLES` in `pit_store.py`):

| Table | Holds | PIT column |
|---|---|---|
| `prices` | OHLCV per symbol/session | `date` |
| `corp_actions` | splits/dividends | `ex_date` |
| `constituents` | index membership windows | `start_date`/`end_date` |
| `filings` | 8-K / 10-K / 10-Q (+ risk-factor text) | `available_at` |
| `form4` | insider transactions | `available_at` |
| `fundamentals` | a contemporaneous snapshot (multiples, growth, estimates, positioning) | `available_at` |
| `fundamentals_history` | quarterly revenue/margins from EDGAR XBRL | `available_at` (filing date) |
| `news` | headlines | `available_at` |
| `links` | economic-link map | `available_at` |

This is why a backtest is honest and why "Deep-dive" never sees tomorrow's news.
The leakage guarantee is enforced by `tests/test_leakage.py` — the one test you
must never weaken.

---

## The agent model

All agents subclass `system.agents.base.Agent`: a pinned model, a versioned
system prompt, and a strict output schema. Each has two implementations behind
one `run()`:
- `deterministic(inputs)` — domain logic over the same numeric evidence, used
  with the `MockLLMClient` (offline/CI/free). Produces coherent, grounded reads.
- `parse(raw, inputs)` — maps a real model's structured JSON onto the schema.

`run()` dispatches on `client.deterministic` and always `validate()`s the result.
This is why the entire pipeline runs identically with or without an API key, and
why offline tests fully exercise the decision logic.

The roster:

| Agent | File | Output | Job |
|---|---|---|---|
| MacroAnalyst | `analysts.py` | `AnalystRead` | one market-wide backdrop read |
| TechnicalAnalyst | `analysts.py` | `AnalystRead` | trend / momentum / timing |
| FundamentalAnalyst | `analysts.py` | `AnalystRead` | filings, insider, news tone |
| ValuationAnalyst | `analysts.py` | `AnalystRead` | multiples, growth-adjusted |
| GrowthAnalyst | `analysts.py` | `AnalystRead` | growth, guidance, est. revisions |
| MoatAnalyst | `analysts.py` | `AnalystRead` | durable advantage + secular trend + trajectory |
| HypothesisAgent (Strategist) | `core.py` | `TradeHypothesis` | propose one thesis or decline |
| SkepticAgent | `core.py` | `Critique` (+ `rejoin`) | adversarial objections, contested round 2 |
| PortfolioManagerAgent | `core.py` | `RiskDecision` | ENTER / ADJUST / PASS |
| EdgeSpecialist | `specialists.py` | `SpecialistRead` | wraps a harness edge as a signal |
| GuardianAgent | `meta.py` | `GuardianDecision` | re-check an open position; exit only |
| ReflectionAgent | `meta.py` | `Lesson` | draft one lesson from a closed trade |
| ResearcherAgent | `meta.py` | `ResearchProposal` | propose edges (no authority) |

The orchestrator (`system/orchestrator.py`) sequences them, records a full
transcript per candidate, and enforces fail-to-PASS.

---

## Module reference

### `app/` — application & live workflow

- **`main.py`** — process entry. `main(argv)` launches the GUI, or dispatches a
  headless flag: `--selftest` (`_selftest`), `--screen [index]`, `--review`,
  `--watch`, `--daily` (review then screen). `_headless` runs one flow with the
  saved config and tees output to a temp log.
- **`gui.py`** — the Tkinter app (`SwingApp`). Builds the sidebar + four pages
  (`_build_run`, `_build_performance`, `_build_lessons`, `_build_config`); the
  themed dark style (`_setup_style`, `_theme_titlebar`); the order-ticket table
  (`_show_orders`, `_place_order`); the watchlist card; the console with
  jump-to-section, colour tagging, and the Stop button (`_console`, `_log`,
  `_jump_section`, `_stop_run`); run dispatch on a worker thread with a queue
  pump (`_start`, `_drain_queue`); scheduling (`_schedule_daily`). Pure
  presentation — all work is delegated to `app.runner`.
- **`runner.py`** — every flow the buttons (and the CLI) call. Each is
  `fn(cfg, emit)` and tees to a timestamped log:
  - `run_screen` — the screen funnel (flow A).
  - `run_recommendations` — single-ticker deep-dive (live) or the synthetic
    scan demo; `_analyze_symbol` is the per-name deep-dive core (flow B).
  - `run_position_review` — manage the open book (flow C); `_infer_exit_reason`,
    `_r_multiple`, `_run_curator` are its helpers.
  - `run_watch` — check the watchlist, toast on a trigger (`_toast`).
  - `run_curation` — on-demand curator pass.
  - `run_strategy_backtest` — the walk-forward vs the benchmark.
  - `run_momentum_trade` — the mechanical one-name momentum strategy.
  - `run_portfolio_status`, `place_manual_order`, `check_alpaca`,
    `run_trade_history` — broker read/write flows.
  - Shared helpers: `_resolve_equity`/`_account_snapshot` (real-account sizing),
    `_build_store`/`_build_ticker_store` (PIT stores), `_resolve_client` (LLM
    preflight + fallback), `_print_transcript`/`_analysis_summary` (the readable
    scorecard), `_wrap` (full-text, never-truncated formatting), `RunStopped`/
    `request_stop` (cooperative cancellation), `_staleness_note`.
- **`screen.py`** — the price-only pre-filter. `_metrics` computes per-name
  signals (momentum, acceleration, trend quality, RSI, distance-from-high, RVOL,
  up/down volume, entry stretch, earnings-gap drift); `_volume_metrics` the
  liquidity/accumulation reads; `market_regime` the benchmark backdrop;
  `prescreen` ranks the universe with the liquidity floor and corrupt-data drop.
- **`strategy.py`** — the strategy brain (pure, testable). `composite_score`
  blends the factors; `BASE_WEIGHTS` + `WEIGHT_PRESETS` + `factor_weights` +
  `select_preset` are the regime/performance-adaptive, self-tuned weighting;
  `select_shortlist` + `diversify` + `correlation_diversify` build the deep-dive
  list; `hidden_gem_score` + `gem_slot_count` the early-acceleration lens;
  `construct_portfolio` the sized allocations (with the throttle's `gross_scale`);
  `walk_forward_backtest` the honest backtest.
- **`reco_ledger.py`** — the recommendation ledger. `record` appends BUYs;
  `evaluate` scores matured ones (and excess vs SPY) into lesson memory;
  `mark_executed`/`mark_closed` link orders and realize closes; `cohort_stats`
  the lens breakdown; `open_for`/`summarize` reads.
- **`watchlist.py`** — `add`/`active`/`remove` (persisted, auto-expiring) and the
  pure `watch_hits` trigger logic (pullback window / breakout on RVOL).
- **`config.py`** — `AppConfig` dataclass (all settings + keys), `load`/`save`
  (`~/.swing_system/config.json`), `apply_to_env`.
- **`momentum.py`** — the local momentum-position tracker (entry/exit dates that
  Alpaca doesn't store). **`gating.py`** — the validated-edge gate
  (`load_validated`/`save_validated`). **`learning.py`** — lesson-memory IO and
  `summarize`.

### `system/` — agent, risk & execution engine

- **`agents/`** — `base.py` (the `Agent` ABC + dispatch), `prompts.py` (every
  versioned system prompt), `llm_client.py` (`MockLLMClient`, the Anthropic
  adapter, `default_client`), `analysts.py` (the six specialists), `core.py`
  (the trio), `specialists.py` (`EdgeSpecialist`), `meta.py`
  (Guardian/Reflection/Researcher).
- **`data_plane/`** — `evidence.py` (`assemble_evidence` and the per-domain
  builders: `_technicals`, `_fundamentals`, `_trajectory`, `_events`, `_filings`,
  `_insider`, `_news`/`_news_sentiment`); `indicators.py` (ATR, RSI, SMA, etc.).
- **`orchestrator.py`** — `Orchestrator`: `run_cycle` (confluence → deliberate
  each candidate) and `deliberate_symbol` (force one name); `_deliberate`
  sequences the trio with the contested round-2 escalation and fail-to-PASS,
  producing a `CycleResult` and a full transcript.
- **`confluence.py`** — `run_confluence`: cross-family agreement gating
  (≥2 families or one very strong) → `Candidate`s.
- **`risk/`** — `governor.py` (`RiskGovernor.evaluate` → an `OrderTicket`: the
  authoritative 1%-risk / ATR-stop / capped sizing); `sizing.py`; `clusters.py`
  (correlation clusters).
- **`execution/`** — `broker.py`: the `Broker` ABC, `PaperBroker` (gap-aware
  fills, two-stage entry), and `AlpacaBroker` (account/positions/orders/fills;
  `submit_entry`, `submit_manual`, `submit_exit_orders`, `close_position`;
  refuses to construct for live without the gate).
- **`reflection/`** — `memory.py` (`LessonMemory`: decayed, recallable lessons +
  outcomes), `calibration.py` (`calibration_table`, `calibrated_probability`,
  `desk_throttle`), `curator.py` (`assess`, `curate` — the evidence-gated
  learning).
- **`monitoring/`** — `scorecard.py` (`HealthScorecard`) and `kill_switch.py`
  (`KillSwitch`, reduce-only) for the paper engine.
- **`governance/`** — `holdout.py` + `promotion.py` (champion/challenger behind a
  human gate). **`run_live.py`** — `PaperTradingEngine`: wires store + agents +
  governor + broker into the daily cycle the app and demos drive.
- **`schemas.py`** — every JSON contract (`AnalystRead`, `TradeHypothesis`,
  `Critique`, `RiskDecision`, `Lesson`, …) with `validate()`. **`config.py`** —
  the frozen `SystemConfig`: hard `RiskLimits`, `SizingParams`, `ModelTiering`,
  debate thresholds. These limits are invariants — no agent can raise them.

### `harness/` — Phase-1 validation

- **`data/`** — `pit_store.py` (the PIT store + as-of view), `corp_actions.py`
  (adjustment math), `calendar.py` (sessions), `loader.py` (the data plane:
  `SyntheticLoader`, `LiveLoader`, and the fetchers — `fetch_prices_yahoo`,
  `fetch_closes_volumes_batch`, `fetch_fundamentals_yahoo`,
  `fetch_fundamental_history`/`parse_companyfacts`, `fetch_news_yahoo`,
  `fetch_options_positioning`, `fetch_edgar_for_symbol`, `fetch_insider_quarter`),
  the universe modules (`sp500.py`, `nasdaq100.py`, `midsmall.py`, shared
  `wiki.py`), `macro.py`, `reddit.py`.
- **`study/`** — `costs.py` (gap-aware fills + realistic costs), `stats.py`
  (Newey-West t-stats, deflated Sharpe), `event_study.py` (`run_event_study`).
- **`signals/`** — `base.py` + the free edges: `edge01_filing`, `edge02_8k`,
  `edge06_insider`, `edge07_links`, `edge08_momo`.
- **`report/`** — `report.py`: `edge_scorecard`, `format_scorecard`,
  `portfolio_summary`. **`run.py`** — the load → signal → study → report driver.

---

## Configuration

Two config objects:
- **`app.config.AppConfig`** — user-facing settings + credentials, saved to
  `~/.swing_system/config.json`. The GUI edits it; the CLI loads it.
- **`system.config.SystemConfig`** (frozen) — the hard invariants: risk limits,
  ATR/sizing params, model tiering, confluence and debate thresholds. Changing
  these is a deliberate, version-controlled code change — never a runtime knob.

---

## Testing

`pytest` runs the full suite offline (no network, no keys), including:
- `test_leakage.py` — the point-in-time integrity guarantee.
- `test_corp_actions.py`, `test_costs.py`, `test_event_study.py` — the harness.
- `test_agents.py`, `test_curator.py`, `test_confluence.py`, `test_governor.py`,
  `test_kill_switch.py`, `test_governance.py` — the engine.
- `test_screen.py`, `test_strategy.py`, `test_discovery.py`, `test_phase_abc.py`,
  `test_learning.py`, `test_macro.py`, `test_paper_engine.py` — the app/strategy.

New behaviour ships with a test; the suite is green on every commit, and the
packaged exe is verified with `--selftest` after each build.

---

## Conventions

- **Deterministic-first** — every agent and signal has a deterministic path so the
  whole system runs free and offline; the real LLM is an upgrade, not a
  dependency.
- **Point-in-time always** — anything that reaches a decision goes through an
  `as_of(T)` view; `available_at` is mandatory on event rows.
- **Reduce-only automation** — code that runs unattended (review, guardian,
  curator, throttle) can only de-risk; anything that adds risk needs a human.
- **Full reasoning, never truncated** — agent text is wrapped, not elided, in
  both the console and the saved logs.
