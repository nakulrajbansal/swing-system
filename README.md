# Swing System — an AI-native multi-agent swing-trading desk

A desktop application and engine that screens the US stock market, runs a panel
of AI analyst agents over the best candidates, manages the full trade lifecycle
on a broker (paper by default), and **learns from its own realized results** —
calibrating its confidence, auditing its lessons, and standing itself down when
it stops paying.

> **Honest framing.** This is a *system built to discover and exploit edges with
> discipline, and to shut itself down if none exist* — **not a proven
> money-maker, and not investment advice.** It runs **paper-only by default**;
> real-money trading is gated behind explicit, multi-step human opt-in.
> Everything works offline with synthetic data and zero API keys.

---

## Table of contents

- [What it does](#what-it-does)
- [The five-minute mental model](#the-five-minute-mental-model)
- [Install & run](#install--run)
- [The desktop app, tab by tab](#the-desktop-app-tab-by-tab)
- [How a recommendation is made (the pipeline)](#how-a-recommendation-is-made-the-pipeline)
- [The signals & edges](#the-signals--edges)
- [The learning loop](#the-learning-loop)
- [Safety model & invariants](#safety-model--invariants)
- [Data sources](#data-sources)
- [Automation (run it daily, hands-off)](#automation-run-it-daily-hands-off)
- [Project layout](#project-layout)
- [Deeper docs](#deeper-docs)

---

## What it does

1. **Screens** an entire index (S&P 500, Nasdaq-100, S&P 400/600 mid/small caps,
   or all of them) with a fast, free, price-only pre-filter — relative strength,
   momentum, acceleration, trend quality, proximity to highs, volume/accumulation,
   and entry timing — bounded to one batched price download.
2. **Deep-dives** only the best handful with a full panel of LLM analyst agents
   (technical, fundamental/forensic, valuation, growth, moat & secular-trend,
   macro) plus a contested **Strategist → Skeptic → Portfolio Manager**
   deliberation, over real evidence (prices, SEC filings & XBRL fundamentals,
   insider buys, news, options/short positioning).
3. **Sizes** every recommendation off your **real broker account equity** through
   a Risk Governor (1%-risk, ATR stops, per-name and per-sector caps), and shows
   a calibrated **win probability** earned from its own past calls.
4. **Executes** selectively to Alpaca (paper or, behind gates, live), arming
   protective stop/target orders.
5. **Manages** the open book: time-exits per the trade's plan, a Guardian agent
   that exits on thesis-breaking news, and breakeven-stop coaching on winners.
6. **Learns**: every recommendation is scored against its realized outcome, a
   **curator** activates/retires lessons on evidence, the desk **calibrates its
   conviction**, **self-tunes its factor weights**, and **throttles itself**
   during cold streaks.

It ships as a polished Windows/macOS desktop app (Tkinter) and as a headless CLI
for scheduled, hands-off operation.

---

## The five-minute mental model

```
  pick a universe ─▶ FREE PRE-FILTER (1500 names, price-only, seconds)
                         │  ranks by a blended opportunity score
                         ▼
                     SHORTLIST  (top ~5, sector- & correlation-diversified,
                         │        with reserved "hidden-gem" slots)
                         ▼
                     DEEP-DIVE per name (costs LLM tokens, bounded to the shortlist)
                         │   gather EVIDENCE ─▶ 6 ANALYST AGENTS
                         │   Strategist proposes ─▶ Skeptic attacks ─▶ (contested? round 2)
                         │   Portfolio Manager decides: ENTER / ADJUST / PASS
                         ▼
                     RISK GOVERNOR sizes it off your real account → ORDER TICKET
                         │   (you review & place; or it routes to Alpaca)
                         ▼
                     REVIEW EXITS (daily): time-exit · Guardian · arm stops · score closes
                         ▼
                     LEARN: ledger → calibration (P win) → curator (lessons) →
                            self-tune weights → self-throttle on cold streaks
```

Two things are **always true**: nothing timestamped after the decision moment is
ever visible to a decision (point-in-time integrity, CI-tested), and automation
can only ever *reduce* risk — opening real-money risk needs a human.

---

## Install & run

Python 3.11+. Everything below works offline with no keys.

```bash
pip install -e ".[dev]"          # core + tests (no network, no LLM)

pytest                           # full suite incl. the point-in-time leakage test
python -m app.main               # launch the desktop app (Tkinter; stdlib)
python -m harness.run            # Phase-1 edge validation -> scorecards
python -m system.run_live        # end-to-end paper-trading cycle (deterministic mock)
```

Optional extras, each independent:

```bash
pip install -e ".[live-data]"    # yfinance + requests: real prices, SEC filings, news
pip install -e ".[llm]"          # anthropic: real LLM agents (else a deterministic mock)
pip install -e ".[gui]"          # pyinstaller: build the standalone executable
```

Build the standalone app:

```bash
.\build\build_windows.ps1        # -> dist\SwingSystem.exe   (run on Windows)
bash build/build_macos.sh        # -> dist/SwingSystem.app   (run on a Mac)
```

Without any extras the app runs in **synthetic + deterministic** mode (a planted
demo universe, the `MockLLMClient`) — ideal for trying the whole workflow with no
accounts. Add keys on the **Settings** tab to switch on real data, real agents,
and a real broker.

---

## The desktop app, tab by tab

A left sidebar switches between four pages. A live header shows what the next run
will use (`DATA · LIVE/SYNTHETIC`, `AGENTS · LLM/DETERMINISTIC`,
`SIZING · REAL ACCOUNT/CONFIGURED`) and a `PAPER` / `⚠ LIVE` badge.

### Desk
The cockpit. From top to bottom:
- **Find opportunities** — a universe picker (S&P 500 / Nasdaq-100 / S&P 400 /
  S&P 600 / mid+small "hidden gems" / S&P 1500 broad) with **Run screen**
  (`Ctrl+R`), and a ticker box with **Deep-dive** for a single-name analysis of
  *any* US symbol.
- **Trade & monitor** — **Review exits** (manage the open book), **Portfolio
  P&L**, **Momentum auto-trade** (a separate mechanical one-name strategy),
  **Strategy backtest**, **Reddit sentiment**.
- **Order tickets** — BUY recommendations from the last run as an editable table
  (symbol, conviction, P(win), ref price, qty, type, limit, stop/target,
  **Place order**). A ◆ marks a hidden-gem pick, a ● marks a name you already
  hold, and the limit pre-fills to the PM's pullback entry when it advised
  waiting for a dip.
- **Watchlist** — names whose *time* hasn't come (WATCH-tier ideas, PM pullback
  calls) with their trigger levels, and **Check triggers now**.
- **Output** — the live, colour-coded console (full agent reasoning, never
  truncated) with a **Jump to section** outline, Copy, Clear, Open-logs, and a
  **Stop run** button (`Esc`).

### Performance
The honest scoreboard, all from your own realized results: an **equity curve** of
scored recommendations, the **calibration table** (does stated conviction match
realized win rate?), **lens cohorts** (hidden-gem vs core vs moat-bullish),
recent closed calls, and **executed trades** (your account's actual fills, via
**Refresh trade history**).

### Learning
The desk's accumulated **lessons + base rates** that inform future runs, plus the
recommendation ledger. **Run AI curator** audits the record and activates/retires
lessons on evidence (this replaced any manual "approve" step — see
[the learning loop](#the-learning-loop)).

### Settings
Credentials (Anthropic, Alpaca, EDGAR User-Agent, Reddit), data/universe options,
strategy parameters, feature toggles, an **Automation** card (schedule the daily
run + watchlist checks via Windows Task Scheduler), and **Test Alpaca connection**.
Keys are saved only to `~/.swing_system/config.json` (never committed or bundled).

---

## How a recommendation is made (the pipeline)

1. **Universe & pre-filter** (`app/screen.py`, `app/strategy.py`). One batched
   download of closes+volumes for the whole index. Each name gets price-only
   metrics; corrupt series (single-session split artifacts) and illiquid names
   (below a dollar-volume / price floor) are dropped. A composite **opportunity
   score** blends the factors below; a market-regime read (benchmark vs its
   200-DMA) can stand the desk down in a downtrend.
2. **Shortlist** (`strategy.select_shortlist`). The top names by score, with a
   light sector cap and **correlation de-duplication** (near-identical names —
   e.g. dual-class shares — are not deep-dived twice), plus reserved
   **hidden-gem** slots for early-acceleration names a pure momentum ranking
   would miss. The slot count self-tunes from the gem lens's realized hit rate.
3. **Evidence** (`system/data_plane/evidence.py`). For each shortlisted name, a
   compact point-in-time packet: technicals, valuation/growth/guidance, a moat
   block (margins, FCF, business summary), a quarterly **trajectory** from EDGAR
   XBRL, scheduled-earnings events, filings & insider activity, news + its tone,
   and short-interest / options positioning.
4. **Analyst panel** (`system/agents/analysts.py`). Six specialists each produce
   a written stance + score: **macro** (one market-wide read), **technical**,
   **fundamental/forensic**, **valuation**, **growth**, and **moat &
   secular-trend** (the "find the next leader" lens).
5. **The trio** (`system/agents/core.py`, `system/orchestrator.py`). A
   **Strategist** proposes one thesis or declines; a **Skeptic** (blind to the
   proposer's conviction) attacks it; the Strategist rebuts; on *contested* calls
   the Skeptic gets a second round; the **Portfolio Manager** weighs it all and
   chooses ENTER / ADJUST (sound thesis, wait for a pullback) / PASS. Any agent
   error defaults the name to PASS — the system never fails *into* a trade.
6. **Risk Governor** (`system/risk/governor.py`). The authoritative sizer: 1%
   account risk, ATR-based stop, per-name and per-sector caps, sized off your
   **real** account equity when broker keys are present. No agent ever sizes or
   places an order — that's the "two keys" rule.
7. **Calibration & throttle** (`system/reflection/calibration.py`). The
   recommendation gets a **P(win)** — its conviction shrunk toward the desk's
   own realized win rate for that conviction band. If the desk is on a cold
   streak, the **self-throttle** raises the entry bar and caps exposure.
8. **Ticket & ledger**. Survivors become order tickets and are logged to the
   recommendation ledger for forward scoring.

---

## The signals & edges

The price-only pre-filter blends (weights in `strategy.BASE_WEIGHTS`, regime- and
performance-adaptive, and nightly **self-tuned** across presets):

| Signal | What it captures |
|---|---|
| Relative strength | Out-performance vs the market over 6 months |
| Momentum (3/6/12-mo) | Trend strength and persistence |
| Acceleration | 3-month pace overtaking the 6-month average — an *igniting* trend |
| Trend & trend-quality | Above the 200-DMA; 50-DMA above a *rising* 200-DMA (weekly confirmation) |
| Entry timing | Resting near the 20-DMA (buyable) vs stretched far above it (a chase) |
| Volume / RVOL / up-down | Accumulation footprints and live ignition |
| Earnings-gap drift (PEAD) | A held post-earnings gap, recency-weighted |
| Near-high & sector rotation | Proximity to 52-week highs; leading sectors |

The deep-dive adds **fundamental & event** signals the agents reason over:
moat/margins/FCF, the quarterly **trajectory** (revenue acceleration + margin
expansion = the pre-rally inflection), **estimate-revision momentum** (analysts
racing to upgrade an emerging leader — a documented leading edge), **earnings
quality** (an accrual red flag when profit doesn't convert to cash), scheduled
**earnings-event risk**, **news tone**, and **short-interest / options
positioning**. See [`docs/SIGNALS.md`](docs/SIGNALS.md) for the full catalogue,
rationale, and measured backtest deltas.

---

## The learning loop

This is what makes it *AI-native* rather than a static screener. Authority is
earned through **realized evidence**, never human sign-off:

- **Recommendation ledger** (`app/reco_ledger.py`) — every BUY is recorded with
  its conviction, lens tags, and plan; once matured (or closed at the broker) it
  is scored against its actual return, market-relative excess included.
- **Calibration** (`system/reflection/calibration.py`) — conviction bands are
  mapped to realized win rates; new recommendations show an honest **P(win)**
  ("58% — calibrated on 23 calls" vs "~55% — uncalibrated, n=4").
- **Curator** (`system/reflection/curator.py`) — reviews the record, **activates**
  a drafted lesson only when the aggregate for its setup backs it, **retires**
  anecdotes the record contradicts, and **writes pattern lessons** (conviction
  miscalibration, lens performance, stop-heavy exit mixes) gated on a minimum
  number of scored calls. Runs automatically whenever outcomes are scored.
- **Self-tuning weights** (`strategy.select_preset`) — a nightly trailing
  walk-forward picks the factor-weight preset currently winning.
- **Self-throttle** (`calibration.desk_throttle`) — on a cold streak or hot
  convictions, raises the entry bar and cuts gross exposure; dormant until enough
  trades are scored, and strictly risk-reducing.

The point: the system gets sharper as its ledger fills with *your* outcomes. A
fresh install starts honest-but-uncalibrated and improves with use.

---

## Safety model & invariants

Enforced in code and guarded by tests:

- **Point-in-time integrity** — at decision time T, nothing timestamped after T
  is visible; no corporate action with `ex_date > T` touches returned prices.
  Guarded by `tests/test_leakage.py` — *never weaken it.*
- **Two keys** — a trade opens only if the Portfolio Manager chooses ENTER *and*
  the Risk Governor approves and sizes it. No agent ever sizes or places orders.
- **Asymmetric autonomy** — automation may only *reduce* risk. Raising a limit,
  scaling capital, placing real-money orders, or enabling live trading all
  require explicit human action. The Guardian can exit but never add; the
  curator governs learning, never money.
- **Fail to PASS** — any agent error defaults the candidate to PASS; the desk
  never fails *into* a trade.
- **Gap-aware fills & realistic costs** on every simulated trade.
- **Paper by default** — `AlpacaBroker` refuses to construct for live unless
  `enable_live_trading` is on *and* the environment is explicitly `live`.

---

## Data sources

All free; each independent and gracefully degrading if absent:

- **Prices + corporate actions** — yfinance (`[live-data]`). Raw prices + splits/
  dividends so the as-of-T adjustment holds. Cached under
  `~/.swing_system/data_store`.
- **SEC filings & fundamentals** — EDGAR (needs a `User-Agent`, e.g. your email,
  on Settings): 8-Ks, Form-4 insider buys, 10-K/10-Q risk-factor text, and the
  XBRL `companyfacts` quarterly revenue/margin history that powers the moat
  trajectory.
- **Fundamentals snapshot** — yfinance: valuation multiples, growth, forward
  estimates, **estimate revisions**, margins/FCF, short interest, options chain,
  next earnings date.
- **News** — yfinance headlines (scored for tone).
- **Reddit** — optional buzz + LLM sentiment (needs free Reddit API creds).
- **Broker** — Alpaca (paper or live) for account, positions, orders, fills.

Without an Anthropic key, the agents run as a deterministic `MockLLMClient`
(coherent, grounded in the same numeric evidence) so the whole pipeline works for
free; with a key, the same prompts/schemas drive the real models (pinned,
temperature 0, structured output, prompt caching).

---

## Automation (run it daily, hands-off)

Headless CLI entry points (used by the packaged exe and Task Scheduler):

```bash
SwingSystem.exe --screen midsmall   # screen a universe headlessly
SwingSystem.exe --review            # manage the open book + score closes
SwingSystem.exe --watch             # check the watchlist, toast on a trigger
SwingSystem.exe --daily             # review exits, then screen (the scheduled job)
SwingSystem.exe --selftest          # verify the bundle (CI / post-build)
```

The **Settings → Automation** card schedules two weekday tasks via Windows Task
Scheduler: the daily run at 4:45 pm and watchlist checks at 12:30 + 3:30 pm
(which fire a Windows notification when a watched name reaches its entry
trigger). The learning loop only sharpens with reps, so daily operation is the
highest-value way to use it.

---

## Project layout

```
app/                     # the desktop application + live workflow
  main.py                #   entry point: GUI launch and the headless CLI flags
  gui.py                 #   the Tkinter app (Desk / Performance / Learning / Settings)
  runner.py              #   every flow the buttons call (screen, deep-dive, review, …)
  screen.py              #   the price-only pre-filter metrics + liquidity/regime
  strategy.py            #   composite score, presets, shortlist, portfolio, backtest
  reco_ledger.py         #   the recommendation ledger (record / score / cohorts)
  watchlist.py           #   persistent watchlist + entry-trigger logic
  config.py              #   AppConfig (saved to ~/.swing_system/config.json)
  momentum.py, gating.py, learning.py   # momentum tracker, edge gate, memory IO

system/                  # the agent + risk + execution engine (paper-default)
  agents/                #   analysts, the core trio, meta (guardian/reflection/curator)
  data_plane/            #   evidence assembly + indicators
  orchestrator.py        #   the deterministic deliberation driver (fail-to-PASS)
  confluence.py          #   cross-family confluence gating
  risk/                  #   Risk Governor (authoritative sizing), clusters
  execution/             #   PaperBroker + gated AlpacaBroker
  reflection/            #   lesson memory, calibration, curator
  monitoring/            #   health scorecard + kill switch (reduce-only)
  governance/            #   holdout vault + champion/challenger promotion (human gate)
  run_live.py            #   the paper-trading engine (used by the app + demos)

harness/                 # Phase-1 validation: prove an edge before trading it
  data/                  #   PIT store, loaders (yfinance/EDGAR), calendar, universes
  study/                 #   gap-aware costs, Newey-West/deflated-Sharpe stats, event study
  signals/               #   the free edges (filing, 8-K, insider, links, momentum)
  report/                #   per-edge scorecards + portfolio verdict
  run.py                 #   load -> signal -> study -> report

docs/                    # this documentation
build/                   # PyInstaller spec + OS build scripts + icon
tests/                   # the full suite (incl. the leakage CI test)
```

---

## Deeper docs

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — operating the app: every tab,
  every button, the daily workflow, reading the output, scheduling, settings.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how it all fits: the full
  pipeline, every module's role and key functions, the point-in-time data model,
  and the end-to-end flows.
- [`docs/SIGNALS.md`](docs/SIGNALS.md) — the signal & edge catalogue: what each
  one is, why it exists, how it's weighted, and the measured backtest deltas.
- `docs/design/` — the original master design and validation-harness spec the
  system was built from.

> Not investment advice. Trading involves risk of loss. Defaults are paper-only;
> any real-money use is your own deliberate, gated decision.
