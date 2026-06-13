# Signals & edges — the catalogue

Every signal the desk uses, why it exists, where it lives in the code, how it's
weighted, and — where it could be backtested — the measured effect. Signals fall
into two groups: **pre-filter factors** (price-only, scored over the whole
universe) and **deep-dive evidence** (fundamental/event signals the agents reason
over on the shortlist).

Honesty note: the price-only factors are backtestable on real data via the
walk-forward. The fundamental signals route through the agent panel and need
historical fundamentals to backtest (which the store doesn't keep), so they are
validated *forward* by the recommendation ledger — their documented direction is
cited, not a fabricated number.

---

## Pre-filter factors (`app/screen.py`, `app/strategy.py`)

Blended by `strategy.composite_score` using `BASE_WEIGHTS`, which are adapted to
the market regime and the desk's own realized results, and **self-tuned nightly**
across presets (`base` / `timing` / `discovery` / `defensive`) by
`strategy.select_preset`.

| Factor | Metric | Weight | Rationale |
|---|---|---|---|
| **Relative strength** | 6-mo return minus the benchmark's | 2.0 | The single most robust cross-sectional momentum signal — lead the market, not just rise with it. |
| **Momentum 6-mo** | `last / close[-126] - 1` | 1.0 | Trend strength. |
| **Momentum 3-mo** | `last / close[-63] - 1` | 0.5 | Shorter-horizon confirmation. |
| **Acceleration** | `mom3 - mom6/2` | 0.6 | The 3-month pace overtaking the 6-month average — an *igniting* trend that scores before it's consensus. Core of the hidden-gem lens. |
| **Trend** | above the 200-DMA (±1) | 0.30 | The long-term filter; risk-off if below. |
| **Trend quality** | 50-DMA above a *rising* 200-DMA (+1), or a daily pop with 50<200 (−1) | 0.35 | Multi-timeframe (weekly) confirmation — rewards a real intermediate uptrend, penalises a counter-trend bounce. |
| **Entry timing** | stretch above the 20-DMA (`ext20`) | 0.45 | Resting near the 20-DMA is buyable; >12% above is a chase that mean-reverts against a 2–20 day hold. |
| **Volume / accumulation** | up-day vs down-day volume; RVOL (today vs 20-day avg); volume expansion | 0.30 | Institutions leave volume footprints before the price move is obvious; unusual volume *now* is the earliest public ignition signal. |
| **Earnings-gap drift (PEAD)** | the largest recent 1-day gap, blended with drift, **recency-weighted** | 0.80 | Post-earnings drift is a documented anomaly; a fresh gap is a live catalyst, a stale one is spent. |
| **Near-high** | distance from the 52-week high | 0.20 | Strength near highs; falling-knife penalty far below. |
| **Sector rotation** | sector-ETF relative strength | 0.60 | Lean into leading sectors. |
| **RSI guardrails** | overbought/oversold penalties | — | Discount extreme overbought entries. |

**Hidden-gem lens** (`strategy.hidden_gem_score`, `gem_slot_count`). Reserved
shortlist slots for early-acceleration, pre-consensus names (igniting, still
below their highs, not yet a crowded 12-month story, volume-confirmed) that a
pure momentum ranking surfaces only after the rally is obvious. The slot count
self-tunes from the lens's realized hit rate.

**Data-quality & liquidity gates** (`screen.prescreen`). A single-session ±45%
discontinuity is treated as a split/vendor artifact and dropped (a large
*cumulative* move is a genuine leader and kept). Names below a median-dollar-volume
or price floor are dropped — essential once the universe includes small caps.

**Correlation de-duplication** (`strategy.correlation_diversify`). Near-identical
names (e.g. dual-class share pairs, or two names that move as one) are pushed down
the shortlist so the deep-dive and portfolio don't spend slots on the same bet
twice. **Measured** on a 515-name, 3-year S&P 500 walk-forward: total return
149.4% → 152.7%, Sharpe 1.66 → 1.69, max drawdown −26.5% → −25.7% (win rate
unchanged) — pure upside, robust to the threshold (it only removes >0.90-correlated
duplicates).

**Trend-quality (weekly) contribution**, same walk-forward: 152.7% → 155.3% total
return with Sharpe/drawdown/win-rate unchanged. Cumulative price-backtestable gain
vs the pre-Tier-1 original: **+3.9% relative return, +1.8% Sharpe, 0.8pp shallower
drawdown.**

---

## Deep-dive evidence (`system/data_plane/evidence.py`)

The shortlisted names get a full point-in-time evidence packet the six analysts
and the trio reason over.

| Block | Signals | Read by | Why |
|---|---|---|---|
| **technicals** | price, % from 52-wk high/low, 6-mo momentum, ATR%, RSI-14, 200-DMA | Technical | The setup for a long swing entry. |
| **valuation** | trailing/forward P/E, P/S, P/B, PEG, EV/EBITDA, target price | Valuation | Cheap/fair/expensive, **growth-adjusted** (a high P/E with PEG≤1.5 is a compounder, not "expensive"). |
| **growth** | revenue/earnings growth, forward-EPS guidance, ROE, margin, **estimate revisions**, analyst coverage | Growth | Trajectory + whether guidance is accelerating. |
| **moat** | gross/operating/FCF margins, ROA, market cap, insider ownership, **business summary** | Moat | Durable advantage + which secular trend the business is levered to. |
| **trajectory** | quarterly revenue YoY + margin trend from EDGAR XBRL, accel/decel flags | Moat | Revenue *accelerating* while margins *expand* = the pre-rally inflection — the "find the next leader" fingerprint a snapshot can't show. |
| **earnings_quality** | profit margin vs FCF margin (the accrual gap) | Skeptic, Fundamental | Earnings that don't convert to cash = an accrual red flag (value-trap / aggressive-accounting risk). Strictly downside-protective. |
| **events** | next earnings date, days-to-earnings, in-window flag | Skeptic, Strategist | A binary print inside the hold window is event risk — exit before it or accept it explicitly. |
| **positioning** | short % of float, days-to-cover, options call/put volume & OI ratios | Skeptic, Fundamental | A free proxy for where money is already committed — read two-sided (short interest is a bear case *and* squeeze fuel). |
| **filings & insider** | 10-K/10-Q risk-factor text change, 8-K cadence, Form-4 open-market buys | Fundamental | Adverse disclosures vs boilerplate; informed insider buying. |
| **news + tone** | recent headlines + a lexicon tone score, bearish items flagged | Fundamental, **Guardian** | Bullish flow corroborates a catalyst; decisively bearish news is a Guardian exit trigger on an open position. |

### Estimate-revision momentum (the headline new fundamental edge)
`loader.fetch_fundamentals_yahoo` reads yfinance `eps_trend` to compute the
forward-EPS consensus drift over 30/90 days, gated on analyst coverage (≥4).
`GrowthAnalyst` weights it ±0.14. Upward revisions are a documented *leading*
signal — analysts revise toward an improving reality and price drifts after them;
it is precisely the "analysts racing to upgrade an emerging leader before the
crowd" pattern. Live-verified examples at build time: NVDA +17.9%, AMD +21.8%
over 90 days. Not price-backtestable (no historical estimates in the store);
validated forward by the ledger.

---

## Confidence & risk meta-signals

These don't pick names — they govern *how much to trust and size* a pick.

- **Calibration / P(win)** (`system/reflection/calibration.py`). Maps the desk's
  stated conviction to its own realized win rate per band, with empirical-Bayes
  shrinkage so a thin record returns the stated conviction (labelled
  "uncalibrated"), and a seasoned record returns the measured rate. Shown on
  every recommendation.
- **Self-throttle** (`calibration.desk_throttle`). On a cold streak (recent hit
  rate < 40%) or hot convictions (gap ≥ 25pp vs realized), it raises the BUY bar
  and caps gross exposure (0.6–0.75×). Dormant below 8 scored calls; strictly
  risk-reducing. This is the "shut down if it has no edge" principle, keyed off
  realized results.
- **Risk Governor** (`system/risk/governor.py`). The authoritative sizer: 1%
  account risk, ATR-based stop, per-name (≤25%) and per-sector caps, sized off
  the real account. The second of the "two keys".
- **Macro backdrop** (`harness/data/macro.py`, `MacroAnalyst`). One market-wide
  read (equity regime, VIX, rates, credit, USD, cyclical-vs-defensive) that sets
  the risk weather for every name that day; a hostile backdrop forces risk-off.

---

## Phase-1 edges (the validation harness, `harness/signals/`)

Separate from the live screen: these are *statistically validated* edges the
harness can PASS or KILL before they're allowed to trade live (the "Only trade
validated edges" gate). Each is an event study with gap-aware costs and
Newey-West / deflated-Sharpe correction.

| Edge | Signal | File |
|---|---|---|
| Edge 1 | Filing-text change predicts abnormal returns | `edge01_filing.py` |
| Edge 2 | 8-K event reaction | `edge02_8k.py` |
| Edge 6 | Insider cluster buying | `edge06_insider.py` |
| Edge 7 | Economic-link read-through | `edge07_links.py` |
| Edge 8 | Price momentum | `edge08_momo.py` |

`harness.run` recovers a planted Edge-1 signal (PASS) and KILLs noise edges,
applying a portfolio-level deflated-Sharpe multiple-testing correction — the
discipline that keeps a backtest honest.

---

## How to extend

- A **new pre-filter factor**: add the metric in `screen._metrics`, a weight in
  `strategy.BASE_WEIGHTS`, a term in `composite_score`, and measure it in the
  walk-forward before trusting it.
- A **new evidence signal**: add it to `evidence.assemble_evidence`, teach the
  relevant analyst (deterministic logic + prompt), surface it in the deep-dive
  scorecard, and let the ledger validate it forward.
- A **new validated edge**: implement `Signal` in `harness/signals/`, register it,
  and run the harness — it must PASS before the live gate lets it trade.
