# User Guide — operating the Swing System desktop app

This guide assumes you have the app open (`python -m app.main`, or
`dist\SwingSystem.exe`). It walks through every tab, every button, what each run
produces, and the daily workflow. No prior knowledge of the codebase is needed.

If this is your first launch, everything works immediately in **synthetic +
deterministic** mode (a demo universe, no network, no keys). To use real markets,
fill in the keys on **Settings** first ([Settings reference](#settings-reference)).

---

## Contents

1. [The window at a glance](#the-window-at-a-glance)
2. [First-time setup](#first-time-setup)
3. [Desk tab — finding opportunities](#desk-tab--finding-opportunities)
4. [Reading a screen's output](#reading-a-screens-output)
5. [Placing orders](#placing-orders)
6. [Managing the open book (Review exits)](#managing-the-open-book-review-exits)
7. [The Watchlist](#the-watchlist)
8. [Performance tab](#performance-tab)
9. [Learning tab](#learning-tab)
10. [Settings reference](#settings-reference)
11. [Automation & scheduling](#automation--scheduling)
12. [Keyboard shortcuts](#keyboard-shortcuts)
13. [Where files live](#where-files-live)
14. [The recommended daily workflow](#the-recommended-daily-workflow)
15. [Troubleshooting](#troubleshooting)

---

## The window at a glance

- **Left sidebar** — switches pages: **Desk**, **Performance**, **Learning**,
  **Settings**. The bottom shows a **PAPER TRADING** badge (turns into a red
  **⚠ LIVE — REAL MONEY** when live is enabled) and a live status line with the
  running task and elapsed time.
- **Page header** — on the Desk, three context chips tell you exactly what the
  next run will use: `DATA · LIVE/SYNTHETIC`, `AGENTS · LLM/DETERMINISTIC`,
  `SIZING · REAL ACCOUNT/CONFIGURED`. If a chip isn't what you expect, fix it on
  Settings before running.

---

## First-time setup

Everything is optional — add only what you want to switch on.

| To enable… | Set on Settings | Effect |
|---|---|---|
| Real market data | `Data source = live` + EDGAR User-Agent | Real prices, SEC filings, news |
| Real AI agents | Anthropic API key + `Use LLM agents` | Real reasoning instead of the deterministic mock |
| A broker | Alpaca key id + secret (`paper` env) | Sizing off real equity, place orders, manage exits |
| Reddit buzz | Reddit client id/secret (+ user/pass) | Sentiment scan |

Then **Save configuration** (`Ctrl+S`). The Desk chips update to confirm.

> **Real money** requires three separate switches: Alpaca env = `live`, **Enable
> LIVE trading**, and **Place approved orders**. Until all three are on, nothing
> can touch a live account.

---

## Desk tab — finding opportunities

**Find opportunities** card:
- **Screen** — pick a universe from the dropdown (S&P 500, Nasdaq-100, S&P 400
  mid-cap, S&P 600 small-cap, *Mid + small caps — hidden gems*, or S&P 1500
  broad), then **Run screen** (`Ctrl+R`). The screen pre-filters the whole index
  for free, then deep-dives only the top few with the agent panel. First run of
  the day downloads prices (slow once, then cached).
- **Analyze** — type any US ticker and **Deep-dive** to run the full agent panel
  on just that name, even one outside the index.

**Trade & monitor** card:
- **Review exits** — manage your open positions (see
  [below](#managing-the-open-book-review-exits)).
- **Portfolio P&L** — read your Alpaca account: equity, day change, every
  position's unrealized P&L, leverage/concentration warnings, and open orders.
- **Momentum auto-trade** — a *separate, mechanical* one-name strategy (no
  agents): it exits any tracked momentum position that hit its hold date, then
  buys the single strongest-momentum name in your screen universe with a
  protective stop. The baseline the agent flow aims to beat.
- **Strategy backtest** — an honest walk-forward of the pre-filter strategy vs
  the benchmark (no look-ahead, with a survivorship-bias caveat).
- **Reddit sentiment** — buzz + LLM sentiment on the most-mentioned tickers.

---

## Reading a screen's output

The console streams the full run. Use **Jump to section** (top-left of Output) to
navigate a long run instead of scrolling. Key markers:

- `[account]` — whether sizing used your real account or the configured equity.
- `[tuner]` — which factor-weight preset is winning the trailing walk-forward.
- `[throttle]` — present only if the desk is de-risking after a cold streak.
- `[regime]` / `[sectors]` — market backdrop and sector leadership.
- `[leaderboard]` — the top pre-filter names with their scores and factor columns
  (`vs20d` = entry stretch above the 20-DMA; `RVOL` = today's volume vs its
  20-day average).
- `[deep-dive]` — the shortlist; `(gem)` marks a hidden-gem slot.
- `# DEEP-DIVE: SYM` — per-name: the evidence packet, then every agent's reasoning
  in full (analyst stances, the Strategist's thesis, the Skeptic's objections, any
  round-2 rejoinder, the PM's decision), then a readable scorecard
  (TECHNICAL / VALUATION & GROWTH / MOAT & FUTURE GROWTH / FILINGS & INSIDER /
  SCHEDULED EVENTS / NEWS TONE / ANALYST READS / WHY THIS VERDICT / TRADE PLAN).
- `RECOMMEND BUY` — the final picks with conviction, **P(win)**, plan
  (entry/stop/target/hold), and — when the PM advised waiting — a pullback price.
- `SUGGESTED PORTFOLIO` — conviction-weighted, capped, account-sized allocations.
- `TOP IDEAS` — every shortlisted name ranked and tiered BUY / WATCH / PASS, so an
  all-PASS run is still actionable.

Recommendations are also saved to the **ledger** for forward scoring, and
WATCH-tier names go to the **Watchlist**.

---

## Placing orders

After a screen or deep-dive, BUY recommendations appear in the **Order tickets**
table: symbol, conviction, P(win), reference price, an editable **qty**, order
**type** (market/limit), **limit price**, a **stop/target** bracket toggle, and
**Place order**. Markers: ◆ = hidden-gem pick, ● = a name you already hold
(placing adds to it). The limit pre-fills to the PM's pullback entry when it
advised waiting for a dip.

Placing checks your buying power first and refuses (with the max affordable
quantity) rather than bouncing at the broker. Each row confirms with
**✓ Sent** / **✗ Failed**. The order is linked to its recommendation, so it shows
as *executed* on the Learning tab.

---

## Managing the open book (Review exits)

**Review exits** is the other half of the trade lifecycle — run it daily. For
every broker position it:

1. **Reports** the position against the **plan** that created it (the ledger
   entry: entry / stop / target / exit-by / days held).
2. **Arms missing protection** — if no stop/target sell orders are actually
   resting at the broker, it places them (an OCO pair, or a stop), because that
   only *reduces* risk. Existing protection is reported, not duplicated.
3. **Time-exits** — closes at market any position held past its planned exit date
   (the trade's own rule).
4. **Runs the Guardian** on the rest — it recommends HOLD or EXIT based on
   thesis-breaking developments (broken trend, contradicting news, earnings
   about to hit). EXIT auto-closes only when **Place orders** is on; otherwise
   it's advisory. Drawdown within the plan's stop is *not* a reason to exit.
5. **Coaches winners** — at +1R unrealized, suggests raising the stop to
   breakeven so the winner runs on the market's money.
6. **Scores closes** — any position sold at the broker (stop/target filled, or
   manually) is scored immediately from the actual fills and fed to the learning
   loop, and matured recommendations are graded — so the desk learns the same
   day, even if you never run a screen.

It opens with an **account-level summary** (equity, gross exposure) and a `[RISK]`
warning if the book is levered above 1.0×.

---

## The Watchlist

Names whose *time* hasn't come: WATCH-tier ideas (0.40–0.55 conviction) and the
PM's "wait for a pullback to ~$X" calls. Each carries a concrete trigger (a
pullback price, or a breakout level + volume). **Check triggers now** (or the
scheduled `--watch` job) fetches just those names and fires a Windows
notification the moment one enters its entry window or breaks out on unusual
volume — then hands it back for a fresh deep-dive at *today's* price. Entries
expire after ~3 weeks so the list stays a watchlist, not a graveyard.

---

## Performance tab

Your own track record, nothing borrowed:

- **Equity curve** — cumulative return of scored recommendations, in order.
- **Scorecard** — overall hit rate / average return / excess vs SPY; the
  **calibration table** (per conviction band: how often did it actually win?);
  **lens cohorts** (hidden-gem vs core vs moat-bullish); recent closed calls.
- **Executed trades** — your account's actual fills (side / qty / symbol / price /
  time). Press **Refresh trade history** to pull them from the broker.

This page is sparse until trades mature, then becomes the most honest view in the
app — it's measured, not modeled.

---

## Learning tab

The desk's accumulated **lessons + base rates** and the recommendation ledger
(open entries marked ● executed / ○ advisory). **Run AI curator** audits the
record: it activates a drafted lesson only when the realized aggregate for its
setup supports it, retires anecdotes the record contradicts, and writes pattern
lessons (e.g. "convictions running hot", "hidden-gem picks are paying") once
there are enough scored calls. **Clear all** wipes the learning memory.

There is **no manual approve step** — authority is earned through realized
evidence. (You can still bypass the curator's gate with the *Auto-activate*
toggle on Settings, but it's not recommended until the ledger has volume.)

---

## Settings reference

**API keys & credentials** — Anthropic API key, Alpaca key id/secret + env
(`paper`/`live`), EDGAR User-Agent (any contact string, e.g. your email — needed
for SEC data), Reddit client id/secret/username/password.

**Data & universe** — `Data source` (`synthetic` demo / `live` real), a single
**Analyze one ticker** override, **Synthetic universe size** (offline demo only),
synthetic date range/seed, starting equity (the fallback when no broker is
connected), out-of-sample start.

**Strategy parameters** — insider/filing history depth, momentum hold/positions,
Reddit top-K, **Screen deep-dive top K** (how many names get the agent panel),
**Screen universe cap** (0 = full index), **Screen index** (default universe).

**Options (toggles)** —
- *Use LLM agents* — real models vs the deterministic mock.
- *Only trade validated edges* — live deliberation trades only edges that passed
  the harness.
- *Self-tune screen weights nightly* — the preset tuner.
- *Show full agent reasoning* — verbose transcripts.
- *Place approved orders on Alpaca* — actually submit (else proposals only).
- *Learn from runs* — accumulate lessons + recall them.
- *Auto-activate new lessons* — skip the curator's evidence gate.
- *Enable LIVE (real-money)* — the asymmetric-autonomy gate for live trading.

**Automation** — schedule/remove the daily run + watchlist checks.
**Save configuration** (`Ctrl+S`) · **Test Alpaca connection**.

---

## Automation & scheduling

Settings → Automation → **Schedule daily run** creates two weekday Windows tasks
pointing at the packaged exe:
- **4:45 pm** — `--daily`: review exits, then screen your saved universe.
- **12:30 & 3:30 pm** — `--watch`: check the watchlist, notify on a trigger.

A status line shows whether they're scheduled; **Remove schedule** deletes both.
(macOS: use launchd/cron with the same flags.) Because the 4:45 pm review runs
after the close, any time-exit it triggers queues as a market order for the next
open — fine for a multi-session swing plan.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+R` | Run the selected screen |
| `Esc` | Stop the running task |
| `Ctrl+S` | Save settings |
| `Enter` (in the ticker box) | Deep-dive that ticker |

A running task can also be stopped with the **Stop run** button; it halts at the
next step, never mid-order, and notes where its partial log was saved.

---

## Where files live

Everything is local, under `~/.swing_system/`:

- `config.json` — your saved settings and keys (owner-only; never committed).
- `data_store/` — cached prices, fundamentals, and per-day pre-filter caches.
- `logs/` — a timestamped log of every run (also shown in the console).
- `recommendations.json` — the ledger.
- `learning_memory.json` — lessons + outcomes.
- `watchlist.json`, `momentum_positions.json`, `trades.json` — supporting state.

Delete a `data_store` cache folder to force a fresh download.

---

## The recommended daily workflow

1. **Review exits** — manage the book, arm stops, score any closes.
2. **Run screen** on your universe — read the recommendations and TOP IDEAS.
3. **Place** the tickets you like (small size; the buying-power check protects
   you).
4. Glance at **Performance** as it fills, and let **Run AI curator** (or the
   automatic post-scoring pass) keep the lessons honest.
5. Better still: **schedule it** and let `--daily` do 1–2 for you, then just
   review the morning's log and place tickets.

The learning systems (calibration, curator, self-tune, throttle) only sharpen
with reps. Running daily for a few weeks is what turns the honest-but-uncalibrated
fresh install into a desk that knows its own edge.

---

## Troubleshooting

- **Chips say SYNTHETIC / DETERMINISTIC / CONFIGURED** — you haven't set
  `Data source = live`, a working Anthropic key + *Use LLM agents*, or Alpaca
  keys. Set them on Settings and Save.
- **"no EDGAR User-Agent"** — set any contact string on Settings; SEC requires it
  for filings/fundamentals.
- **Orders bounce** — the buying-power pre-check shows the max affordable qty;
  open (unfilled) orders also reserve funds — cancel stale ones from Portfolio
  P&L or your Alpaca dashboard.
- **A screen is slow** — the first run per universe per day downloads the whole
  index; it's cached afterwards.
- **Building the exe fails with "Access is denied"** — the app is open; close it
  (the build replaces the running file).
