# Automation — running the desk on a schedule

The desk can run itself on a schedule, either **locally** (Windows Task
Scheduler) or **in the cloud** (GitHub Actions — runs even when your PC is off).
Both share the same timing logic and the same headless commands.

- [The timing rationale](#the-timing-rationale)
- [The schedule presets](#the-schedule-presets)
- [Local scheduling (Windows)](#local-scheduling-windows)
- [Cloud scheduling (GitHub Actions)](#cloud-scheduling-github-actions)
- [Manage-exits-only safety](#manage-exits-only-safety)
- [Headless commands](#headless-commands)

---

## The timing rationale

This is a 2–20 day **swing** strategy on **daily bars**, so nothing is
intraday-critical — but each job has a different optimal time relative to the
9:30 AM – 4:00 PM ET session:

| Job | Best time (ET) | Why |
|---|---|---|
| **Review exits** | ~10:00 (≈30 min after open) | Arm protective stops *early*; fire time-exits at live, liquid prices instead of next-day gaps; act on overnight news after it's digested; avoid the opening-auction chaos. |
| **Watchlist** | 11:00 / 13:00 / 15:00 | Entry triggers (a pullback into your window, a breakout on volume) happen *during* the session — the only job that benefits from intraday checks. |
| **Screen** | ~17:00 (after close) | Ranking needs *final* daily bars (the close isn't final until 4:00 PM). Review tickets in the evening, place at the next open — disciplined end-of-day swing. |

The old "everything at 4:45 PM" was suboptimal: it queued time-exits for the
*next* open (extra overnight gap risk) and screened on not-yet-final bars. The
presets below fix that.

---

## The schedule presets

Defined once in `app/schedule.py` (ET times), used by both the local scheduler
and the cloud workflow:

| Preset | Review | Watch | Screen |
|---|---|---|---|
| **eod_swing** (recommended) | 10:00 | 11:00, 13:00, 15:00 | 17:00 |
| **morning** (hands-off) | — | 12:00 | combined `daily` run at 08:00 |
| **active** | 10:00 | 10:30, 12:00, 13:30, 15:00 | 17:00 |
| **custom** | your `custom_review_et` | your `custom_watch_et` | your `custom_screen_et` |

The **screen universe(s)** are configurable — a comma list like
`sp500,midsmall` — so a scheduled screen can cover one or several indices.

---

## Local scheduling (Windows)

Settings → **Automation**:
1. Pick a **Preset** and the **Screen universes** (comma list).
2. **Apply schedule** — creates one Windows Task per job/time (ET converted to
   your PC's local clock), named `SwingSystem <job> <time>ET`. **Remove schedule**
   deletes them all.

Requires the packaged `dist\SwingSystem.exe` (the tasks run it headlessly).
Caveat: the local schedule only runs **when your PC is on** — for always-on, use
the cloud below. (macOS: use `launchd`/`cron` with the [headless
commands](#headless-commands).)

---

## Cloud scheduling (GitHub Actions)

Runs on GitHub's servers on a weekday schedule — **no PC required**, free for a
private repo. State (ledger / learning / watchlist) persists across runs on a
dedicated `swing-state` branch; logs are uploaded as run artifacts. The workflow
is `.github/workflows/swing-desk.yml`.

**One-time setup:**
1. **Make the repo private.** The ledger reveals your positions, so it must not
   be public.
2. **Add secrets** — repo → Settings → Secrets and variables → Actions →
   *Secrets*:
   - `ANTHROPIC_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET`, `EDGAR_USER_AGENT`.
3. **(Optional) choose universes** — same page, *Variables* tab: add
   `SWING_SCREEN_UNIVERSES` = e.g. `sp500,midsmall`. (Default `sp500`.)
4. **Enable workflows** on the Actions tab. Use **Run workflow** (manual
   dispatch) to test — pick a job and, for a screen, an optional universe list.

**Schedule (UTC crons in the workflow; ET assumes EDT = UTC−4):**
- `0 14 * * 1-5` → review 10:00 ET
- `0 15 / 0 17 / 0 19 * * 1-5` → watch 11:00 / 13:00 / 15:00 ET
- `0 21 * * 1-5` → screen 17:00 ET

> **DST note:** crons can't follow daylight saving, so in winter (EST) these
> fire one hour later in ET. For a swing strategy that's immaterial (the review
> is reduce-only and idempotent); shift the crons +1h if you want exact ET.

**What runs:** `SWING_PLACE_ORDERS=false` and `SWING_AUTO_MANAGE_EXITS=true`, so
the cloud desk manages exits (reduce-only) and produces recommendations but
**never auto-buys** — see below.

The workflow uses `SWING_HOME` to keep state in the workspace and the env-var
overlay (`AppConfig` reads `ANTHROPIC_API_KEY`, `ALPACA_ENV`, `SWING_*`, …) so it
needs **no committed config file** — secrets stay in GitHub, never in the repo.

---

## Manage-exits-only safety

Unattended runs follow the asymmetric-autonomy rule — they may only **reduce**
risk:
- **Review** arms missing protective stops, fires time-exits, and (with
  `auto_manage_exits`) executes Guardian exits on thesis-breaking news. All
  reduce-only.
- **Watch** only sends notifications / logs triggers.
- **Screen** ranks and produces **recommendations + a suggested portfolio**, and
  logs them to the ledger — but does **not** place buy orders unless you
  explicitly set `SWING_PLACE_ORDERS=true` (cloud) or *Place approved orders*
  (local). The default is advise-only: you review the run's log and place
  tickets yourself when you're back.

To go fully hands-off (auto-place new buys too), set `SWING_PLACE_ORDERS=true` —
and note it will then open positions while you're away, sized by the Risk
Governor. Real-money still additionally requires `ALPACA_ENV=live` + the live
gate.

---

## Headless commands

The same entry points the schedulers call (the exe, or `python -m app.main`):

```bash
--review                 # manage the open book (reduce-only)
--watch                  # check the watchlist, notify on triggers
--screen [a,b]           # screen the given universe(s), or the configured default
--daily [a,b]            # review, then screen the universe(s)
--selftest               # verify the bundle (CI / post-build)
```

Environment overrides (used by the cloud, or any server/cron): `SWING_HOME`
(state dir), `ANTHROPIC_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET`, `ALPACA_ENV`,
`EDGAR_USER_AGENT`, `SWING_DATA_SOURCE`, `SWING_USE_LLM`, `SWING_PLACE_ORDERS`,
`SWING_AUTO_MANAGE_EXITS`, `SWING_SCREEN_UNIVERSES`.
