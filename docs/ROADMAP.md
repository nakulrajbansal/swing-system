# Roadmap

Forward-looking plan for the swing-trading desk. The north star is a desk that
**earns the right to trade real cash** — measured, not asserted — by clearing the
deployment-readiness gate (`system/reflection/readiness.py`) while every
[non-negotiable invariant](../CLAUDE.md#non-negotiable-invariants-enforced-in-code-never-weaken)
holds. Everything below is ordered by how directly it moves that needle.

> This is the aspirational plan. The dated build log of what already shipped
> lives in git history and the design docs; this file is where we're going.

## Where we are (2026-07)
The full stack is built and in daily paper use: validation harness, agent/risk/
execution engine, desktop app, and an **automated learning loop** (curator +
cohort-gated lessons + self-tuning presets/gem-slots + de-risking throttle).
Recent hardening: stop-persistence (nested OCO legs), order-time stop-leg
verification on both entry paths, winner-extend at time-exit, and clean
class-share/no-data handling in the price loader.

**The desk is deliberately NOT cleared for real cash yet** — the readiness score
is gated on a longer, better-calibrated track record (PSR, MinTRL, Brier,
drawdown, diversity). Clearing those gates is the whole game.

## Now → next (highest leverage)
These directly improve the odds or the evidence quality the readiness gate reads.

- [ ] **Calibration quality** — Brier score is the current weakest gate. Feed the
  conviction-calibration engine more signal (per-lens, per-sector base rates;
  shrinkage tuned on the growing sample) so stated P(win) tracks realized odds.
- [x] **Auto stop-raising via broker order-replace** — the review now *acts* on a
  ≥1R winner: cancel+replace the OCO stop up to breakeven (one-shot, never lowers)
  under the asymmetric-autonomy invariant, gated by the managing flags. Advisory
  when managing is off. Next: trail beyond breakeven (e.g. to 1R) as R climbs.
- [ ] **Class-share data parity** — the broker normalises `MOG.A`↔`MOG-A`; give the
  *data loader* the same so class-share names are screenable where Yahoo serves
  them (genuine provider gaps like `CWEN-A` stay excluded, now logged cleanly).
- [ ] **Readiness explainability** — surface, per gate, exactly what would flip it
  green (e.g. "need 20 more scored trades", "cut maxDD below X"), so the path to
  deployment is a concrete checklist, not a single opaque score.

## Learning loop — keep it honest and self-pruning
The invariant the owner set: *if a lesson isn't good, it isn't a lesson* — fully
automated, no human gate on knowledge (only on money).

- [x] Anecdotes activate/retire on their lens cohort's realized avg return.
- [x] Pattern lessons recomputed each run; stale ones retire (replace-by-`kind`).
- [x] **LLM-synthesised lessons age out** — "re-earn or retire": refreshed when the
  synthesis re-derives them, retired after `LLM_TTL_DAYS` otherwise, hard-capped.
- [ ] **Regime tagging** — tag lessons/outcomes with the macro regime they were
  learned in so recall can weight by *today's* regime, not all history equally.
- [ ] **Lesson→parameter provenance** — show, on the Learning tab, which live
  parameter each active lesson is currently moving (closing the loop visibly).

## Execution & risk
- [ ] **Partial-fill & reconciliation robustness** — verify filled qty vs intended,
  reconcile the ledger to actual broker fills nightly (stale-fill guard exists;
  extend to a full nightly reconcile).
- [ ] **Portfolio-level exit logic** — correlated-drawdown de-risking across the
  book (cluster risk), not just per-name stops.
- [ ] **Headless scheduling parity** — the CLI (`--screen/--review/--daily`) exists;
  add a supervised scheduled-run mode with health checks + failure alerting so the
  desk can run unattended and *tell you* when something needs a human.

## Data & universe integrity
- [ ] **Historical index membership** — kill backtest survivorship by screening the
  point-in-time constituent set, not today's.
- [ ] **Intraday / better fills** — optional intraday bars for entry timing and more
  realistic fill modelling (still gap-aware, still costed).
- [ ] Paid positioning feeds (short interest, options flow) behind the existing
  opt-in gate.

## Longer-term / researchy
- [ ] Transcript-tone analyst (earnings-call sentiment) as a deterministic-first
  signal with an LLM path.
- [ ] Local-LLM path (Ollama) for cost-free agent runs (previously deferred).
- [ ] macOS scheduling parity; packaged installers.

## Explicitly out of scope (for now)
- Raising any risk/capital/live-trading limit automatically. **A human always
  flips live trading and sets money limits** — automation may only reduce risk.
- Anything that weakens point-in-time integrity, the two-keys rule, fail-to-PASS,
  or gap-aware/realistic-cost fills. These are load-bearing.

---
*Update this file when priorities shift; keep the dated "what shipped" record in
git history, not here.*
