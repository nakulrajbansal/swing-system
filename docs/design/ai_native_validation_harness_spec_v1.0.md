# Validation Harness Build Spec
### AI-Native Multi-Agent Swing Trading System (the first real build)

| | |
|---|---|
| **Version** | 1.0 |
| **Status** | Build-ready |
| **Scope** | One reusable harness that validates the free edges (1, 2, 6, 7, 8) on free data via a shared event-study engine. Closes the Bucket 1 design items. |
| **Parent** | `ai_native_MASTER_design_v1.0.md` |

> This harness does **not** trade, call any LLM, or run agents. It answers one question per edge: does the signal predict abnormal returns, net of costs, out-of-sample? Deterministic signals are tested here (no look-ahead, free). LLM-lift testing comes later (forward/post-cutoff), per the master doc.

---

## 1. What this resolves (Bucket 1)

1. **Universe and period** (Section 3).
2. **Point-in-time data store schema** (Section 4).
3. **Benchmark/control choice** for abnormal returns (Section 6).
4. **Edge-signal plug-in interface** (Section 5).
5. **Statistical choices and pass/kill** (Sections 6 and 7).

---

## 2. Module architecture

```
harness/
  data/
    loader.py        # pull EDGAR filings + free daily prices
    pit_store.py     # point-in-time store (read/write, as-of queries)
    corp_actions.py  # split/dividend as-of adjustment
  signals/
    base.py          # the EdgeSignal interface (Section 5)
    edge01_filing.py # filing-text change (deterministic)
    edge02_8k.py     # 8-K event (deterministic baseline)
    edge06_insider.py# Form 4 cluster buying
    edge07_links.py  # economic-link read-through (deterministic baseline)
    edge08_momo.py   # price/52-week-high momentum
  study/
    event_study.py   # forward returns, abnormal-return adjustment
    stats.py         # quintiles, spreads, Newey-West, deflated Sharpe
    costs.py         # commission + spread + slippage, gap-aware
  report/
    report.py        # per-edge scorecard + plots
  run.py             # orchestrates: load -> signal -> study -> report
```

Everything is local and deterministic. No network calls except the data loader (EDGAR + free price API), which caches to the PIT store so runs are reproducible.

---

## 3. Universe and period (decisions)

- **Universe:** US common stocks and liquid ETFs with, as of each test date T: price >= $5, 20-day median dollar volume >= $5M, and >= 250 sessions of history. Membership is **point-in-time** (include names later delisted; pull a historical constituent list or reconstruct from a delisting-inclusive price source). Start with a tractable set (e.g., the ~1,000 most liquid names) to keep EDGAR pulls and compute manageable.
- **Period:** at least 8 years of history for walk-forward, with the **last 3 years reserved as the out-of-sample window** that must independently pass. A further segregated **holdout slice** (e.g., the most recent 6 to 12 months) is touched at most once per edge.
- **Frequency:** daily bars.

## 4. Point-in-time data store schema

Local store (parquet or SQLite); the key discipline is that every row is queryable *as of* a timestamp.

```
prices(symbol, date, open, high, low, close, volume, adj_factor_asof)
corp_actions(symbol, ex_date, type, ratio_or_amount)
constituents(symbol, start_date, end_date)         # PIT membership
filings(symbol, cik, form_type, available_at, accession, doc_uri,
        section_text_riskfactors, section_text_mdna)  # for edges 1,2
form4(symbol, cik, available_at, insider_role, txn_code, shares, value)  # edge 6
links(focal_symbol, linked_symbol, link_type, source_filing)            # edge 7
news(symbol, available_at, headline, body_uri, source)                  # edge 7 trigger
```

Rules: `available_at` (not event time) governs visibility; an `as_of(T)` query returns only rows with `available_at <= T` and applies only corporate actions with `ex_date <= T`. A CI test asserts no future leakage.

## 5. The edge-signal plug-in interface

Every edge implements the same interface so one study engine serves all of them.

```python
class EdgeSignal(Protocol):
    edge_id: str
    family: str          # "A","B","C","D","E"
    direction: str       # "long" | "short" | "avoid"

    def triggers(self, store, date) -> list[str]:
        """Symbols whose trigger fired as of `date` (e.g., a new filing)."""

    def score(self, store, symbol, date) -> dict:
        """Return {raw_score: float, confidence: float, evidence: dict}
           computed using ONLY store.as_of(date). Higher raw_score = stronger signal."""
```

The study engine calls `triggers()` to find event dates, `score()` to rank, then measures forward returns. This same interface is later reused by the live specialist agents (they implement `score()` with an LLM), so the harness and the live system share one contract.

## 6. Event-study engine and benchmark choice

- **Entry:** t+1 open (per the master's two-stage model), with gap-aware fills and costs.
- **Forward windows:** [t+1, t+5], [t+1, t+10], [t+1, t+20] trading days.
- **Abnormal return (decision):** **sector-ETF-adjusted** by default (subtract the matched sector ETF's return over the same window). Rationale: simple, robust, avoids the survivorship and estimation issues of matched-control baskets; market-adjusted is the fallback if a clean sector map is unavailable.
- **Bucketing:** sort triggered names into quintiles by `raw_score` each period; compute mean abnormal return per quintile and the top-minus-bottom spread.
- **Overlapping returns:** use non-overlapping samples where possible, else Newey-West standard errors with lag = window length.

## 7. Costs and pass/kill

- **Costs (`costs.py`):** commission + half-spread per side + slippage; gap-aware (fill at the open when gapped through). Apply to every simulated entry/exit.
- **Pass/kill per edge (illustrative, tunable):**
  - Monotonic ordering of quintile abnormal returns with `raw_score`.
  - Top-minus-bottom 20-day spread positive, clearly above round-trip costs, with t-stat > 2.5 (proper standard errors).
  - Holds in the **last 3 years** specifically and in subperiods; not driven by one sector or a few names.
  - Long-only tradable form (long the favorable tail or use as an avoid-filter) survives net of costs.
- **Portfolio-level:** report a **Deflated Sharpe** across the edges tested to account for multiple comparisons; an edge that only clears the naive bar does not pass.
- **Kill:** no monotonic, cost-surviving, OOS spread → archive the edge with its result.

## 8. Outputs

Per edge, a scorecard: quintile abnormal-return table, top-minus-bottom spread with t-stat, OOS vs full-sample comparison, subperiod and sector breakdowns, equity curve of the long-only tradable form net of costs, and a clear PASS / KILL verdict. A portfolio summary lists which edges passed and feeds the live confluence engine's initial membership.

## 9. Explicitly out of scope (do not build yet)

No agents, no LLM calls, no live data, no execution, no orchestrator, no confluence engine, no dashboard. Those come only after at least one edge passes here. The deterministic-stage results gate everything downstream.

## 10. First experiment

Implement the harness, then run **Edge 1 (filing-text change)** end to end as the first edge, because it is free, leak-free, and the purest test of the AI-native premise's foundation. Then run Edges 2, 6, 7, 8 through the same harness. The set of passing edges is the project's first real evidence and determines whether to proceed to the agent build.

---

### Build order within this spec
1. `pit_store.py` + `corp_actions.py` + the leakage CI test (the integrity foundation).
2. `data/loader.py` for prices and EDGAR (filings, Form 4).
3. `study/event_study.py` + `stats.py` + `costs.py`.
4. `signals/edge01_filing.py`, validate end to end.
5. Remaining free-edge signals (2, 6, 7, 8).
6. `report/report.py` and the portfolio summary.
