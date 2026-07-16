"""The Lesson Curator: AI-native review of the desk's own track record.

The Reflection agent drafts one lesson per closed trade — useful but anecdotal.
The curator is the self-assessment loop the platform owner asked for: it
compares what the desk RECOMMENDED (conviction, lens, plan) against what
actually HAPPENED (scored ledger outcomes), and

  * ACTIVATES pending lessons that the aggregate evidence supports,
  * RETIRES anecdotes the aggregate evidence contradicts,
  * WRITES pattern lessons no single trade can show — conviction calibration
    drift, lens performance (hidden-gem / moat), and exit-mix problems —
    each gated on a minimum number of scored outcomes (n >= 5) so the desk
    never teaches itself from noise.

Activated lessons are marked reviewed by the curator itself (no human gate):
self-authored authority is earned through realized evidence, not sign-off.
"""

from __future__ import annotations

import datetime as _dt

from system.schemas import Lesson

MIN_BUCKET = 5      # a pattern needs at least this many scored outcomes
MIN_RETIRE = 8      # contradicting an anecdote needs even more evidence
LLM_TTL_DAYS = 21   # an LLM lesson the synthesis stops re-deriving retires after this
LLM_MAX = 6         # hard cap on concurrent LLM lessons (the freshest are kept)


def _parse_day(s):
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _days_stale(as_of: str, run_as_of: str):
    """Days between an LLM lesson's last re-derivation (`as_of`) and the current
    run (`run_as_of`), or None when either date is unparseable — in which case we
    never age the lesson out blindly."""
    a, b = _parse_day(as_of), _parse_day(run_as_of)
    return (b - a).days if (a is not None and b is not None) else None


def _bucket(rows: list[dict]) -> dict:
    rets = [r["return_pct"] for r in rows
            if isinstance(r.get("return_pct"), (int, float))]
    if not rets:
        return {"n": 0}
    wins = sum(1 for x in rets if x > 0)
    convs = [r["conviction"] for r in rows
             if isinstance(r.get("conviction"), (int, float))]
    stops = sum(1 for r in rows if r.get("outcome") == "stop")
    return {"n": len(rets),
            "win_rate_pct": round(100.0 * wins / len(rets), 0),
            "avg_return_pct": round(sum(rets) / len(rets), 2),
            "avg_conviction": round(sum(convs) / len(convs), 2) if convs else None,
            "stop_rate_pct": round(100.0 * stops / len(rets), 0)}


def _win_rate(rows: list[dict]):
    rets = [r["return_pct"] for r in rows
            if isinstance(r.get("return_pct"), (int, float))]
    if not rets:
        return None, 0
    return round(100.0 * sum(1 for x in rets if x > 0) / len(rets), 0), len(rets)


def _conviction_inversion(scored: list[dict]) -> tuple | None:
    """Bucket-level calibration: split the scored calls at their MEDIAN
    conviction and compare win rates. If the higher-conviction half wins
    materially LESS than the lower half (with enough samples on each side),
    conviction is inversely related to outcome — the desk is most wrong exactly
    when it bets biggest. The aggregate conviction-vs-winrate gate misses this
    because the averages can net out; the split surfaces it. Returns
    (high_wr, low_wr, n_high, n_low) when inverted, else None."""
    convs = sorted(r["conviction"] for r in scored
                   if isinstance(r.get("conviction"), (int, float)))
    if len(convs) < 2 * MIN_BUCKET:
        return None
    med = convs[len(convs) // 2]
    low = [r for r in scored if isinstance(r.get("conviction"), (int, float))
           and r["conviction"] < med]
    high = [r for r in scored if isinstance(r.get("conviction"), (int, float))
            and r["conviction"] >= med]
    low_wr, n_low = _win_rate(low)
    high_wr, n_high = _win_rate(high)
    if n_low < MIN_BUCKET or n_high < MIN_BUCKET:
        return None
    if low_wr is None or high_wr is None or (low_wr - high_wr) < 15:
        return None
    return high_wr, low_wr, n_high, n_low


def assess(ledger_rows: list[dict]) -> dict:
    """Expectation vs realization, by lens and overall — plus the pattern
    lessons the evidence currently supports. Pure and deterministic."""
    scored = [r for r in ledger_rows if r.get("status") == "evaluated"]
    # MUTUALLY-EXCLUSIVE lenses (gem > moat-bullish > core), so a lens lesson
    # judges only that lens: without this, a bullish-moat pick that is ALSO a gem
    # counts in both, and the gem drag makes 'moat-bullish' look worse than the
    # non-gem moat picks actually are (they ran +1.7% while gems ran -2.8%).
    stats = {
        "overall": _bucket(scored),
        "hidden_gem": _bucket([r for r in scored if _lens_of(r) == "hidden-gem"]),
        "core": _bucket([r for r in scored if _lens_of(r) == "core"]),
        "moat_bullish": _bucket([r for r in scored if _lens_of(r) == "moat-bullish"]),
    }
    as_of = max((str(r.get("evaluated_on") or "") for r in scored), default="")
    patterns: list[Lesson] = []
    calibration = None

    def lesson(text: str, worked: bool, kind: str) -> Lesson:
        return Lesson("confluence_swing", text, worked, "curated",
                      as_of=as_of, kind=kind)

    ov = stats["overall"]
    if ov.get("n", 0) >= MIN_BUCKET:
        # 1) CALIBRATION: did stated conviction match realized odds?
        ac, wr, n = ov.get("avg_conviction"), ov["win_rate_pct"], ov["n"]
        if isinstance(ac, (int, float)):
            gap = ac * 100 - wr
            if gap >= 20:
                calibration = (f"stated convictions run ~{gap:.0f}pp ABOVE the "
                               f"realized win rate ({wr:.0f}% over {n} scored calls)"
                               " - treat conviction as optimistic and demand more "
                               "confirmation before sizing up")
                patterns.append(lesson(calibration, False, "calibration"))
            elif gap <= -15:
                calibration = (f"stated convictions run ~{-gap:.0f}pp BELOW the "
                               f"realized win rate ({wr:.0f}% over {n} scored calls)"
                               " - the desk is under-betting its own edge")
                patterns.append(lesson(calibration, True, "calibration"))
        # 1b) BUCKET CALIBRATION: aggregate can net out, so also check whether
        # high-conviction calls actually beat low-conviction ones.
        inv = _conviction_inversion(scored)
        if inv:
            high_wr, low_wr, n_high, n_low = inv
            patterns.append(lesson(
                f"INVERTED conviction: the desk's higher-conviction half wins "
                f"only {high_wr:.0f}% ({n_high} calls) vs {low_wr:.0f}% "
                f"({n_low} calls) for the lower-conviction half - conviction is "
                "inversely related to outcome here, so treat HIGH conviction with "
                "extra skepticism and demand confirmation before sizing up",
                False, "calibration_buckets"))
        # 2) EXIT MIX: too many stops means entries are chasing.
        if ov.get("stop_rate_pct", 0) >= 50:
            patterns.append(lesson(
                f"{ov['stop_rate_pct']:.0f}% of the last {n} exits were STOPS - "
                "entries are chasing extended prices; prefer pullback entries "
                "near the 20-DMA and honor the PM's wait-for-pullback calls",
                False, "exit_mix"))
    # 3) LENS PERFORMANCE: does each discovery lens actually pay?
    for key, label in (("hidden_gem", "hidden-gem"), ("moat_bullish", "moat-bullish")):
        b = stats[key]
        if b.get("n", 0) >= MIN_BUCKET:
            if b["win_rate_pct"] >= 60:
                patterns.append(lesson(
                    f"{label} picks are PAYING ({b['win_rate_pct']:.0f}% of "
                    f"{b['n']} scored, avg {b['avg_return_pct']:+.1f}%) - this "
                    "lens has earned weight in the thesis", True, f"lens:{label}"))
            elif b["win_rate_pct"] <= 40:
                patterns.append(lesson(
                    f"{label} picks are NOT paying ({b['win_rate_pct']:.0f}% of "
                    f"{b['n']} scored) - demand stronger confirmation before "
                    "trusting this lens", False, f"lens:{label}"))
    return {"stats": stats, "pattern_lessons": patterns, "calibration": calibration}


AVG_POS = 1.0      # cohort avg return (%) at/above which a "worked" lesson holds
AVG_NEG = -1.0     # ... at/below which a "did not pay" lesson holds


def _cohort_returns(rows: list[dict]) -> dict:
    """{lens: (n, avg_return_pct)} over scored calls, the evidence anecdote
    lessons are gated on. Avg return (not win rate) captures payoff asymmetry —
    the core lens pays on a ~50% hit rate a win-rate gate would miss."""
    scored = [r for r in rows if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))]
    out: dict[str, tuple] = {}
    for key in ("hidden-gem", "core", "moat-bullish"):
        c = [r["return_pct"] for r in scored if _lens_of(r) == key]
        out[key] = (len(c), round(sum(c) / len(c), 2) if c else 0.0)
    return out


def _lens_of(r: dict) -> str:
    if r.get("hidden_gem"):
        return "hidden-gem"
    if r.get("moat_stance") == "bullish":
        return "moat-bullish"
    return "core"


def curate(mem, ledger_rows: list[dict], client=None, model: str = "",
           max_llm_lessons: int = 2) -> dict:
    """One self-review pass over the lesson memory. Mutates `mem`; the caller
    persists it. Returns a report dict for display.

    Fully automated, evidence-gated — no human sign-off anywhere:
      * every anecdote lesson is (re)judged against its LENS COHORT's realized
        avg return: ACTIVE while that cohort backs its claim, demoted to pending
        when the evidence is neutral, RETIRED when the evidence contradicts it;
      * pattern lessons are recomputed from the aggregates each run and any whose
        condition no longer holds is retired.
    So a lesson carries weight only while the numbers keep earning it."""
    report = assess(ledger_rows)
    cohorts = _cohort_returns(ledger_rows)
    # Migration / fallback: older anecdotes were saved without a cohort tag; map
    # each symbol to the lens its scored rec used so they can be judged too.
    sym_lens = {r["symbol"]: _lens_of(r) for r in ledger_rows
                if r.get("status") == "evaluated" and r.get("symbol")}

    # -- anecdotes: activation/retirement driven purely by cohort evidence -----
    activated = retired = 0
    keep = []
    for e in mem.entries:
        if e.lesson.kind:                      # pattern lessons handled below
            keep.append(e)
            continue
        was_active = e.human_reviewed
        cohort = e.lesson.cohort or sym_lens.get(e.lesson.symbol, "")
        n, avg = cohorts.get(cohort, (0, 0.0))
        worked = bool(e.lesson.thesis_correct)
        supported = n >= MIN_BUCKET and ((worked and avg >= AVG_POS)
                                         or (not worked and avg <= AVG_NEG))
        contradicted = n >= MIN_RETIRE and ((worked and avg <= AVG_NEG)
                                            or (not worked and avg >= AVG_POS))
        if contradicted:
            retired += 1                       # evidence rejects it -> not a lesson
            continue
        e.human_reviewed = supported           # active iff the cohort backs it now
        if supported and not was_active:
            activated += 1
        keep.append(e)
    mem.entries[:] = keep

    # -- pattern lessons: recompute from aggregates; retire the stale ----------
    supported_kinds = {les.kind for les in report["pattern_lessons"]}
    before = len(mem.entries)
    mem.entries[:] = [e for e in mem.entries
                      if not e.lesson.kind or e.lesson.kind in supported_kinds
                      or e.lesson.kind.startswith("llm:")]   # LLM synth persists
    retired += before - len(mem.entries)       # patterns whose evidence lapsed

    new = 0
    for les in report["pattern_lessons"]:
        prior = [e for e in mem.entries if e.lesson.kind == les.kind]
        changed = not prior or prior[0].lesson.lesson != les.lesson
        mem.entries[:] = [e for e in mem.entries if e.lesson.kind != les.kind]
        mem.add(les, human_reviewed=True)
        if changed:
            new += 1
    existing = {e.lesson.lesson for e in mem.entries}
    run_as_of = max((str(r.get("evaluated_on") or "")
                     for r in ledger_rows if r.get("status") == "evaluated"),
                    default="")

    # Optional LLM synthesis + lifecycle: patterns the rule set has no name for,
    # grounded strictly in the provided numbers (never invented trades). Unlike a
    # rule pattern, an LLM lesson has no recomputable condition, so it lives by
    # "re-earn or retire": each run that re-derives it refreshes its as_of; one
    # the synthesis stops producing ages out after LLM_TTL_DAYS, and a hard cap
    # bounds the count. This keeps the automated-learning invariant — a lesson
    # persists only while the evidence keeps regenerating it — with no human
    # pruning. (Deterministic/offline runs never touch LLM lessons.)
    if client is not None and not getattr(client, "deterministic", True):
        from system.agents.prompts import CURATOR
        scored = [{k: r.get(k) for k in ("symbol", "date", "conviction",
                                         "return_pct", "outcome", "hidden_gem",
                                         "moat_stance", "sector")}
                  for r in ledger_rows if r.get("status") == "evaluated"][-40:]
        fresh = None
        try:
            raw = client.complete(CURATOR,
                                  {"stats": report["stats"], "scored": scored},
                                  "CuratorLessons", model=model, max_tokens=600)
            fresh = [(t, bool(item.get("worked", False)))
                     for item in (raw.get("lessons") or [])[:max_llm_lessons]
                     if (t := str(item.get("text", "")).strip())]
        except Exception:
            fresh = None                       # synthesis failed: don't age/prune

        if fresh is not None:
            fresh_texts = {t for t, _ in fresh}
            by_text = {e.lesson.lesson: e for e in mem.entries
                       if e.lesson.kind.startswith("llm:")}
            for text, worked in fresh:
                if text in by_text:
                    by_text[text].lesson.as_of = run_as_of       # re-earned
                elif text not in existing:
                    mem.add(Lesson("confluence_swing", text, worked, "curated",
                                   kind=f"llm:{abs(hash(text)) % 100000}",
                                   as_of=run_as_of), human_reviewed=True)
                    existing.add(text)
                    new += 1
            # Age out LLM lessons this run's synthesis no longer supports.
            kept = []
            for e in mem.entries:
                if (not e.lesson.kind.startswith("llm:")
                        or e.lesson.lesson in fresh_texts):
                    kept.append(e)
                    continue
                stale = _days_stale(e.lesson.as_of, run_as_of)
                if stale is not None and stale > LLM_TTL_DAYS:
                    retired += 1               # not re-derived, past its grace
                else:
                    kept.append(e)
            mem.entries[:] = kept
            # Hard cap: keep only the freshest LLM lessons, retire the oldest.
            llm = [e for e in mem.entries if e.lesson.kind.startswith("llm:")]
            if len(llm) > LLM_MAX:
                oldest = sorted(llm, key=lambda e: e.lesson.as_of or "")
                drop = {id(e) for e in oldest[:len(llm) - LLM_MAX]}
                mem.entries[:] = [e for e in mem.entries if id(e) not in drop]
                retired += len(drop)

    return {"activated": activated, "retired": retired, "new_patterns": new,
            "pending": sum(1 for e in mem.entries if not e.human_reviewed),
            "active": sum(1 for e in mem.entries if e.human_reviewed),
            "cohorts": cohorts,
            "stats": report["stats"], "calibration": report["calibration"]}
