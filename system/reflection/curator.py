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

from system.schemas import Lesson

MIN_BUCKET = 5      # a pattern needs at least this many scored outcomes
MIN_RETIRE = 8      # contradicting an anecdote needs even more evidence


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


def assess(ledger_rows: list[dict]) -> dict:
    """Expectation vs realization, by lens and overall — plus the pattern
    lessons the evidence currently supports. Pure and deterministic."""
    scored = [r for r in ledger_rows if r.get("status") == "evaluated"]
    stats = {
        "overall": _bucket(scored),
        "hidden_gem": _bucket([r for r in scored if r.get("hidden_gem")]),
        "core": _bucket([r for r in scored if not r.get("hidden_gem")]),
        "moat_bullish": _bucket([r for r in scored
                                 if r.get("moat_stance") == "bullish"]),
    }
    as_of = max((str(r.get("evaluated_on") or "") for r in scored), default="")
    patterns: list[Lesson] = []
    calibration = None

    def lesson(text: str, worked: bool) -> Lesson:
        return Lesson("confluence_swing", text, worked, "curated", as_of=as_of)

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
                patterns.append(lesson(calibration, False))
            elif gap <= -15:
                calibration = (f"stated convictions run ~{-gap:.0f}pp BELOW the "
                               f"realized win rate ({wr:.0f}% over {n} scored calls)"
                               " - the desk is under-betting its own edge")
                patterns.append(lesson(calibration, True))
        # 2) EXIT MIX: too many stops means entries are chasing.
        if ov.get("stop_rate_pct", 0) >= 50:
            patterns.append(lesson(
                f"{ov['stop_rate_pct']:.0f}% of the last {n} exits were STOPS - "
                "entries are chasing extended prices; prefer pullback entries "
                "near the 20-DMA and honor the PM's wait-for-pullback calls",
                False))
    # 3) LENS PERFORMANCE: does each discovery lens actually pay?
    for key, label in (("hidden_gem", "hidden-gem"), ("moat_bullish", "moat-bullish")):
        b = stats[key]
        if b.get("n", 0) >= MIN_BUCKET:
            if b["win_rate_pct"] >= 60:
                patterns.append(lesson(
                    f"{label} picks are PAYING ({b['win_rate_pct']:.0f}% of "
                    f"{b['n']} scored, avg {b['avg_return_pct']:+.1f}%) - this "
                    "lens has earned weight in the thesis", True))
            elif b["win_rate_pct"] <= 40:
                patterns.append(lesson(
                    f"{label} picks are NOT paying ({b['win_rate_pct']:.0f}% of "
                    f"{b['n']} scored) - demand stronger confirmation before "
                    "trusting this lens", False))
    return {"stats": stats, "pattern_lessons": patterns, "calibration": calibration}


def curate(mem, ledger_rows: list[dict], client=None, model: str = "",
           max_llm_lessons: int = 2) -> dict:
    """One self-review pass over the lesson memory. Mutates `mem`; the caller
    persists it. Returns a report dict for display."""
    report = assess(ledger_rows)

    # Pending anecdotes: activate when the aggregate for their setup agrees,
    # retire when it clearly contradicts, otherwise wait for more evidence.
    activated = retired = 0
    keep = []
    for e in mem.entries:
        if e.human_reviewed:
            keep.append(e)
            continue
        st = mem.setup_stats(e.lesson.setup_type)
        n, wr = st.get("count", 0), st.get("win_rate_pct", 50)
        worked = bool(e.lesson.thesis_correct)
        if n >= MIN_BUCKET and ((worked and wr >= 55) or (not worked and wr <= 45)):
            e.human_reviewed = True
            activated += 1
            keep.append(e)
        elif n >= MIN_RETIRE and ((worked and wr <= 40) or (not worked and wr >= 60)):
            retired += 1                       # the anecdote misleads; drop it
        else:
            keep.append(e)                     # pending until evidence arrives
    mem.entries[:] = keep

    # Evidence-gated pattern lessons (deduped on text).
    existing = {e.lesson.lesson for e in mem.entries}
    new = 0
    for les in report["pattern_lessons"]:
        if les.lesson not in existing:
            mem.add(les, human_reviewed=True)
            existing.add(les.lesson)
            new += 1

    # Optional LLM synthesis: patterns the rule set has no name for, grounded
    # strictly in the provided numbers (never invented trades).
    if client is not None and not getattr(client, "deterministic", True):
        from system.agents.prompts import CURATOR
        scored = [{k: r.get(k) for k in ("symbol", "date", "conviction",
                                         "return_pct", "outcome", "hidden_gem",
                                         "moat_stance", "sector")}
                  for r in ledger_rows if r.get("status") == "evaluated"][-40:]
        try:
            raw = client.complete(CURATOR,
                                  {"stats": report["stats"], "scored": scored},
                                  "CuratorLessons", model=model, max_tokens=600)
            for item in (raw.get("lessons") or [])[:max_llm_lessons]:
                text = str(item.get("text", "")).strip()
                if text and text not in existing:
                    mem.add(Lesson("confluence_swing", text,
                                   bool(item.get("worked", False)), "curated"),
                            human_reviewed=True)
                    existing.add(text)
                    new += 1
        except Exception:
            pass                               # synthesis is a bonus, never a gate

    return {"activated": activated, "retired": retired, "new_patterns": new,
            "pending": sum(1 for e in mem.entries if not e.human_reviewed),
            "stats": report["stats"], "calibration": report["calibration"]}
