"""Recommendation ledger: persist every BUY/ADJUST the desk recommends and later
score how it actually performed - whether or not the trade was executed.

This closes the learning loop for advice (not just executed trades): each
recommendation is saved with its entry/stop/target and hold window, and once the
window elapses its forward return is measured from price history and fed into the
shared LessonMemory (so the agents learn from their own calls). Stored at:

    ~/.swing_system/recommendations.json
"""

from __future__ import annotations

import json

import pandas as pd

from app.config import CONFIG_DIR
from system.reflection.memory import TradeOutcome
from system.schemas import Lesson

LEDGER_PATH = CONFIG_DIR / "recommendations.json"


def load(path=None) -> list[dict]:
    p = path or LEDGER_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save(records: list[dict], path=None) -> None:
    p = path or LEDGER_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def record(recs: list[dict], source: str, as_of: str, path=None) -> int:
    """Append BUY/ADJUST recommendations as open ledger entries (idempotent per
    day+symbol). The same name surfaced by two screens (e.g. S&P 500 and QQQ) on
    one day is one opportunity, so it is recorded once — not double-counted in the
    learning stats. Returns how many new entries were added."""
    led = load(path)
    have = {(r["date"], r["symbol"]) for r in led}
    added = 0
    for r in recs:
        key = (as_of, r["symbol"])
        if key in have:
            continue
        led.append({
            "id": f"{as_of}:{r['symbol']}:{source}",
            "date": as_of, "symbol": r["symbol"], "action": "buy",
            "sector": r.get("sector", "?"),
            "entry": r.get("entry"), "stop": r.get("stop"), "target": r.get("target"),
            "conviction": r.get("conviction"), "hold_days": r.get("hold_days", 10),
            "exit_by": r.get("exit_by"), "thesis": (r.get("thesis") or "")[:300],
            "setup_type": "confluence_swing", "source": source, "status": "open",
            # Cohort tags: which lens produced the pick, so the desk can learn
            # whether hidden-gem and moat-bullish picks actually outperform.
            "hidden_gem": bool(r.get("hidden_gem")),
            "moat_stance": r.get("moat_stance"),
        })
        have.add(key)
        added += 1
    if added:
        save(led, path)
    return added


def mark_executed(symbol: str, qty: int | None = None, path=None) -> bool:
    """Link a placed order to its recommendation: the latest open entry for
    `symbol` is flagged executed, so the Learning/Performance views show which
    calls you actually took (advice quality vs. account performance)."""
    led = load(path)
    cands = [r for r in led if r.get("symbol") == symbol and r.get("status") == "open"]
    if not cands:
        return False
    r = max(cands, key=lambda x: x.get("date") or "")
    r["executed"] = True
    if qty:
        r["executed_qty"] = int(qty)
    r.setdefault("executed_on", pd.Timestamp.now().date().isoformat())
    save(led, path)
    return True


def open_for(symbol: str, path=None) -> dict | None:
    """The most recent OPEN ledger entry for a symbol (the trade's intent:
    exit_by, stop, target, thesis), or None."""
    cands = [r for r in load(path)
             if r.get("symbol") == symbol and r.get("status") == "open"]
    return max(cands, key=lambda r: r.get("date") or "") if cands else None


def extend_plan(symbol: str, new_exit_by: str, new_stop: float, path=None) -> dict | None:
    """Roll a winning open position's exit date forward and raise its stop (a
    one-shot trail, marked so it happens at most once). Returns the updated entry
    or None. Lets a winner run on the market's money instead of a hard time-exit,
    which the record shows leaves the big target moves on the table."""
    led = load(path)
    cands = [r for r in led if r.get("symbol") == symbol and r.get("status") == "open"]
    if not cands:
        return None
    r = max(cands, key=lambda x: x.get("date") or "")
    if r.get("extended"):
        return None                                # already trailed once
    r["exit_by"] = new_exit_by
    r["stop"] = round(float(new_stop), 2)
    r["extended"] = True
    save(led, path)
    return r


def raise_stop(symbol: str, new_stop: float, path=None) -> dict | None:
    """Raise a winning open position's protective stop ONCE (risk-reducing only),
    marked so it happens at most once and never LOWERS an existing stop. Returns
    the updated entry or None. Distinct from :func:`extend_plan`: the exit-by
    window is untouched — this only tightens the downside on a live winner, which
    the asymmetric-autonomy invariant allows the review to do unattended."""
    led = load(path)
    cands = [r for r in led if r.get("symbol") == symbol and r.get("status") == "open"]
    if not cands:
        return None
    r = max(cands, key=lambda x: x.get("date") or "")
    if r.get("stop_raised"):
        return None                                # already raised once
    if float(new_stop) <= float(r.get("stop") or 0):
        return None                                # only ever raise, never lower
    r["stop"] = round(float(new_stop), 2)
    r["stop_raised"] = True
    save(led, path)
    return r


def mark_closed(symbol: str, exit_price: float, when: str, reason: str,
                entry_price: float | None = None, memory=None, path=None) -> dict | None:
    """Close out the most recent open entry for `symbol` with the REALIZED
    result (actual fill prices, not the close-path estimate), feeding the
    outcome into LessonMemory. Returns the updated entry or None."""
    led = load(path)
    cands = [r for r in led if r.get("symbol") == symbol and r.get("status") == "open"]
    if not cands:
        return None
    r = max(cands, key=lambda x: x.get("date") or "")
    entry = float(entry_price if entry_price is not None else (r.get("entry") or 0))
    if entry <= 0 or exit_price <= 0:
        return None
    pnl_pct = round((float(exit_price) / entry - 1) * 100.0, 2)
    r.update(status="evaluated", return_pct=pnl_pct, outcome=reason,
             evaluated_on=when, realized=True)
    save(led, path)
    if memory is not None:
        conv = float(r.get("conviction") or 0.0)
        memory.record_outcome(TradeOutcome("confluence_swing", symbol, conv,
                                           pnl_pct, reason, when))
        verb = "paid" if pnl_pct > 0 else "did not pay"
        # PENDING: the curator activates this only while the trade's lens cohort
        # actually pays, and retires it otherwise (fully automated, evidence-gated).
        memory.add(Lesson("confluence_swing",
                          f"executed {symbol} {verb} {pnl_pct:+.1f}% (exit: {reason}).",
                          pnl_pct > 0, "clean" if reason in {"target", "time"} else "stopped",
                          symbol=symbol, as_of=when, pnl_pct=pnl_pct, conviction=conv,
                          cohort=_lens_cohort(r)),
                   human_reviewed=False)
    return r


def _price_on_or_after(series: pd.Series, when: str):
    idx = series.index[series.index >= pd.Timestamp(when)]
    if len(idx):
        v = float(series.loc[idx[0]])
        return v if v == v else None
    return None


def evaluate(closes: pd.DataFrame, today: str, memory=None, path=None) -> dict:
    """Score matured open recommendations from price history; feed outcomes into
    LessonMemory. Returns a summary. `closes` is the wide adjusted-close frame the
    screen already downloads."""
    led = load(path)
    evaluated = wins = 0
    rets = []
    changed = False
    for r in led:
        if r.get("status") != "open" or not r.get("exit_by"):
            continue
        if r["exit_by"] > today:                      # window not elapsed yet
            continue
        sym = r["symbol"]
        if sym not in closes.columns:
            continue
        s = closes[sym].dropna()
        entry = _price_on_or_after(s, r["date"])
        exit_px = _price_on_or_after(s, r["exit_by"])
        if exit_px is None and len(s):                # matured past data end -> last
            exit_px = float(s.iloc[-1])
        if entry is None or exit_px is None or entry <= 0:
            continue
        pnl_pct = (exit_px / entry - 1) * 100.0      # time exit: same series, split-safe
        # Close-path reason (no intraday): stop checked first (conservative).
        path_s = s[(s.index >= pd.Timestamp(r["date"])) & (s.index <= pd.Timestamp(r["exit_by"]))]
        reason = "time"
        # The recorded stop/target are ABSOLUTE levels in the rec-time price basis;
        # the re-fetched `closes` may use a different split/adjustment basis (a name
        # that split after the rec). Comparing the recorded level to the re-fetched
        # series mixes bases and can FALSELY fire a stop and compute a nonsensical
        # return. Use entry-relative RATIOS, which are invariant to later splits.
        rec_entry = r.get("entry")
        if rec_entry and float(rec_entry) > 0 and len(path_s):
            re_ = float(rec_entry)
            stop, target = r.get("stop"), r.get("target")
            if stop and float(path_s.min()) <= entry * (float(stop) / re_):
                reason, pnl_pct = "stop", (float(stop) / re_ - 1) * 100.0
            elif target and float(path_s.max()) >= entry * (float(target) / re_):
                reason, pnl_pct = "target", (float(target) / re_ - 1) * 100.0
        r["status"] = "evaluated"
        r["return_pct"] = round(pnl_pct, 2)
        r["outcome"] = reason
        r["evaluated_on"] = today
        # Market-relative result over the same window (when SPY is in frame):
        # beating cash and beating the market are different claims.
        if "SPY" in closes.columns:
            spy = closes["SPY"].dropna()
            s0 = _price_on_or_after(spy, r["date"])
            s1 = _price_on_or_after(spy, r["exit_by"]) or (float(spy.iloc[-1])
                                                           if len(spy) else None)
            if s0 and s1 and s0 > 0:
                r["excess_pct"] = round(pnl_pct - (s1 / s0 - 1) * 100.0, 2)
        evaluated += 1
        wins += 1 if pnl_pct > 0 else 0
        rets.append(pnl_pct)
        changed = True
        if memory is not None:
            conv = float(r.get("conviction") or 0.0)
            memory.record_outcome(TradeOutcome("confluence_swing", sym, conv,
                                               round(pnl_pct, 2), reason, r["exit_by"]))
            verb = "paid" if pnl_pct > 0 else "did not pay"
            # PENDING: the curator activates only while this lens cohort pays.
            memory.add(Lesson("confluence_swing",
                              f"recommended {sym} {verb} {pnl_pct:+.1f}% (exit: {reason}).",
                              pnl_pct > 0, "clean" if reason in {"target", "time"} else "stopped",
                              symbol=sym, as_of=r["exit_by"], pnl_pct=round(pnl_pct, 2),
                              conviction=conv, cohort=_lens_cohort(r)), human_reviewed=False)
    if changed:
        save(led, path)
    avg = round(sum(rets) / len(rets), 1) if rets else 0.0
    return {"evaluated": evaluated, "win_rate_pct": round(100 * wins / evaluated, 0) if evaluated else 0,
            "avg_return_pct": avg, "open": sum(1 for r in led if r.get("status") == "open"),
            "cohorts": cohort_stats(led)}


def _trade_weight(r: dict, risk_per_trade: float, max_weight: float,
                  typical_stop: float = 0.10) -> float:
    """The fraction of capital the desk's own sizer would put in this trade.

    Position size is `equity * risk_per_trade / (entry - stop)` shares, so the
    NOTIONAL weight is `risk_per_trade / stop_distance` — i.e. a tight stop earns
    a bigger position for the same 1% risk — capped at the single-name limit. A
    row missing its stop falls back to a typical stop distance."""
    entry, stop = r.get("entry"), r.get("stop")
    dist = typical_stop
    try:
        e, s = float(entry), float(stop)
        if e > 0 and e > s > 0:
            dist = (e - s) / e
    except (TypeError, ValueError):
        pass
    if dist <= 0:
        dist = typical_stop
    return min(risk_per_trade / dist, max_weight)


def equity_curve(rows: list[dict] | None = None, path=None) -> dict:
    """A faithful track record of the desk's scored calls, compounded at the
    position size the desk WOULD have used (1% equity at risk, weight =
    risk / stop-distance, capped at the single-name limit), in ENTRY-date order.

    The naive version compounded each call's full per-trade return as if 100% of
    capital rode every one sequentially — which turns a +1.2%-avg, ~0%-account
    record into a fictional +40% curve. Risk-weighting each call by its real
    ~10% notional makes the curve track what the advice would actually have done
    to an account. This is an ADVISORY-signal curve (every scored call, executed
    or not); the account's own equity comes from broker history."""
    from system.config import DEFAULT_CONFIG
    rows = load(path) if rows is None else rows
    lim = DEFAULT_CONFIG.limits
    scored = [r for r in rows if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))]
    # Entry-date order is when capital was actually deployed; evaluated_on
    # bunches many calls onto the few days their windows happened to elapse.
    scored.sort(key=lambda r: (str(r.get("date") or r.get("evaluated_on") or ""),
                               str(r.get("symbol") or "")))
    eq, v = [1.0], 1.0
    weights = []
    for r in scored:
        w = _trade_weight(r, lim.risk_per_trade, lim.max_single_name)
        weights.append(w)
        v *= 1.0 + (float(r["return_pct"]) / 100.0) * w
        eq.append(round(v, 4))
    return {"curve": eq, "n": len(scored),
            "total_return_pct": round((v - 1.0) * 100.0, 2),
            "avg_weight_pct": round(100.0 * sum(weights) / len(weights), 1)
            if weights else 0.0}


def _lens_cohort(r: dict) -> str:
    """The discovery lens a rec belongs to, used to validate its anecdote lesson
    against the right realized cohort. Most specific first: a hidden-gem pick is
    tagged 'hidden-gem', else a bullish-moat pick 'moat-bullish', else 'core'."""
    if r.get("hidden_gem"):
        return "hidden-gem"
    if r.get("moat_stance") == "bullish":
        return "moat-bullish"
    return "core"


def cohort_avg_returns(rows: list[dict] | None = None, path=None) -> dict:
    """{cohort: (n, avg_return_pct)} over scored calls, by the same lens tags the
    anecdote lessons carry — the evidence the curator gates lesson activation on.
    Gating on AVG RETURN (not just win rate) captures payoff asymmetry: the core
    lens pays on a ~50% hit rate, which a win-rate gate would miss."""
    rows = load(path) if rows is None else rows
    scored = [r for r in rows if r.get("status") == "evaluated"
              and isinstance(r.get("return_pct"), (int, float))]
    out: dict[str, tuple] = {}
    for key in ("hidden-gem", "core", "moat-bullish"):
        c = [r["return_pct"] for r in scored if _lens_cohort(r) == key]
        out[key] = (len(c), round(sum(c) / len(c), 2) if c else 0.0)
    return out


def _cohort(rows: list[dict]) -> dict:
    rets = [r["return_pct"] for r in rows if isinstance(r.get("return_pct"), (int, float))]
    if not rets:
        return {"n": 0}
    wins = sum(1 for x in rets if x > 0)
    return {"n": len(rets), "win_rate_pct": round(100 * wins / len(rets), 0),
            "avg_return_pct": round(sum(rets) / len(rets), 1)}


def cohort_stats(led: list[dict] | None = None, path=None) -> dict:
    """Performance split by the lens that produced each pick, over ALL scored
    recommendations: hidden-gem vs core, and moat-bullish vs the rest. This is
    how the desk learns whether the discovery lenses actually pay."""
    rows = [r for r in (led if led is not None else load(path))
            if r.get("status") == "evaluated"]
    return {
        "hidden_gem": _cohort([r for r in rows if r.get("hidden_gem")]),
        "core": _cohort([r for r in rows if not r.get("hidden_gem")]),
        "moat_bullish": _cohort([r for r in rows if r.get("moat_stance") == "bullish"]),
        "moat_other": _cohort([r for r in rows if r.get("moat_stance") not in (None, "bullish")]),
    }


def summarize(path=None) -> str:
    led = load(path)
    if not led:
        return "No recommendations recorded yet."
    done = [r for r in led if r.get("status") == "evaluated"]
    openr = sorted((r for r in led if r.get("status") == "open"),
                   key=lambda r: (str(r.get("exit_by") or ""), r.get("symbol", "")))
    n_exec = sum(1 for r in openr if r.get("executed"))
    lines = [f"Recommendation ledger: {len(led)} total - {len(openr)} open "
             f"({n_exec} executed in your account), {len(done)} scored."]
    if done:
        wins = sum(1 for r in done if (r.get("return_pct") or 0) > 0)
        avg = sum(r.get("return_pct") or 0 for r in done) / len(done)
        lines.append(f"  scored hit rate {100 * wins / len(done):.0f}%  |  "
                     f"avg forward return {avg:+.1f}%")
        for r in sorted(done, key=lambda x: x.get("return_pct") or 0, reverse=True)[:10]:
            ex = "  [executed]" if r.get("executed") else ""
            lines.append(f"   {r['date']}  {r['symbol']:<6} {r.get('return_pct', 0):+6.1f}%  "
                         f"({r.get('outcome', '?')}){ex}")
    if openr:
        lines.append("  open, by exit date (● = executed in your account, "
                     "○ = advisory only):")
        for r in openr:
            mark = "●" if r.get("executed") else "○"
            qty = f" x{r['executed_qty']}" if r.get("executed_qty") else ""
            lines.append(f"   {mark} {r['date']}  {r['symbol']:<6}{qty:<5} "
                         f"entry {r.get('entry')}  -> exit by {r.get('exit_by')}  "
                         f"[{r.get('source', '?')}]")
    return "\n".join(lines)
