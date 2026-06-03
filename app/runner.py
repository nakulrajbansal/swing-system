"""Run backend: drive the harness / paper engine and stream output to a callback.

GUI-agnostic so it can also run headless (``app.main --selftest``). Any print()
inside the system is redirected to the callback, and key results are emitted
explicitly. Designed to run inside a worker thread.
"""

from __future__ import annotations

import contextlib
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import CONFIG_DIR, AppConfig

Emit = Callable[[str], None]

# Every run is persisted here so it can be inspected afterwards.
LOGS_DIR = CONFIG_DIR / "logs"


@contextlib.contextmanager
def _run_logger(emit: Emit, kind: str):
    """Tee `emit` to a timestamped logfile for the duration of a run.

    Yields (log, path): `log` forwards each line to both the file and the
    original `emit` (the GUI console / headless stdout).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    f = open(path, "w", encoding="utf-8")

    def log(line: str) -> None:
        try:
            f.write(line + "\n")
            f.flush()
        except Exception:
            pass
        emit(line)

    log(f"[log] saving this run to: {path}")
    try:
        yield log, path
    finally:
        log(f"[log] run log saved to: {path}")
        f.close()


class _StreamToEmit(io.TextIOBase):
    """A file-like object that forwards written text, line-wise, to `emit`."""

    def __init__(self, emit: Emit):
        self._emit = emit
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self):
        if self._buf:
            self._emit(self._buf)
            self._buf = ""


@contextlib.contextmanager
def _redirect(emit: Emit):
    stream = _StreamToEmit(emit)
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        yield
    stream.flush()


def _resolve_client(cfg: AppConfig, log: Emit):
    """Pick the LLM client and, for a real one, PREFLIGHT it so API failures are
    surfaced loudly instead of silently swallowed. Returns (client, real_llm).

    On any preflight failure (no credits, bad key, bad model, network) we report
    the exact error and fall back to the free deterministic mock so the run still
    produces useful output.
    """
    from system.agents.llm_client import MockLLMClient, default_client
    from system.config import DEFAULT_CONFIG

    if not (cfg.use_llm_agents and cfg.anthropic_api_key):
        log("[agents] deterministic agents (MockLLMClient) — free, no API calls.")
        return MockLLMClient(), False

    client = default_client()
    if client.deterministic:
        log("[warn] LLM requested but the Anthropic SDK/key is unavailable — using "
            "the deterministic mock (no tokens spent).")
        return client, False

    log("[agents] verifying Anthropic API access (one tiny preflight call) ...")
    try:
        client.complete("preflight", {"ok": 1}, "object",
                        model=DEFAULT_CONFIG.models.framing, max_tokens=8)
        log(f"[agents] Anthropic API OK (model {DEFAULT_CONFIG.models.framing}). "
            "Real LLM agents are active.")
        return client, True
    except Exception as exc:
        log(f"[error] Anthropic API is not usable: {type(exc).__name__}: {exc}")
        log("[error] Most often this means the account has no credits. Add credits at "
            "https://console.anthropic.com/settings/billing, or turn OFF 'Use LLM "
            "agents'.")
        log("[agents] Falling back to the free deterministic agents so this run still "
            "produces output.")
        return MockLLMClient(), False


def _build_store(cfg: AppConfig, emit: Emit):
    """Build (or load) the PIT store for this run; returns (store, sector_map)."""
    if cfg.data_source == "live":
        return _build_live_store(cfg, emit)
    return _build_synthetic_store(cfg, emit)


def _build_synthetic_store(cfg: AppConfig, emit: Emit):
    from harness.data.loader import SyntheticConfig, SyntheticLoader
    from harness.data.pit_store import PITStore
    import tempfile

    emit(f"[data] SYNTHETIC universe (planted signal): {cfg.n_symbols} symbols, "
         f"{cfg.start_date} -> {cfg.end_date} (seed {cfg.seed}) ...")
    store = PITStore(tempfile.mkdtemp(prefix="swing_app_"))
    loader = SyntheticLoader(store, SyntheticConfig(
        n_symbols=int(cfg.n_symbols), start=cfg.start_date, end=cfg.end_date,
        seed=int(cfg.seed)))
    loader.load_all()
    emit("[data] synthetic universe ready (NOTE: planted edge; not real markets).")
    return store, loader.sector_map()


def _build_live_store(cfg: AppConfig, emit: Emit):
    """Real free data (yfinance), cached to ~/.swing_system/data_store so repeat
    runs don't re-hit the network. Delete that folder to force a refresh."""
    from harness.data.loader import LiveLoader, live_symbols
    from harness.data.pit_store import PITStore

    syms = live_symbols(int(cfg.n_symbols))
    cache = (CONFIG_DIR / "data_store" /
             f"live_{len(syms)}_{cfg.start_date}_{cfg.end_date}")
    store = PITStore(cache)
    loader = LiveLoader(store, symbols=syms, start=cfg.start_date,
                        end=cfg.end_date, emit=emit)
    if (cache / "prices.parquet").exists():
        emit(f"[data] LIVE: using cached real data at {cache}")
        emit("[data] (delete that folder to refetch from Yahoo)")
    else:
        emit(f"[data] LIVE: fetching real data for {len(syms)} symbols from "
             "Yahoo Finance - first run is slow ...")
        loader.load_all()
    emit("[data] live universe ready (REAL market prices + corporate actions).")
    emit("[note] filing/insider/news tables are not yet wired for live data, so "
         "only the price-based momentum edge has real inputs.")
    return store, loader.sector_map()


def run_validation(cfg: AppConfig, emit: Emit) -> dict:
    """Run the Phase-1 validation harness; return the portfolio summary."""
    from harness.report.report import edge_scorecard, format_scorecard, portfolio_summary
    from harness.signals import ALL_FREE_EDGES
    from harness.study.costs import CostModel
    from harness.study.event_study import run_event_study

    cfg.apply_to_env()
    with _run_logger(emit, "validation") as (log, _path):
        t0 = time.time()
        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            costs = CostModel()
            cards, results = [], []
            for EdgeCls in ALL_FREE_EDGES:
                sig = EdgeCls()
                log(f"[edge] studying {sig.edge_id} ...")
                res = run_event_study(store, sig, sector_map, costs=costs,
                                      oos_start=cfg.oos_start)
                card = edge_scorecard(res, costs)
                cards.append(card)
                results.append(res)
                log(format_scorecard(card))
            summary = portfolio_summary(cards, results)
        log(f"\n[done] validation finished in {time.time() - t0:.1f}s")
        log(f"[summary] edges passed: {summary['edges_passed'] or 'none'}")
        log(f"[summary] portfolio verdict: {summary['portfolio_verdict']}")
    return summary


def run_paper(cfg: AppConfig, emit: Emit) -> dict:
    """Run the end-to-end paper-trading engine; return a result summary."""
    from system.run_live import PaperTradingEngine

    cfg.apply_to_env()
    with _run_logger(emit, "paper") as (log, _path):
        client, real_llm = _resolve_client(cfg, log)
        if real_llm:
            log("[warn] real LLM over a multi-cycle backtest can make MANY calls "
                "(capped at ANTHROPIC_MAX_CALLS) and spends tokens. The cheaper, "
                "design-correct use is 'Run live deliberation' (a single day).")
        t0 = time.time()
        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            log("[engine] starting paper-trading cycles (paper-only) ...")
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity))
            result = engine.run()
        log(f"\n[done] paper run finished in {time.time() - t0:.1f}s")
        log(f"[result] cycles={result.n_cycles}  "
            f"final_equity={result.final_equity:,.0f} ({result.total_return_pct:+.1f}%)  "
            f"trades={len(result.closed_trades)}")
        log(f"[result] scorecard={result.scorecard}")
    return {
        "cycles": result.n_cycles,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "trades": len(result.closed_trades),
        "scorecard": result.scorecard,
    }


def _fmt_payload(agent: str, p: dict) -> str:
    if agent == "hypothesis":
        if p.get("decision") != "propose":
            return "decline"
        return (f"PROPOSE (conviction {p.get('raw_conviction', 0):.2f}, "
                f"hold {p.get('expected_hold_days')}d) - {p.get('mechanism', '')[:160]}")
    if agent == "skeptic":
        objs = p.get("objections", [])
        return (f"verdict={p.get('verdict')} | strongest: {p.get('strongest', '')[:140]} "
                f"({len(objs)} objections)")
    if agent == "portfolio_manager":
        if p.get("action") == "pass":
            return f"PASS - {p.get('decisive_factor', '')[:140]}"
        return (f"{str(p.get('action', '')).upper()} (final conviction "
                f"{p.get('final_conviction', 0):.2f}) entry~{p.get('entry')} "
                f"stop~{p.get('stop')} target~{p.get('target')} - "
                f"{p.get('decisive_factor', '')[:120]}")
    return str(p)[:160]


def run_deliberation(cfg: AppConfig, emit: Emit) -> dict:
    """Run ONE decision cycle on the most recent session and show the full
    deliberation (specialists -> confluence -> Hypothesis/Skeptic/PM -> Risk
    Governor). Bounded cost: a single day, so the LLM is called a small,
    predictable number of times. This is the design-correct way to use the
    agents (a forward decision, not a backtest). No orders are placed."""
    from system.risk.governor import GovernorContext
    from system.run_live import PaperTradingEngine
    from system.data_plane.indicators import last_atr

    cfg.apply_to_env()
    with _run_logger(emit, "deliberation") as (log, _path):
        client, real_llm = _resolve_client(cfg, log)
        if real_llm:
            log("[agents] single-day deliberation: a small, bounded number of API calls.")

        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity))
            session = engine._last_session()
            from harness.data.loader import available_at_for_session
            T = available_at_for_session(session)
            log(f"\n[deliberation] session = {session.date()}  |  universe = "
                f"{len(engine.universe)} names")
            cycle = engine.orchestrator.run_cycle(T)

        if not cycle.candidates:
            log("\n[result] No high-confidence candidates today - confluence not "
                "satisfied (needs >=2 families agreeing, or one very strong family). "
                "The system's default is to do nothing.")
        for cand in cycle.candidates:
            log(f"\n=== CANDIDATE: {cand.symbol} ===")
            log(f"  families={cand.families}  edges={cand.edge_ids}  "
                f"score={cand.combined_score:.2f}  strong_single={cand.strong_single}")
            for env in cycle.deliberation.get(cand.symbol, []):
                log(f"  [{env['agent']}] {_fmt_payload(env['agent'], env['payload'])}")

        # Two-key view: what the Risk Governor would actually approve/size today.
        log("\n[two-key] Risk Governor sizing of any ENTER decisions:")
        any_enter = False
        for d in cycle.decisions:
            if d.action not in {"enter", "adjust"}:
                continue
            any_enter = True
            px = engine.panels.get(d.symbol)
            if px is None or session not in px.index:
                continue
            ref = float(px.loc[session, "close"])
            atr = last_atr(px.loc[px.index <= session].reset_index())
            ctx = GovernorContext(equity=float(cfg.starting_equity), reference_price=ref,
                                  atr_value=atr if atr == atr else 0.0,
                                  sector=sector_map.get(d.symbol, "?"))
            ticket = engine.governor.evaluate(d.symbol, d.action, ctx)
            if ticket.approved:
                log(f"  {d.symbol}: APPROVE {ticket.shares} sh @ ~{ticket.entry:.2f} "
                    f"stop {ticket.stop:.2f} target {ticket.target:.2f} "
                    f"(binding cap: {ticket.binding_cap})")
            else:
                log(f"  {d.symbol}: REJECT - {ticket.reason}")
        if not any_enter:
            log("  (no ENTER decisions - nothing to size)")

        calls = getattr(client, "calls", 0)
        if real_llm:
            log(f"\n[cost] LLM API calls made this deliberation: {calls} "
                f"(cap {getattr(client, 'max_calls', '?')})")
        log(f"[done] deliberation complete for {session.date()}.")

    return {"session": str(session.date()),
            "candidates": [c.symbol for c in cycle.candidates],
            "decisions": [(d.symbol, d.action) for d in cycle.decisions],
            "llm_calls": getattr(client, "calls", 0)}
