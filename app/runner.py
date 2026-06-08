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

import pandas as pd

from app.config import CONFIG_DIR, AppConfig

Emit = Callable[[str], None]

# Every run is persisted here so it can be inspected afterwards.
LOGS_DIR = CONFIG_DIR / "logs"

REDDIT_PROMPT = (
    "You are a social-media sentiment analyst for equities. From the Reddit post "
    "titles provided about ONE ticker, judge the crowd's directional sentiment, "
    "your calibrated conviction (0-1), the key themes in a short phrase, and "
    "whether the discussion is substantive or hype. Ignore joke price targets. "
    'Return ONLY a JSON object: {"sentiment":"bullish|bearish|neutral", '
    '"conviction":0.0-1.0, "themes":"...", "quality":"substantive|hype|mixed"}.'
)


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
        log("[agents] deterministic agents (MockLLMClient) - free, no API calls.")
        return MockLLMClient(), False

    client = default_client()
    if client.deterministic:
        log("[warn] LLM requested but the Anthropic SDK/key is unavailable - using "
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


def _gated_edges(cfg: AppConfig, log: Emit) -> list:
    """Edge classes allowed to trade. With the gate on, only edges that PASSED
    the most recent validation (master principle: validate before trading)."""
    from harness.signals import ALL_FREE_EDGES
    if not cfg.only_validated_edges:
        log("[gate] 'Only trade validated edges' is OFF -> using ALL edges (ungated).")
        return list(ALL_FREE_EDGES)
    from app.gating import load_validated
    v = load_validated()
    passed = set(v.get("passed") or [])
    if not passed:
        return []
    if v.get("data_source") and v["data_source"] != cfg.data_source:
        log(f"[gate] WARNING: validated edges were from data_source={v['data_source']!r}, "
            f"but this run uses {cfg.data_source!r}. Re-run validation on the same data "
            "for a meaningful gate.")
    allowed = [E for E in ALL_FREE_EDGES if getattr(E, "edge_id", None) in passed]
    log(f"[gate] validated edges in use: {[getattr(E, 'edge_id') for E in allowed]} "
        f"(validated {v.get('saved_at')}).")
    return allowed


# --------------------------------------------------------------------------
# Alpaca broker integration
# --------------------------------------------------------------------------
def _alpaca_broker(cfg: AppConfig):
    """Construct an AlpacaBroker for the configured environment. Live (real money)
    requires `enable_live_trading` to be on as well (asymmetric-autonomy gate)."""
    from system.execution.broker import AlpacaBroker
    live = cfg.alpaca_env == "live"
    return AlpacaBroker(cfg.alpaca_key_id, cfg.alpaca_secret, env=cfg.alpaca_env,
                        confirm_live=(live and bool(cfg.enable_live_trading)))


def check_alpaca(cfg: AppConfig, emit: Emit) -> dict:
    """Determine whether the app can transact on Alpaca: hit /v2/account and
    report status, buying power, and the active (paper/live) environment.
    Read-only - places no orders."""
    cfg.apply_to_env()
    result = {"ok": False, "env": cfg.alpaca_env}
    with _run_logger(emit, "alpaca-check") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[alpaca] No API keys saved. Enter your Alpaca key id + secret on the "
                "Configuration tab and Save first.")
            return result
        log(f"[alpaca] checking the {cfg.alpaca_env.upper()} account ...")
        try:
            broker = _alpaca_broker(cfg)
            log(f"[alpaca] endpoint: {broker.base}")
            ok, reason = broker.is_tradable()
            log(f"[alpaca] {reason}")
            result["ok"] = ok
            if ok:
                log("[alpaca] RESULT: account is ACTIVE and able to make transactions.")
            else:
                log("[alpaca] RESULT: reachable but NOT currently tradable (see flags).")
        except Exception as exc:
            log(f"[error] Alpaca check failed: {exc}")
            log("[hint] 401/403 usually means wrong keys or wrong environment - paper "
                "keys and live keys are DIFFERENT. For live, also enable live trading. "
                "Generate keys at https://app.alpaca.markets (paper) or your live dashboard.")
    return result


def _maybe_place_orders(cfg: AppConfig, tickets, sector_map, log: Emit) -> None:
    if not cfg.place_orders:
        if tickets:
            log("\n[orders] 'Place orders on Alpaca' is OFF - showing proposals only, "
                "nothing was submitted.")
        return
    if not tickets:
        log("\n[orders] nothing approved to submit.")
        return
    env = cfg.alpaca_env
    if env == "live" and not cfg.enable_live_trading:
        log("\n[orders] place_orders is ON and env=LIVE, but 'Enable live trading' is "
            "OFF - refusing to send real-money orders. Turn it on to proceed, or use "
            "env=paper.")
        return
    log(f"\n[orders] submitting {len(tickets)} approved order(s) to Alpaca "
        f"{env.upper()} {'(REAL MONEY)' if env == 'live' else '(paper)'} ...")
    try:
        broker = _alpaca_broker(cfg)
    except Exception as exc:
        log(f"[error] could not open Alpaca broker: {exc}")
        return
    for t in tickets:
        band = 0.005
        try:
            o = broker.submit_entry(t.symbol, t.shares, band_low=t.entry * (1 - band),
                                    band_high=t.entry * (1 + band), stop=t.stop,
                                    target=t.target, sector=sector_map.get(t.symbol, "?"))
            log(f"  [orders] {t.symbol}: submitted {t.shares} sh - order id "
                f"{o.get('id', '?')} status {o.get('status', '?')}")
        except Exception as exc:
            log(f"  [orders] {t.symbol}: FAILED - {exc}")


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
    ua = cfg.edgar_user_agent or None
    tag = "edgar" if ua else "noedgar"
    cache = (CONFIG_DIR / "data_store" /
             f"live_{len(syms)}_{cfg.start_date}_{cfg.end_date}_{tag}")
    store = PITStore(cache)
    loader = LiveLoader(store, symbols=syms, start=cfg.start_date,
                        end=cfg.end_date, emit=emit, edgar_user_agent=ua)
    if (cache / "prices.parquet").exists():
        emit(f"[data] LIVE: using cached real data at {cache}")
        emit("[data] (delete that folder to refetch)")
    else:
        emit(f"[data] LIVE: fetching real data for {len(syms)} symbols "
             f"({'with' if ua else 'without'} EDGAR) - first run is slow ...")
        loader.load_all()
    emit("[data] live universe ready (REAL market prices + corporate actions"
         f"{' + EDGAR 8-K/insider' if ua else ''}).")
    if not ua:
        emit("[note] no EDGAR User-Agent set -> only price/momentum edge has live "
             "inputs. Add one on the Configuration tab to enable 8-K + insider edges.")
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
        # Record which edges are cleared to trade live (the validation gate).
        from app.gating import save_validated
        save_validated(summary["edges_passed"], cfg.data_source)
        log(f"[gate] recorded validated edges ({cfg.data_source}): "
            f"{summary['edges_passed'] or 'none'} -> live deliberation will use these.")
    return summary


def run_insider_validation(cfg: AppConfig, emit: Emit) -> dict:
    """Validate the INSIDER edge (edge 6) on YEARS of REAL history, using SEC's
    quarterly bulk insider-transaction datasets (no per-filing fetches). Updates
    only edge 6's status in the validation gate (other edges' validation is
    preserved), so a passing insider edge can trade live."""
    from harness.data.loader import (LIVE_UNIVERSE, LiveLoader, fetch_insider_quarter,
                                     live_symbols, recent_quarters)
    from harness.data.pit_store import PITStore
    from harness.report.report import edge_scorecard, format_scorecard
    from harness.signals import Edge06Insider
    from harness.study.costs import CostModel
    from harness.study.event_study import run_event_study
    from app.gating import load_validated, save_validated

    cfg.apply_to_env()
    with _run_logger(emit, "insider-validation") as (log, _path):
        ua = cfg.edgar_user_agent
        if not ua:
            log("[error] EDGAR User-Agent required for bulk SEC data. Set it on the "
                "Configuration tab (e.g. your email) and Save.")
            return {"passed": []}

        syms = live_symbols(int(cfg.n_symbols))
        nq = int(cfg.insider_history_quarters)
        quarters = recent_quarters(nq)
        start = f"{min(q[0] for q in quarters)}-01-01"
        cache = CONFIG_DIR / "data_store" / f"insider_val_{len(syms)}_{nq}q"
        store = PITStore(cache)

        with _redirect(log):
            if not (cache / "prices.parquet").exists():
                log(f"[hist] fetching {nq}q of prices for {len(syms)} symbols from "
                    f"{start} - slow, cached after ...")
                LiveLoader(store, symbols=syms, start=start, end=None, emit=log).load_all()
            else:
                log(f"[hist] using cached prices at {cache}")
            if not (cache / "form4.parquet").exists():
                tickset, frames = set(syms), []
                for (y, q) in quarters:
                    log(f"[hist] insider dataset {y}Q{q} ...")
                    try:
                        frames.append(fetch_insider_quarter(y, q, ua, tickset))
                    except Exception as exc:
                        log(f"[hist] skip {y}Q{q}: {type(exc).__name__}: {exc}")
                frames = [f for f in frames if not f.empty]
                if frames:
                    store.write_form4(pd.concat(frames, ignore_index=True))
                log(f"[hist] insider PURCHASE events loaded: {sum(len(f) for f in frames)}")
            else:
                log(f"[hist] using cached insider history at {cache}")

            sector_map = {s: etf for etf, tk in LIVE_UNIVERSE.items() for s in tk}
            costs = CostModel()
            oos = (pd.Timestamp.now() - pd.Timedelta(days=365)).date().isoformat()
            sig = Edge06Insider()
            log(f"[edge] studying {sig.edge_id} on real history ...")
            res = run_event_study(store, sig, sector_map, costs=costs, oos_start=oos)
            card = edge_scorecard(res, costs)
            log(format_scorecard(card))

        # Merge into the gate: only flip edge 6's status; keep other edges as-is.
        prev = set(load_validated().get("passed") or [])
        if card["verdict"] == "PASS":
            prev.add(sig.edge_id)
        else:
            prev.discard(sig.edge_id)
        save_validated(sorted(prev), "live-historical")
        log(f"\n[gate] edge 6 verdict: {card['verdict']} "
            f"({card['n_events']} events). Gate now allows: {sorted(prev) or 'none'}.")
    return {"verdict": card["verdict"], "n_events": card["n_events"],
            "passed": sorted(prev)}


def run_filing_validation(cfg: AppConfig, emit: Emit) -> dict:
    """Validate the FILING-TEXT edge (edge 1) on REAL history: fetch each stock's
    last N 10-K/10-Q filings with cleaned full text, build a cached store with
    prices, and validate how filing-text change predicts abnormal returns.
    Updates only edge 1's status in the gate."""
    from harness.data.loader import (LIVE_UNIVERSE, LiveLoader, fetch_cik_map,
                                     fetch_edgar_for_symbol, live_symbols)
    from harness.data.pit_store import PITStore
    from harness.report.report import edge_scorecard, format_scorecard
    from harness.signals import Edge01Filing
    from harness.study.costs import CostModel
    from harness.study.event_study import run_event_study
    from app.gating import load_validated, save_validated

    cfg.apply_to_env()
    with _run_logger(emit, "filing-validation") as (log, _path):
        ua = cfg.edgar_user_agent
        if not ua:
            log("[error] EDGAR User-Agent required (set it on the Configuration tab).")
            return {"passed": []}

        syms = live_symbols(int(cfg.n_symbols))
        n = int(cfg.filing_history_count)
        cache = CONFIG_DIR / "data_store" / f"filing_val_{len(syms)}_{n}"
        store = PITStore(cache)

        with _redirect(log):
            if not (cache / "prices.parquet").exists():
                start = (pd.Timestamp.now() - pd.Timedelta(days=365 * 5)).date().isoformat()
                log(f"[hist] fetching prices for {len(syms)} symbols from {start} ...")
                LiveLoader(store, symbols=syms, start=start, end=None, emit=log).load_all()
            else:
                log("[hist] using cached prices.")
            if not (cache / "filings.parquet").exists():
                cm = fetch_cik_map(ua)
                rows = []
                for i, s in enumerate(syms, 1):
                    cik = cm.get(s)
                    if not cik:
                        continue
                    log(f"[hist] filings {s} ({i}/{len(syms)}) ...")
                    try:
                        filings, _ = fetch_edgar_for_symbol(s, cik, ua, since_days=0,
                                                            periodic_text=n)
                    except Exception as exc:
                        log(f"[hist] skip {s}: {type(exc).__name__}: {exc}")
                        continue
                    rows.extend(filings)
                if rows:
                    store.write_filings(pd.DataFrame(rows))
                log(f"[hist] periodic filings loaded: {len(rows)}")
            else:
                log("[hist] using cached filings.")

            sector_map = {s: etf for etf, tk in LIVE_UNIVERSE.items() for s in tk}
            oos = (pd.Timestamp.now() - pd.Timedelta(days=365)).date().isoformat()
            sig = Edge01Filing()
            log(f"[edge] studying {sig.edge_id} on real history ...")
            res = run_event_study(store, sig, sector_map, costs=CostModel(), oos_start=oos)
            card = edge_scorecard(res)
            log(format_scorecard(card))

        prev = set(load_validated().get("passed") or [])
        if card["verdict"] == "PASS":
            prev.add(sig.edge_id)
        else:
            prev.discard(sig.edge_id)
        save_validated(sorted(prev), "live-historical")
        log(f"\n[gate] edge 1 verdict: {card['verdict']} ({card['n_events']} events). "
            f"Gate now allows: {sorted(prev) or 'none'}.")
    return {"verdict": card["verdict"], "n_events": card["n_events"],
            "passed": sorted(prev)}


def _short(obj, n: int = 300) -> str:
    import json
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ..."


def _sent(text, n: int = 900) -> str:
    """Trim to a whole sentence (never mid-word) so the summary reads cleanly."""
    s = str(text or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    dot = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:dot + 1] if dot > n * 0.5 else cut.rsplit(" ", 1)[0]) + " ..."


def _step_out(transcript: dict, agent: str) -> dict:
    return next((s["output"] for s in transcript.get("steps", []) if s["agent"] == agent), {})


def _print_transcript(log: Emit, symbol: str, transcript: dict, verbose: bool) -> None:
    """Print the full deliberation: evidence given, then each agent's prompt,
    inputs, and output (so it is not a black box)."""
    ev = transcript.get("evidence", {})
    log(f"\n----- EVIDENCE GIVEN TO THE AGENTS  ({symbol}) -----")
    log(f"  technicals: {_short(ev.get('technicals', {}), 400)}")
    fl = dict(ev.get("filings", {}))
    snippet = fl.pop("risk_text_snippet", None)
    log(f"  filings: {_short(fl, 400)}")
    if snippet and verbose:
        log(f"  risk-factor text (snippet): {snippet[:450]} ...")
    log(f"  insider: {_short(ev.get('insider', {}), 300)}")
    if ev.get("recent_news"):
        log(f"  news: {ev.get('recent_news')}")
    log(f"\n----- AGENT DELIBERATION  ({symbol}) -----")
    for st in transcript.get("steps", []):
        log(f"\n  >>> AGENT: {st['agent']}  (model {st.get('model', '-')})")
        if verbose and st.get("system_prompt"):
            log(f"      PROMPT:  {_short(st['system_prompt'], 600)}")
            log(f"      INPUTS:  {_short(st.get('inputs', {}), 400)}")
        log(f"      OUTPUT:  {_short(st.get('output', {}), 1600)}")


def _build_ticker_store(cfg: AppConfig, ticker: str, emit: Emit):
    """A focused PIT store for ONE symbol (prices ~2y + EDGAR), for single-ticker
    analysis — even a symbol outside the standard universe."""
    import datetime

    from harness.data.loader import (LIVE_UNIVERSE, fetch_cik_map, fetch_edgar_for_symbol,
                                     fetch_prices_yahoo)
    from harness.data.pit_store import PITStore

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=520)).isoformat()
    cache = CONFIG_DIR / "data_store" / f"ticker_{ticker}_{today.isoformat()}"
    store = PITStore(cache)
    if not (cache / "prices.parquet").exists():
        emit(f"[data] fetching prices for {ticker} ...")
        prices, actions = fetch_prices_yahoo(ticker, start, today.isoformat())
        if prices.empty:
            raise RuntimeError(f"no price data for {ticker!r} (check the symbol).")
        store.write_prices(prices)
        if not actions.empty:
            store.write_corp_actions(actions)
        store.write_constituents(pd.DataFrame({"symbol": [ticker],
                                               "start_date": [prices["date"].min()],
                                               "end_date": [pd.NaT]}))
        if cfg.edgar_user_agent:
            try:
                cik = fetch_cik_map(cfg.edgar_user_agent).get(ticker)
                if cik:
                    emit(f"[edgar] fetching filings for {ticker} ...")
                    filings, form4 = fetch_edgar_for_symbol(
                        ticker, cik, cfg.edgar_user_agent, since_days=90, periodic_text=2)
                    if filings:
                        store.write_filings(pd.DataFrame(filings))
                    if form4:
                        store.write_form4(pd.DataFrame(form4))
                else:
                    emit(f"[edgar] no CIK for {ticker}; skipping filings.")
            except Exception as exc:
                emit(f"[edgar] skipped: {exc}")
    else:
        emit(f"[data] using cached data for {ticker}.")
    sector = next((etf for etf, tk in LIVE_UNIVERSE.items() if ticker in tk), "SPY")
    return store, {ticker: sector}


def _assess_technicals(t: dict) -> list[str]:
    if not t.get("available"):
        return ["  (insufficient price history)"]
    out = [f"  price: ${t.get('price')}"]
    out.append("  trend: above 200-DMA (uptrend) [GOOD]" if t.get("above_200dma")
               else "  trend: below 200-DMA (downtrend) [BAD]")
    m = t.get("momentum_6mo_pct")
    if m is not None:
        tag = "[GOOD]" if m > 10 else "[BAD]" if m < -10 else "[neutral]"
        out.append(f"  6-month momentum: {m:+.0f}% {tag}")
    p = t.get("pct_below_52wk_high")
    if p is not None:
        tag = "[GOOD: near highs]" if p > -10 else "[BAD: far below]" if p < -30 else "[neutral]"
        out.append(f"  vs 52-week high: {p:+.0f}% {tag}")
    rsi = t.get("rsi14")
    if rsi is not None:
        tag = "[overbought]" if rsi > 70 else "[oversold]" if rsi < 30 else "[neutral]"
        out.append(f"  RSI(14): {rsi:.0f} {tag}")
    if t.get("atr_pct_of_price") is not None:
        out.append(f"  volatility: ATR is {t['atr_pct_of_price']:.1f}% of price")
    return out


def _assess_filings(f: dict, insider: dict) -> list[str]:
    out = []
    if f.get("available"):
        lp = f.get("latest_periodic", {})
        if lp:
            out.append(f"  latest report: {lp.get('form')} filed {lp.get('date')}")
        if f.get("text_change_vs_prior_pct") is not None:
            out.append(f"  filing-text change vs prior {lp.get('form', '')}: "
                       f"{f['text_change_vs_prior_pct']:+.0f}% "
                       "[large change can flag new risks]")
        out.append(f"  recent 8-K events: {f.get('recent_8k_count', 0)} "
                   f"(last {f.get('last_8k_date')})")
    else:
        out.append("  no recent filings available")
    buys = (insider or {}).get("recent_purchases", [])
    if buys:
        tot = sum(b.get("value", 0) for b in buys)
        out.append(f"  insider buying: {len(buys)} purchase(s), ~${tot:,.0f} "
                   "[GOOD: informed buying]")
    else:
        out.append("  insider buying: none recently [neutral]")
    return out


def _analysis_summary(log: Emit, symbol: str, evidence: dict, transcript: dict,
                      rec: dict | None, equity: float) -> None:
    """Readable scorecard: technicals, filings, what's good/bad, the agents'
    reasoning, the buy/no-buy verdict, and the trade plan if it's a buy."""
    tech = _step_out(transcript, "technical_analyst")
    fund = _step_out(transcript, "fundamental_analyst")
    hyp = _step_out(transcript, "hypothesis")
    crit = _step_out(transcript, "skeptic")
    pm = _step_out(transcript, "portfolio_manager")
    action = pm.get("action", "pass")
    verdict = "BUY" if action in {"enter", "adjust"} else "DO NOT BUY (PASS)"
    conv = pm.get("final_conviction")
    conv_s = f"  (conviction {conv:.2f})" if isinstance(conv, (int, float)) else ""

    def _panel(label: str, r: dict) -> None:
        if not r:
            return
        log(f"  {label} analyst: {str(r.get('stance', '?')).upper()} "
            f"(score {r.get('score', '?')})")
        if r.get("assessment"):
            log(f"    {_sent(r['assessment'], 400)}")
        for g in (r.get("positives") or [])[:4]:
            log(f"    + {g}")
        for b in (r.get("concerns") or [])[:4]:
            log(f"    - {b}")

    log(f"\n{'=' * 60}")
    log(f"ANALYSIS: {symbol}   ->   VERDICT: {verdict}{conv_s}")
    log("=" * 60)
    log("Technical picture:")
    for ln in _assess_technicals(evidence.get("technicals", {})):
        log(ln)
    log("Filings & insider activity:")
    for ln in _assess_filings(evidence.get("filings", {}), evidence.get("insider", {})):
        log(ln)
    if tech or fund:
        log("Analyst reads:")
        _panel("Technical", tech)
        _panel("Fundamental", fund)
    log("Why this verdict:")
    if hyp.get("decision") == "propose" and str(hyp.get("mechanism", "")).strip() not in ("", "None"):
        log(f"  bull thesis: {_sent(hyp.get('mechanism'), 900)}")
    else:
        log("  bull thesis: the strategist declined to propose a thesis "
            "(the bull case was not strong enough to commit to).")
    log(f"  main risk (skeptic): {_sent(crit.get('strongest', '-'), 400)} "
        f"(verdict: {crit.get('verdict', '-')})")
    log(f"  decision rationale (PM): {_sent(pm.get('decisive_factor', '-'), 900)}")
    if action not in {"enter", "adjust"}:
        log("  => PASS is the system's disciplined default: it recommends a BUY only "
            "when a real edge survives the bear case. The positives above did not "
            "outweigh the risks here.")
    if rec:
        e, s, tg = rec["entry"], rec["stop"], rec["target"]
        sp = f"{(s / e - 1) * 100:+.1f}%" if (e and s) else "?"
        tp = f"{(tg / e - 1) * 100:+.1f}%" if (e and tg) else "?"
        rr = f"{abs((tg - e) / (e - s)):.1f}" if (e and s and tg and e != s) else "?"
        log("Trade plan (advisory):")
        log(f"  entry ~${e}   stop ${s} ({sp})   target ${tg} ({tp})   reward/risk {rr}:1")
        log(f"  hold ~{rec['hold_days']} sessions  ->  exit by {rec['exit_by']}")
        log(f"  size @ ${equity:,.0f}: {rec['shares_at_ref_equity']} shares (1% account risk)")


def run_recommendations(cfg: AppConfig, emit: Emit) -> dict:
    """Advisory mode: run the full AI-agent pipeline (specialists -> confluence ->
    Hypothesis -> Skeptic -> rebuttal -> Portfolio Manager) and output RECOMMENDED
    entries/exits with the agents' FULL reasoning shown. NO orders are placed.

    If cfg.ticker is set, analyze just that one stock; otherwise scan the universe.
    Ungated and read-only; uses the real LLM when Anthropic credits are present.
    """
    import dataclasses
    import datetime

    from harness.data import calendar as cal
    from harness.data.loader import available_at_for_session
    from harness.signals import ALL_FREE_EDGES
    from system.data_plane.indicators import last_atr
    from system.risk.governor import GovernorContext, RiskGovernor
    from system.run_live import PaperTradingEngine
    from system.config import DEFAULT_CONFIG

    cfg.apply_to_env()
    with _run_logger(emit, "recommendations") as (log, _path):
        client, real_llm = _resolve_client(cfg, log)
        verbose = bool(cfg.verbose_agents)
        ticker = (cfg.ticker or "").strip().upper()
        log("[mode] RECOMMENDATIONS ONLY - the AI agents analyze and suggest; "
            "NO orders are placed.")
        if real_llm:
            log("[mode] using the real LLM agents (spends a bounded amount of credits).")

        if cfg.data_source == "live":
            today = datetime.date.today().isoformat()
            if str(cfg.end_date) < today:
                cfg = dataclasses.replace(cfg, end_date=today)

        recs, considered = [], []
        with _redirect(log):
            if ticker:
                store, sector_map = _build_ticker_store(cfg, ticker, log)
            else:
                store, sector_map = _build_store(cfg, log)
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity),
                                        edges=list(ALL_FREE_EDGES))   # ungated
            session = engine._last_session()
            T = available_at_for_session(session)
            gov = RiskGovernor(DEFAULT_CONFIG)
            mode = "real LLM" if real_llm else "deterministic"

            if ticker:
                log(f"\n[analyze] {ticker} | session {session.date()} | agents: {mode}")
                cand, _decision, transcript = engine.orchestrator.deliberate_symbol(T, ticker)
                items = [(cand, transcript)]
            else:
                log(f"\n[scan] session {session.date()} | {len(engine.universe)} names | "
                    f"agents: {mode}")
                cycle = engine.orchestrator.run_cycle(T)
                items = [(c, cycle.deliberation.get(c.symbol, {})) for c in cycle.candidates]
                if not items:
                    log("[scan] confluence surfaced no candidates "
                        "(set a Ticker to force analysis of a specific name).")

            for cand, transcript in items:
                _print_transcript(log, cand.symbol, transcript, verbose)
                hyp = _step_out(transcript, "hypothesis")
                crit = _step_out(transcript, "skeptic")
                pm = _step_out(transcript, "portfolio_manager")
                considered.append((cand.symbol, pm.get("action", "pass"),
                                   pm.get("decisive_factor", "")))
                rec = None
                if pm.get("action") in {"enter", "adjust"}:
                    px = engine.panels.get(cand.symbol)
                    if px is not None and session in px.index:
                        ref = float(px.loc[session, "close"])
                        atr = last_atr(px.loc[px.index <= session].reset_index())
                        ctx = GovernorContext(equity=float(cfg.starting_equity),
                                              reference_price=ref,
                                              atr_value=atr if atr == atr else 0.0,
                                              sector=sector_map.get(cand.symbol, "?"))
                        ticket = gov.evaluate(cand.symbol, "enter", ctx)
                        hold = int(hyp.get("expected_hold_days", 10) or 10)
                        sessions = cal.sessions(session,
                                                session + pd.Timedelta(days=hold * 2 + 14))
                        exit_on = sessions[min(hold, len(sessions) - 1)].date().isoformat()
                        rec = {
                            "symbol": cand.symbol, "families": cand.families,
                            "entry": round(ref, 2),
                            "stop": round(float(ticket.stop), 2) if ticket.stop else None,
                            "target": round(float(ticket.target), 2) if ticket.target else None,
                            "shares_at_ref_equity": ticket.shares if ticket.approved else 0,
                            "hold_days": hold, "exit_by": exit_on,
                            "conviction": round(float(pm.get("final_conviction", 0)), 2),
                            "thesis": hyp.get("mechanism", ""),
                            "skeptic": crit.get("verdict", "?"),
                            "decisive_factor": pm.get("decisive_factor", ""),
                        }
                        recs.append(rec)
                _analysis_summary(log, cand.symbol, transcript.get("evidence", {}),
                                  transcript, rec, float(cfg.starting_equity))

        # --- recommendation summary ---
        log("\n" + "=" * 60)
        log(f"TRADE RECOMMENDATIONS  ({session.date()})  -- advisory only, no orders")
        log("=" * 60)
        if not recs:
            log("No BUY recommendation (the agents' consensus was to PASS).")
        for r in recs:
            log(f"\n  RECOMMEND BUY {r['symbol']}  (families {r['families']}, "
                f"conviction {r['conviction']})")
            log(f"    entry ~{r['entry']}   stop {r['stop']}   target {r['target']}")
            log(f"    hold ~{r['hold_days']} sessions  ->  exit by {r['exit_by']}")
            log(f"    size @ ${float(cfg.starting_equity):,.0f}: {r['shares_at_ref_equity']} sh (1% risk)")
            log(f"    thesis: {r['thesis'][:240]}")
            log(f"    skeptic verdict: {r['skeptic']}   |   decisive: {r['decisive_factor'][:140]}")
        if considered:
            log("\n  consensus per name:")
            for sym, act, why in considered:
                log(f"    {sym}: {act.upper()} - {_sent(why, 280)}")
        calls = getattr(client, "calls", 0)
        if real_llm:
            log(f"\n[cost] LLM calls this run: {calls}")
        log(f"\n[done] {len(recs)} recommendation(s).")
    return {"session": str(session.date()), "recommendations": recs}


def run_momentum_trade(cfg: AppConfig, emit: Emit) -> dict:
    """Simple momentum swing on Alpaca: each run (1) exits any tracked position
    held to its stipulated date, then (2) enters the strongest-momentum name if
    there is capacity, risk-sized with a protective stop. The time exit is the
    primary exit; the stop covers downside. Run daily (or schedule) so exits fire.

    Note: momentum is NOT a validation-passed edge here; this is a deliberate,
    mechanical strategy, not a statistically blessed one.
    """
    import datetime

    from harness.data import calendar as cal
    from harness.data.loader import LiveLoader, available_at_for_session, live_symbols
    from harness.data.pit_store import PITStore
    from harness.signals import Edge08Momentum
    from system.config import DEFAULT_CONFIG
    from system.data_plane.indicators import last_atr
    from system.risk.governor import GovernorContext, RiskGovernor
    from app.momentum import load_positions, save_positions

    cfg.apply_to_env()
    with _run_logger(emit, "momentum") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[error] Alpaca key id + secret required (Configuration tab).")
            return {}
        try:
            broker = _alpaca_broker(cfg)
            ok, reason = broker.is_tradable()
            log(f"[alpaca] {cfg.alpaca_env.upper()} account: {reason}")
            acct = broker.account()
            equity = float(acct.get("equity") or acct.get("cash") or cfg.starting_equity)
        except Exception as exc:
            log(f"[error] Alpaca not usable: {exc}")
            return {}
        if not ok:
            log("[error] account not tradable; aborting.")
            return {}
        if cfg.alpaca_env == "live":
            log("[warn] LIVE (real-money) momentum trading.")

        # Prices-only live store through today (momentum needs ~1y of history).
        syms = live_symbols(int(cfg.n_symbols))
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=500)).isoformat()
        cache = CONFIG_DIR / "data_store" / f"momentum_{len(syms)}_{today.isoformat()}"
        store = PITStore(cache)
        loader = LiveLoader(store, symbols=syms, start=start, end=today.isoformat(), emit=log)
        with _redirect(log):
            if not (cache / "prices.parquet").exists():
                log(f"[data] fetching prices for {len(syms)} symbols ...")
                loader.load_all()
        sector_map = loader.sector_map()
        last = pd.to_datetime(store.read_table("prices")["date"]).max()
        view = store.as_of(available_at_for_session(last))
        d_today = last.date().isoformat()

        env = cfg.alpaca_env
        state = load_positions()
        tracked = state.get(env, {})
        live_pos = broker.positions()

        # --- 1) EXITS ---
        log("\n[exits] reviewing tracked positions ...")
        for sym in list(tracked):
            info = tracked[sym]
            if sym not in live_pos:
                log(f"  {sym}: already closed at broker (stop hit / manual). Untracking.")
                tracked.pop(sym)
            elif d_today >= info["exit_on"]:
                log(f"  {sym}: held to {d_today} >= exit date {info['exit_on']} -> close at market.")
                broker.close_position(sym)
                tracked.pop(sym)
            else:
                log(f"  {sym}: holding (entered {info['entry_date']}, exit on {info['exit_on']}).")
        if not tracked:
            log("  (no open momentum positions)")

        # --- 2) ENTRY ---
        result_entry = None
        if len(tracked) < int(cfg.momentum_max_positions):
            ranked = []
            for s in view.universe():
                sc = Edge08Momentum().score(view, s, last)
                if sc["raw_score"] > 0:
                    ranked.append((s, sc["raw_score"], sc.get("evidence", {})))
            ranked.sort(key=lambda x: x[1], reverse=True)
            log("\n[scan] top momentum names:")
            for s, sc, ev in ranked[:5]:
                log(f"  {s}: score={sc:.3f}  {ev}")

            pick = next((r for r in ranked if r[0] not in tracked and r[0] not in live_pos), None)
            if pick is None:
                log("[enter] no eligible momentum name to enter.")
            else:
                sym, score, ev = pick
                px = view.prices(sym).set_index("date")
                ref = float(px["close"].iloc[-1])
                atr = last_atr(px.reset_index())
                ctx = GovernorContext(equity=equity, reference_price=ref, atr_value=atr,
                                      sector=sector_map.get(sym, "?"))
                ticket = RiskGovernor(DEFAULT_CONFIG).evaluate(sym, "enter", ctx)
                if not ticket.approved:
                    log(f"[enter] {sym} rejected by Risk Governor: {ticket.reason}")
                else:
                    sessions = cal.sessions(last, last + pd.Timedelta(
                        days=int(cfg.momentum_hold_days) * 2 + 14))
                    hd = int(cfg.momentum_hold_days)
                    exit_on = sessions[min(hd, len(sessions) - 1)].date().isoformat()
                    band = 0.005
                    try:
                        o = broker.submit_buy_with_stop(sym, ticket.shares,
                                                        limit=ref * (1 + band), stop=ticket.stop)
                        tracked[sym] = {"entry_date": d_today, "exit_on": exit_on,
                                        "shares": ticket.shares, "entry_price": ref}
                        log(f"\n[enter] {sym} = strongest momentum (score {score:.3f}). "
                            f"BUY {ticket.shares} sh @ ~{ref:.2f}, protective stop "
                            f"{ticket.stop:.2f}; exit on {exit_on} ({hd} sessions). "
                            f"Order id {o.get('id', '?')} status {o.get('status', '?')}.")
                        result_entry = sym
                    except Exception as exc:
                        log(f"[enter] {sym}: order FAILED - {exc}")
        else:
            log(f"\n[enter] already at max momentum positions "
                f"({len(tracked)}/{cfg.momentum_max_positions}); no new entry.")

        state[env] = tracked
        save_positions(state)
        log(f"\n[done] momentum cycle complete. Open positions: {sorted(tracked)}.")
    return {"entered": result_entry, "open": sorted(tracked)}


def run_reddit_scan(cfg: AppConfig, emit: Emit) -> dict:
    """Analyze Reddit with our model: fetch recent finance-subreddit posts, count
    universe-ticker mentions (buzz), then have the LLM judge sentiment/conviction
    on the most-mentioned names. Needs Reddit API credentials; uses the model when
    Anthropic credits are present (else reports buzz only)."""
    from harness.data import reddit as rd
    from harness.data.loader import live_symbols
    from system.config import DEFAULT_CONFIG

    cfg.apply_to_env()
    with _run_logger(emit, "reddit") as (log, _path):
        if not (cfg.reddit_client_id and cfg.reddit_client_secret):
            log("[error] Reddit API credentials required. Create a (free) app at "
                "https://www.reddit.com/prefs/apps -> 'create app' -> type 'script', "
                "then set Reddit client id + secret (and your Reddit username/password "
                "for a script app) on the Configuration tab and Save.")
            return {}
        ua = f"swing-system:reddit-scan:1.0 (by /u/{cfg.reddit_username or 'anon'})"
        try:
            token = rd.get_token(cfg.reddit_client_id, cfg.reddit_client_secret, ua,
                                 cfg.reddit_username or None, cfg.reddit_password or None)
        except Exception as exc:
            log(f"[error] {exc}")
            return {}
        log("[reddit] authenticated; fetching posts ...")
        posts = rd.fetch_posts(token, ua, limit=100)
        log(f"[reddit] {len(posts)} posts across {len(rd.DEFAULT_SUBREDDITS)} subreddits.")

        universe = live_symbols(int(cfg.n_symbols))
        mentions = rd.extract_mentions(posts, universe)
        if not mentions:
            log("[reddit] no universe tickers mentioned in the current posts.")
            return {"mentions": {}}
        ranked = sorted(mentions.items(),
                        key=lambda kv: (kv[1]["count"], kv[1]["comments"]), reverse=True)
        log("\n[buzz] universe tickers mentioned (by post count):")
        for t, e in ranked[:15]:
            log(f"  {t}: {e['count']} posts, {e['score']} upvotes, {e['comments']} comments")

        client, real_llm = _resolve_client(cfg, log)
        topk = ranked[: int(cfg.reddit_top_k)]
        analyzed = []
        log("\n[model] analyzing sentiment on the top mentioned tickers ...")
        for t, e in topk:
            titles = "\n".join(f"- ({p['subreddit']}, {p['score']} up) {p['title']}"
                               for p in e["posts"])
            if not real_llm:
                log(f"  {t}: (LLM off — buzz only; add Anthropic credits for sentiment)")
                analyzed.append({"ticker": t, "count": e["count"], "sentiment": None})
                continue
            try:
                out = client.complete(REDDIT_PROMPT, {"ticker": t, "reddit_posts": titles},
                                      "RedditSentiment", model=DEFAULT_CONFIG.models.framing,
                                      max_tokens=300)
            except Exception as exc:
                log(f"  {t}: model error - {exc}")
                continue
            sent = out.get("sentiment", "?")
            conv = out.get("conviction", "?")
            log(f"  {t}: sentiment={sent} conviction={conv} quality={out.get('quality','?')} "
                f"| {str(out.get('themes',''))[:110]}")
            analyzed.append({"ticker": t, "count": e["count"], **out})

        calls = getattr(client, "calls", 0)
        if real_llm:
            log(f"\n[cost] LLM calls this scan: {calls}")
        log(f"[done] reddit scan: {len(mentions)} tickers mentioned; analyzed top {len(topk)}.")
    return {"ranked": [(t, e["count"]) for t, e in ranked], "analyzed": analyzed}


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

        # A live deliberation is a decision about TODAY: it must use data through
        # the latest session, not a backtest end date. Otherwise it would price
        # orders off stale history (e.g. a 2-year-old close).
        if cfg.data_source == "live":
            import dataclasses
            import datetime
            today = datetime.date.today().isoformat()
            if str(cfg.end_date) < today:
                log(f"[live] deliberation uses data through {today} (latest), not the "
                    f"configured backtest end {cfg.end_date}.")
                cfg = dataclasses.replace(cfg, end_date=today)

        # The validation gate: trade only edges that PASSED (master principle).
        edges = _gated_edges(cfg, log)
        if not edges:
            log("\n[gate] No validated edges to trade. Run the validation harness "
                "first (and pass an edge), or turn off 'Only trade validated edges'. "
                "Nothing to deliberate — the system's default is to do nothing.")
            return {"session": None, "candidates": [], "decisions": [], "llm_calls":
                    getattr(client, "calls", 0)}

        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity),
                                        edges=edges)
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
            for st in cycle.deliberation.get(cand.symbol, {}).get("steps", []):
                log(f"  [{st['agent']}] {_fmt_payload(st['agent'], st['output'])}")

        # Two-key view: what the Risk Governor would actually approve/size today.
        log("\n[two-key] Risk Governor sizing of any ENTER decisions:")
        approved_tickets = []
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
                approved_tickets.append(ticket)
                log(f"  {d.symbol}: APPROVE {ticket.shares} sh @ ~{ticket.entry:.2f} "
                    f"stop {ticket.stop:.2f} target {ticket.target:.2f} "
                    f"(binding cap: {ticket.binding_cap})")
            else:
                log(f"  {d.symbol}: REJECT - {ticket.reason}")
        if not any_enter:
            log("  (no ENTER decisions - nothing to size)")

        # Safety: never route orders off stale data (prices would be wrong).
        import datetime as _dt
        stale_days = (_dt.date.today() - session.date()).days
        if cfg.place_orders and stale_days > 5:
            log(f"\n[orders] BLOCKED: latest data session ({session.date()}) is "
                f"{stale_days} days old. Refusing to place orders on stale prices — "
                "set Data source=live with a current End date and refresh.")
        else:
            # Optionally route the approved orders to Alpaca (the real transaction).
            _maybe_place_orders(cfg, approved_tickets, sector_map, log)

        calls = getattr(client, "calls", 0)
        if real_llm:
            log(f"\n[cost] LLM API calls made this deliberation: {calls} "
                f"(cap {getattr(client, 'max_calls', '?')})")
        log(f"[done] deliberation complete for {session.date()}.")

    return {"session": str(session.date()),
            "candidates": [c.symbol for c in cycle.candidates],
            "decisions": [(d.symbol, d.action) for d in cycle.decisions],
            "llm_calls": getattr(client, "calls", 0)}
