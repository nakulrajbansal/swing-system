"""Run backend: drive the harness / paper engine and stream output to a callback.

GUI-agnostic so it can also run headless (``app.main --selftest``). Any print()
inside the system is redirected to the callback, and key results are emitted
explicitly. Designed to run inside a worker thread.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from app.config import CONFIG_DIR, AppConfig
from app.learning import load_memory, save_memory

Emit = Callable[[str], None]

# Every run is persisted here so it can be inspected afterwards.
LOGS_DIR = CONFIG_DIR / "logs"


class RunStopped(Exception):
    """The user pressed Stop. Raised cooperatively at the next log line —
    every long loop in the system logs constantly (per download chunk, per
    deep-dive, per position), so cancellation lands within one step."""


_STOP = threading.Event()


def request_stop() -> None:
    _STOP.set()


def clear_stop() -> None:
    _STOP.clear()

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
        if _STOP.is_set():
            try:
                f.write("[stopped] run cancelled by the user.\n")
                f.flush()
            except Exception:
                pass
            raise RunStopped()
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
        try:
            log(f"[log] run log saved to: {path}")
        except RunStopped:
            emit(f"[stopped] run cancelled - partial log at: {path}")
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


def run_portfolio_status(cfg: AppConfig, emit: Emit) -> dict:
    """Read the Alpaca account and report live performance of the open positions:
    each name's quantity, cost basis, current price, market value and unrealized
    P&L, the account equity and day change, and the open positions' blended return
    vs SPY over the same window. Read-only; places no orders. Use this to validate
    how the desk's recommendations actually perform."""
    cfg.apply_to_env()
    out = {"positions": [], "equity": None}
    with _run_logger(emit, "portfolio") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[error] Alpaca key id + secret required (Configuration tab).")
            return out
        try:
            broker = _alpaca_broker(cfg)
            acct = broker.account()
            raw = broker._req("GET", "/v2/positions")
        except Exception as exc:
            log(f"[error] could not read Alpaca account: {exc}")
            return out
        env = cfg.alpaca_env.upper()
        equity = float(acct.get("equity") or 0.0)
        last_eq = float(acct.get("last_equity") or equity)
        cash = float(acct.get("cash") or 0.0)
        out["equity"] = equity
        day_chg = equity - last_eq
        day_pct = (day_chg / last_eq * 100) if last_eq else 0.0
        log(f"[alpaca] {env} account  |  equity ${equity:,.0f}  cash ${cash:,.0f}")
        log(f"[alpaca] today: {day_chg:+,.0f} ({day_pct:+.2f}%) vs prior close")

        if not raw:
            log("\n[positions] none open. Place trades (or use the momentum flow) and "
                "re-check here to validate performance.")
            return out
        log("\n" + "=" * 64)
        log(f"OPEN POSITIONS  ({env})")
        log("=" * 64)
        log("  symbol   qty   avg cost    last    mkt value   unreal P&L     %")
        tot_cost = tot_val = tot_pl = 0.0
        rows = []
        for p in raw:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            avg = float(p.get("avg_entry_price", 0))
            last = float(p.get("current_price", 0) or 0)
            mv = float(p.get("market_value", 0) or 0)
            pl = float(p.get("unrealized_pl", 0) or 0)
            plpc = float(p.get("unrealized_plpc", 0) or 0) * 100
            tot_cost += avg * qty
            tot_val += mv
            tot_pl += pl
            rows.append((sym, qty, avg, last, mv, pl, plpc))
            log(f"  {sym:<6} {qty:>5.0f}  {avg:>9.2f}  {last:>7.2f}  {mv:>10,.0f}  "
                f"{pl:>+11,.0f}  {plpc:>+6.1f}")
        tot_pct = (tot_pl / tot_cost * 100) if tot_cost else 0.0
        log("-" * 64)
        log(f"  TOTAL positions cost ${tot_cost:,.0f}  ->  value ${tot_val:,.0f}   "
            f"unrealized {tot_pl:+,.0f} ({tot_pct:+.1f}%)")
        out["positions"] = [{"symbol": r[0], "qty": r[1], "unrealized_pl": r[5],
                             "unrealized_plpc": r[6]} for r in rows]
        out["unrealized_pl"] = tot_pl
        out["unrealized_plpc"] = tot_pct

        winners = [r for r in rows if r[5] > 0]
        log(f"\n[scorecard] {len(winners)}/{len(rows)} positions in profit; "
            f"best {max(rows, key=lambda r: r[6])[0]} "
            f"({max(r[6] for r in rows):+.1f}%), worst "
            f"{min(rows, key=lambda r: r[6])[0]} ({min(r[6] for r in rows):+.1f}%).")

        # Risk guardrails: leverage (gross exposure vs equity) and single-name
        # concentration vs the desk's 25%-per-name cap.
        gross = tot_val / equity if equity else 0.0
        if gross > 1.05:
            log(f"\n[RISK] gross exposure ${tot_val:,.0f} is {gross:.2f}x your equity "
                f"${equity:,.0f} (margin) - at {gross:.1f}x, a 10% adverse move is "
                f"~{gross * 10:.0f}% of the account. The desk's plan is <=1.0x.")
        heavy = [r for r in rows if equity and r[4] / equity > 0.25]
        for r in heavy:
            log(f"[RISK] {r[0]} is {r[4] / equity * 100:.0f}% of equity - above the "
                "system's 25%-per-name cap; consider trimming.")

        # Open (working / unfilled) orders — e.g. a limit set at a stale price.
        try:
            openo = broker._req("GET", "/v2/orders?status=open")
        except Exception:
            openo = []
        if openo:
            log(f"\n[open orders] {len(openo)} working (not yet filled):")
            for o in openo:
                lp = o.get("limit_price") or o.get("stop_price") or "mkt"
                log(f"    {o.get('side', '?').upper()} {o.get('qty', '?')} "
                    f"{o.get('symbol', '?')}  {o.get('type', '?')} @ {lp}  "
                    f"status {o.get('status', '?')}")
            log("    (a limit set at a stale screen price may never fill - re-place at "
                "a current/marketable price, or cancel from your Alpaca dashboard.)")

        log("\n[note] unrealized P&L is gross of any open stop/target. The desk's edge "
            "shows up over many closed trades, not one snapshot - keep re-checking and "
            "let the Learning tab accumulate outcomes.")
    return out


def place_manual_order(cfg: AppConfig, order: dict, emit: Emit) -> dict:
    """Submit ONE order chosen from the recommendations panel. `order` carries
    symbol, qty, order_type ('market'|'limit'), limit_price, stop, target, and
    attach_bracket. Paper by default; live requires the enable_live_trading gate."""
    cfg.apply_to_env()
    with _run_logger(emit, "order") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[error] Alpaca key id + secret required (Configuration tab).")
            return {"ok": False}
        if cfg.alpaca_env == "live" and not cfg.enable_live_trading:
            log("[blocked] environment is LIVE but 'Enable live trading' is OFF. "
                "Refusing to place a real-money order.")
            return {"ok": False}
        sym = order.get("symbol")
        qty = int(order.get("qty") or 0)
        otype = order.get("order_type", "market")
        bracket = bool(order.get("attach_bracket"))
        try:
            broker = _alpaca_broker(cfg)
        except Exception as exc:
            log(f"[error] could not open Alpaca broker: {exc}")
            return {"ok": False, "error": str(exc)}
        # Affordability pre-check: refuse with a useful message instead of
        # letting Alpaca bounce the order with a raw 403.
        ref_px = order.get("limit_price") or order.get("ref_price")
        try:
            acct = broker.account()
            bp = float(acct.get("buying_power") or acct.get("cash") or 0.0)
            log(f"[account] buying power: ${bp:,.0f}")
            if ref_px and qty:
                est = float(ref_px) * qty
                if est > bp:
                    afford = int(bp // float(ref_px))
                    log(f"[blocked] {qty} {sym} @ ~${float(ref_px):,.2f} needs "
                        f"~${est:,.0f} but buying power is ${bp:,.0f}.")
                    log(f"[hint] max affordable qty at this price: {max(afford, 0)} "
                        "shares. Reduce the quantity, free up cash (check open "
                        "orders holding funds), or skip this trade.")
                    return {"ok": False, "error": "insufficient buying power"}
        except Exception:
            pass                                    # pre-check is best-effort
        try:
            o = broker.submit_manual(
                sym, qty, side="buy", order_type=otype,
                limit_price=order.get("limit_price"),
                stop=order.get("stop") if bracket else None,
                target=order.get("target") if bracket else None)
        except Exception as exc:
            log(f"[error] order failed: {exc}")
            if "insufficient buying power" in str(exc).lower() or "40310000" in str(exc):
                log("[hint] the account does not have enough free cash for this "
                    "order. Open (unfilled) orders also reserve buying power - "
                    "cancel stale ones from the Portfolio view or your Alpaca "
                    "dashboard, or reduce the quantity.")
            return {"ok": False, "error": str(exc)}
        if isinstance(o, dict) and o.get("error"):
            log(f"[error] {o['error']}")
            return {"ok": False, "error": o["error"]}
        oid = (o or {}).get("id", "?")
        status = (o or {}).get("status", "?")
        px = f"limit ${order.get('limit_price')}" if otype == "limit" else "market"
        log(f"[order] {cfg.alpaca_env.upper()}: BUY {qty} {sym} ({px}"
            f"{', + stop/target' if bracket else ''}) submitted - id {oid}, status {status}.")
        # Link the order to its recommendation so Learning/Performance show
        # which calls were actually taken.
        from app import reco_ledger
        if reco_ledger.mark_executed(sym, qty):
            log(f"[ledger] {sym} marked EXECUTED on its recommendation - it now "
                "shows with a ● on the Learning tab.")
        log("[order] check the Paper portfolio (P&L) button to see it once filled.")
        return {"ok": True, "id": oid, "status": status}


def _run_curator(mem, client, log: Emit) -> dict:
    """One curator pass over the lesson memory (the AI-native review that
    replaced the manual approve gate). Prints a short self-review report."""
    from app import reco_ledger
    from system.config import DEFAULT_CONFIG
    from system.reflection.curator import curate

    rep = curate(mem, reco_ledger.load(), client=client,
                 model=DEFAULT_CONFIG.models.framing)
    log(f"[curator] self-review: {rep['activated']} lesson(s) activated by "
        f"evidence, {rep['retired']} retired as contradicted, "
        f"{rep['new_patterns']} pattern lesson(s) written "
        f"({rep['pending']} pending more evidence).")
    if rep.get("calibration"):
        log(f"[curator] calibration: {rep['calibration']}")
    return rep


def run_curation(cfg: AppConfig, emit: Emit) -> dict:
    """On-demand self-assessment: the curator grades the desk's own record —
    expectation vs realization, lens performance, conviction calibration —
    activates evidence-backed lessons and retires contradicted ones."""
    from app import reco_ledger
    from system.reflection.curator import assess

    cfg.apply_to_env()
    with _run_logger(emit, "curator") as (log, _path):
        client, _real = _resolve_client(cfg, log)
        mem = load_memory()
        rows = reco_ledger.load()
        scored = [r for r in rows if r.get("status") == "evaluated"]
        log(f"[curator] reviewing the desk's record: {len(scored)} scored "
            f"recommendation(s), {len(mem.entries)} lesson(s) in memory.")
        if not scored:
            opens = sorted(str(r.get("exit_by") or "") for r in rows
                           if r.get("status") == "open" and r.get("exit_by"))
            if opens:
                log(f"[curator] nothing to grade yet: {len(opens)} open "
                    f"recommendation(s); the earliest matures {opens[0]} - the "
                    "self-review gets its first real evidence then.")
            else:
                log("[curator] no recommendations on the ledger yet - run a "
                    "screen or a deep-dive to start the record.")
        stats = assess(reco_ledger.load())["stats"]
        for key, label in (("overall", "overall"), ("hidden_gem", "hidden-gem"),
                           ("core", "core"), ("moat_bullish", "moat-bullish")):
            b = stats.get(key, {})
            if b.get("n"):
                log(f"  {label:<12} n={b['n']:<3} hit {b['win_rate_pct']:.0f}%  "
                    f"avg {b['avg_return_pct']:+.1f}%  "
                    f"stops {b.get('stop_rate_pct', 0):.0f}%"
                    + (f"  avg conviction {b['avg_conviction']:.2f}"
                       if b.get("avg_conviction") else ""))
        rep = _run_curator(mem, client, log)
        save_memory(mem)
        log("[done] curation complete - see the Learning tab for the active "
            "lesson set.")
    return rep


TRADES_CACHE = CONFIG_DIR / "trades.json"


def run_trade_history(cfg: AppConfig, emit: Emit) -> dict:
    """Every trade actually executed in the account (broker FILL activities),
    printed chronologically and cached for the Performance page. Read-only."""
    import json

    cfg.apply_to_env()
    with _run_logger(emit, "trades") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[error] Alpaca key id + secret required (Settings).")
            return {"fills": 0}
        try:
            fills = _alpaca_broker(cfg).fills(200)
        except Exception as exc:
            log(f"[error] could not read trade history: {exc}")
            return {"fills": 0}
        if not fills:
            log("[trades] no executed trades yet.")
            TRADES_CACHE.write_text("[]", encoding="utf-8")
            return {"fills": 0}
        rows = []
        for f in fills:
            rows.append({"when": str(f.get("transaction_time", ""))[:16].replace("T", " "),
                         "side": str(f.get("side", "?")).upper(),
                         "qty": float(f.get("qty", 0) or 0),
                         "symbol": f.get("symbol", "?"),
                         "price": float(f.get("price", 0) or 0)})
        rows.sort(key=lambda r: r["when"])
        log(f"[trades] {len(rows)} fill(s) on the {cfg.alpaca_env.upper()} account "
            "(oldest first):")
        log("  when              side  qty    symbol   price")
        for r in rows:
            log(f"  {r['when']:<17} {r['side']:<5} {r['qty']:>5.0f}  "
                f"{r['symbol']:<7} {r['price']:>9.2f}")
        TRADES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TRADES_CACHE.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        log(f"\n[done] trade history cached - the Performance page shows it under "
            "EXECUTED TRADES.")
    return {"fills": len(rows)}


def _toast(title: str, message: str) -> bool:
    """Best-effort Windows toast notification (no extra dependencies)."""
    import subprocess
    import sys as _sys
    if _sys.platform != "win32":
        return False
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType=WindowsRuntime] > $null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$x=$t.GetElementsByTagName('text');"
        f"$x.Item(0).AppendChild($t.CreateTextNode('{title}')) > $null;"
        f"$x.Item(1).AppendChild($t.CreateTextNode('{message}')) > $null;"
        "$n=[Windows.UI.Notifications.ToastNotification]::new($t);"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'Swing System').Show($n)")
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       capture_output=True, timeout=15)
        return True
    except Exception:
        return False


def run_watch(cfg: AppConfig, emit: Emit) -> dict:
    """Check the watchlist for triggered entries: a watched name reaching its
    pullback window, or breaking out on unusual volume. Fires a Windows toast
    per hit; never places orders. Cheap (only the watched symbols are fetched)
    — schedule it during market hours via the Automation card."""
    import datetime

    from app import watchlist as wl
    from harness.data.loader import fetch_closes_volumes_batch

    cfg.apply_to_env()
    today = datetime.date.today().isoformat()
    with _run_logger(emit, "watch") as (log, _path):
        items = wl.active(today)
        if not items:
            log("[watch] watchlist is empty - screens add WATCH-tier names and "
                "PM pullback calls automatically.")
            return {"hits": [], "watching": 0}
        log(f"[watch] checking {len(items)} watched name(s): "
            + ", ".join(it["symbol"] for it in items))
        start = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
        closes, volumes = fetch_closes_volumes_batch(
            [it["symbol"] for it in items], start, emit=lambda s: None)
        if closes.empty:
            log("[error] no price data for the watchlist (network?).")
            return {"hits": [], "watching": len(items)}
        hits = wl.watch_hits(items, closes, volumes if not volumes.empty else None)
        for h in hits:
            if h["kind"] == "pullback":
                msg = (f"{h['symbol']} entered its entry window: {h['price']} vs "
                       f"pullback target {h['pullback_target']}")
            else:
                rv = f" on {h['rvol']}x volume" if h.get("rvol") else ""
                msg = (f"{h['symbol']} broke out: {h['price']} through "
                       f"{h['breakout_level']}{rv}")
            log(f"[ALERT] {msg}")
            for ln in _wrap(h.get("reason"), "          ", first="        on watch: "):
                log(ln)
            _toast("Swing System - entry trigger", msg)
        if not hits:
            log("[watch] no triggers - all watched names are still waiting.")
        else:
            # A triggered name has served its purpose: hand it to a deep-dive.
            wl.remove({h["symbol"] for h in hits})
            log(f"\n[watch] {len(hits)} trigger(s). These names left the watchlist - "
                "run a deep-dive to re-evaluate them at TODAY'S price before acting.")
        log(f"[done] watch check complete ({len(items)} watched).")
    return {"hits": hits, "watching": len(items)}


def run_position_review(cfg: AppConfig, emit: Emit) -> dict:
    """Manage the OPEN side of the book — the other half of the trade lifecycle.

    For every broker position: show it against its plan (the ledger entry that
    recommended it), CLOSE positions held past their planned exit date (a time
    exit honors the trade's own plan and only reduces risk), and run the
    Guardian on the rest for thesis-breaking developments. Guardian EXIT advice
    auto-closes only when 'Place orders' is ON; otherwise it is advisory.
    Momentum-flow positions are managed by the momentum cycle and only noted."""
    import datetime

    from app import reco_ledger
    from app.momentum import load_positions
    from harness.data.loader import available_at_for_session
    from system.agents.meta import GuardianAgent
    from system.config import DEFAULT_CONFIG
    from system.data_plane.evidence import assemble_evidence

    cfg.apply_to_env()
    out = {"open": 0, "time_exits": [], "guardian_exits": [], "advised": [],
           "protected": []}
    with _run_logger(emit, "review") as (log, _path):
        if not (cfg.alpaca_key_id and cfg.alpaca_secret):
            log("[error] Alpaca key id + secret required (Settings).")
            return out
        try:
            broker = _alpaca_broker(cfg)
            raw = broker._req("GET", "/v2/positions")
        except Exception as exc:
            log(f"[error] could not read Alpaca positions: {exc}")
            return out
        if not raw:
            log("[review] no open positions - nothing to manage.")
            return out
        # Resting SELL orders per symbol: a plan's stop/target only protects
        # the position if real orders are working at the broker.
        sells_by_sym: dict[str, list] = {}
        try:
            for o in broker._req("GET", "/v2/orders?status=open") or []:
                if str(o.get("side")) == "sell":
                    sells_by_sym.setdefault(o.get("symbol"), []).append(o)
        except Exception:
            pass

        client, real_llm = _resolve_client(cfg, log)
        guardian = GuardianAgent(client, DEFAULT_CONFIG.models.framing)
        mem = _load_memory(cfg, log)
        _memb = (len(mem.entries), len(mem.outcomes)) if mem else (0, 0)
        today = datetime.date.today().isoformat()
        momentum_syms = set(load_positions().get(cfg.alpaca_env, {}))
        out["open"] = len(raw)
        log(f"[review] {len(raw)} open position(s) on {cfg.alpaca_env.upper()} - "
            "checking each against its plan ...")

        # Account-level risk first: every ticket is sized per-trade, so only an
        # aggregate check catches a book that quietly margined itself up.
        try:
            acct = broker.account()
            equity = float(acct.get("equity") or 0.0)
            gross = sum(abs(float(p.get("market_value", 0) or 0)) for p in raw)
            if equity > 0:
                ratio = gross / equity
                out["gross_exposure"] = round(ratio, 2)
                log(f"[account] equity ${equity:,.0f} | open positions ${gross:,.0f} "
                    f"| gross exposure {ratio:.2f}x")
                if ratio > 1.05:
                    log(f"[RISK] the book is {ratio:.1f}x LEVERED - a 10% adverse move "
                        f"is ~{ratio * 10:.0f}% of the account. The desk's plan is "
                        "<=1.0x: do not add positions until exits free up capital, "
                        "and consider trimming the lowest-conviction names.")
        except Exception:
            pass

        for p in raw:
            sym = p.get("symbol")
            qty = float(p.get("qty", 0))
            avg = float(p.get("avg_entry_price", 0) or 0)
            cur = float(p.get("current_price", 0) or 0)
            plpc = float(p.get("unrealized_plpc", 0) or 0) * 100
            plan = reco_ledger.open_for(sym)
            held = ((datetime.date.today()
                     - datetime.date.fromisoformat(plan["date"])).days
                    if plan and plan.get("date") else None)
            bar = "-" * 56
            log(f"\n  {sym}: {qty:.0f} sh @ {avg:.2f} -> {cur:.2f}  ({plpc:+.1f}%)")
            log(f"  {bar}")
            if sym in momentum_syms:
                log("  managed by the momentum flow (time exit fires on its own "
                    "cycle) - not touched here.")
                continue
            if not plan:
                log("  no plan on file (not from a screen/deep-dive ticket) - "
                    f"review manually, or Deep-dive {sym} to create a plan.")
                continue
            log(f"  plan: entry ~{plan.get('entry')}  stop {plan.get('stop')}  "
                f"target {plan.get('target')}  exit by {plan.get('exit_by')}  "
                f"(held {held}d)")
            # Backfill the executed link: a broker position matching a plan IS
            # that recommendation taken (covers orders placed before linking
            # existed, and manual fills).
            if not plan.get("executed"):
                reco_ledger.mark_executed(sym, int(qty))
                log("  linked: recommendation marked EXECUTED (now visible with "
                    "a ● on the Learning tab).")

            # ARM MISSING PROTECTION: if no sell orders are resting, the plan's
            # stop/target are wishes, not protection. Placing them only reduces
            # risk, so it happens automatically.
            resting = sells_by_sym.get(sym, [])
            if resting:
                levels = ", ".join(
                    f"{o.get('type', '?')} @ {o.get('stop_price') or o.get('limit_price') or '?'}"
                    for o in resting[:3])
                log(f"  protection resting at the broker: {levels}")
            elif plan.get("stop") or plan.get("target"):
                kind = ("OCO stop+target" if plan.get("stop") and plan.get("target")
                        else "stop" if plan.get("stop") else "target")
                log(f"  [protect] NO exit orders resting at the broker - arming "
                    f"{kind} for {qty:.0f} sh (stop {plan.get('stop')}, "
                    f"target {plan.get('target')}).")
                try:
                    o = broker.submit_exit_orders(sym, int(qty), plan.get("stop"),
                                                  plan.get("target"))
                    if isinstance(o, dict) and o.get("error"):
                        log(f"  [protect] failed: {o['error']}")
                    else:
                        out["protected"].append(sym)
                        log(f"  [protect] exit orders ARMED - id {(o or {}).get('id', '?')} "
                            f"status {(o or {}).get('status', '?')}.")
                except Exception as exc:
                    log(f"  [protect] failed: {exc}")

            # 1) TIME EXIT: past the plan's exit date -> close (the plan's own rule).
            if plan.get("exit_by") and today >= str(plan["exit_by"]):
                log(f"  [exit] {today} >= planned exit {plan['exit_by']} -> "
                    "closing at market (time exit per the trade's own plan).")
                try:
                    broker.close_position(sym)
                except Exception as exc:
                    log(f"  [error] close failed: {exc}")
                    continue
                reco_ledger.mark_closed(sym, cur, today, "time",
                                        entry_price=avg, memory=mem)
                out["time_exits"].append(sym)
                continue

            # 2) GUARDIAN: thesis-breaking developments? (advisory by default)
            ginputs = {"symbol": sym, "thesis": plan.get("thesis", ""),
                       "pnl_pct": round(plpc, 1), "days_held": held,
                       "plan": {k: plan.get(k) for k in
                                ("entry", "stop", "target", "exit_by", "hold_days")}}
            if plan.get("stop") and cur and cur <= float(plan["stop"]):
                ginputs["thesis_broken"] = True
                ginputs["reason"] = (f"price {cur:.2f} is at/below the planned stop "
                                     f"{plan['stop']} - the protective order may not "
                                     "have fired")
            if real_llm:
                try:
                    with _redirect(log):
                        store, _sm = _build_ticker_store(cfg, sym, log)
                        last = pd.to_datetime(store.read_table("prices")["date"]).max()
                        view = store.as_of(available_at_for_session(last))
                        ginputs["evidence"] = assemble_evidence(view, sym)
                except Exception as exc:
                    log(f"  [guardian] evidence fetch skipped ({exc}).")
            try:
                d = guardian.run(ginputs)
            except Exception as exc:
                log(f"  [guardian] failed ({exc}) - defaulting to HOLD.")
                continue
            if d.action == "exit":
                log("  [guardian] EXIT recommended:")
                for ln in _wrap(d.reason, "      "):
                    log(ln)
                if cfg.place_orders:
                    try:
                        broker.close_position(sym)
                        reco_ledger.mark_closed(sym, cur, today, "guardian",
                                                entry_price=avg, memory=mem)
                        out["guardian_exits"].append(sym)
                        log(f"  [exit] closed at market ({plpc:+.1f}% realized) - "
                            "scored and fed to learning (place_orders is ON).")
                    except Exception as exc:
                        log(f"  [error] close failed: {exc}")
                else:
                    out["advised"].append(sym)
                    log("  [advice] advisory only - close manually or turn ON "
                        "'Place approved orders' to let the review close it.")
            else:
                if d.reason:
                    log("  [guardian] HOLD - thesis intact:")
                    for ln in _wrap(d.reason, "      "):
                        log(ln)
                else:
                    log("  [guardian] HOLD - thesis intact.")
                # Winner management: at >=1R unrealized, advise de-risking.
                r_mult = _r_multiple(avg, plan.get("stop"), cur)
                if r_mult >= 1.0:
                    log(f"  [manage] up {r_mult:.1f}R - consider raising the stop "
                        f"to breakeven (~{avg:.2f}): with risk removed, the "
                        "winner can run on the market's money.")

        # CLOSED AT THE BROKER: an executed plan whose position is gone was
        # sold by its resting stop/target (or manually). Score it NOW from the
        # actual fills — learning must not wait for the exit date.
        held_syms = {p.get("symbol") for p in raw}
        gone = [r for r in reco_ledger.load()
                if r.get("status") == "open" and r.get("executed")
                and r["symbol"] not in held_syms
                and r["symbol"] not in momentum_syms]
        if gone:
            sells: dict[str, dict] = {}
            buys: dict[str, dict] = {}
            try:
                for f in broker.fills(200):          # newest first
                    side = str(f.get("side", ""))
                    sym_f = f.get("symbol")
                    if side == "sell":
                        sells.setdefault(sym_f, f)
                    elif side == "buy":
                        buys.setdefault(sym_f, f)
            except Exception as exc:
                log(f"[fills] could not read trade history ({exc}).")
            for r in gone:
                f = sells.get(r["symbol"])
                if not f:
                    continue
                px = float(f.get("price") or 0)
                entry_f = buys.get(r["symbol"], {})
                entry_px = float(entry_f.get("price") or r.get("entry") or 0)
                when = str(f.get("transaction_time", today))[:10] or today
                reason = _infer_exit_reason(px, r.get("stop"), r.get("target"))
                closed = reco_ledger.mark_closed(r["symbol"], px, when, reason,
                                                 entry_price=entry_px or None,
                                                 memory=mem)
                if closed:
                    log(f"\n[closed] {r['symbol']} was sold at the broker: "
                        f"{reason.upper()} fill @ {px:.2f} on {when} "
                        f"({closed['return_pct']:+.1f}% realized) - scored and "
                        "fed to learning.")

        # Grade matured recommendations too (advice learning must not depend on
        # a screen happening to run): fetch closes for just the due symbols.
        due = sorted({r["symbol"] for r in reco_ledger.load()
                      if r.get("status") == "open" and r.get("exit_by")
                      and str(r["exit_by"]) <= today})
        if due:
            try:
                from harness.data.loader import fetch_closes_batch
                start = (datetime.date.today()
                         - datetime.timedelta(days=120)).isoformat()
                closes = fetch_closes_batch(due + ["SPY"], start, emit=lambda s: None)
                ev = reco_ledger.evaluate(closes, today, mem)
                if ev["evaluated"]:
                    log(f"\n[ledger] scored {ev['evaluated']} matured "
                        f"recommendation(s): hit rate {ev['win_rate_pct']:.0f}%, "
                        f"avg {ev['avg_return_pct']:+.1f}% ({ev['open']} still open).")
                    co = ev.get("cohorts", {})
                    gem, core = co.get("hidden_gem", {}), co.get("core", {})
                    if gem.get("n"):
                        log(f"[ledger] lens scoreboard - gems: {gem['n']} scored, "
                            f"hit {gem['win_rate_pct']:.0f}%, "
                            f"avg {gem['avg_return_pct']:+.1f}%  |  core: "
                            f"{core.get('n', 0)} scored, "
                            f"hit {core.get('win_rate_pct', 0):.0f}%, "
                            f"avg {core.get('avg_return_pct', 0):+.1f}%")
                    if mem is not None:
                        _run_curator(mem, client, log)
            except Exception as exc:
                log(f"[ledger] scoring skipped ({type(exc).__name__}: {exc}).")

        _save_memory(cfg, mem, log, _memb)
        log(f"\n[done] review complete: {len(out['time_exits'])} time exit(s), "
            f"{len(out['guardian_exits'])} guardian exit(s), "
            f"{len(out['advised'])} advisory exit(s), "
            f"{len(out['protected'])} position(s) newly protected.")
    return out


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


def _account_snapshot(cfg: AppConfig, log: Emit) -> dict | None:
    """Real Alpaca account equity/buying power, or None (no keys / unreachable).

    Sizing every ticket and suggested portfolio off the REAL account is what
    makes the numbers in the app actionable — a $100k default produced 21-share
    tickets against $2.6k of buying power."""
    if not (cfg.alpaca_key_id and cfg.alpaca_secret):
        return None
    try:
        acct = _alpaca_broker(cfg).account()
        eq = float(acct.get("equity") or 0.0)
        bp = float(acct.get("buying_power") or acct.get("cash") or 0.0)
        if eq > 0:
            return {"equity": eq, "buying_power": bp}
    except Exception as exc:
        log(f"[account] could not read the Alpaca account ({exc}); "
            "sizing from the configured equity instead.")
    return None


def _resolve_equity(cfg: AppConfig, log: Emit) -> tuple[float, float | None]:
    """(sizing equity, buying_power | None) — real account when available."""
    acct = _account_snapshot(cfg, log)
    if acct:
        log(f"[account] sizing from your real {cfg.alpaca_env.upper()} account: "
            f"equity ${acct['equity']:,.0f}, buying power ${acct['buying_power']:,.0f}.")
        if acct["buying_power"] < 0.05 * acct["equity"]:
            log("[RISK] buying power is nearly exhausted - the book is fully "
                "deployed (possibly on margin). New tickets will have ~0 "
                "affordable shares; run 'Review exits' to free capital before "
                "adding positions.")
        return acct["equity"], acct["buying_power"]
    log(f"[account] no broker account available - sizing from configured equity "
        f"${float(cfg.starting_equity):,.0f}.")
    return float(cfg.starting_equity), None


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
    # Backfill fundamentals if a price cache predates fundamentals support, so the
    # valuation/growth analysts always have data in the scan path too.
    if store.read_table("fundamentals").empty:
        stocks = store.read_table("constituents")["symbol"].tolist()
        if stocks:
            emit(f"[data] backfilling fundamentals for {len(stocks)} symbols ...")
            try:
                loader._load_fundamentals(stocks)
            except Exception as exc:
                emit(f"[data] fundamentals backfill skipped: {exc}")
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


def _wrap(text, indent: str = "    ", width: int = 100,
          first: str | None = None) -> list[str]:
    """The FULL text as wrapped, indented lines — nothing human-written gets
    truncated; `first` lets the opening line carry a label/bullet prefix."""
    import textwrap
    s = " ".join(str(text or "").split())
    if not s:
        return []
    return textwrap.wrap(s, width=width,
                         initial_indent=first if first is not None else indent,
                         subsequent_indent=indent)


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


def _infer_exit_reason(exit_price, stop, target) -> str:
    """Why a broker-side sell most plausibly fired: proximity to the plan's
    levels (within 1.5%) names the trigger; anything else reads as manual."""
    try:
        px = float(exit_price)
    except (TypeError, ValueError):
        return "manual"
    try:
        if target and abs(px / float(target) - 1) <= 0.015:
            return "target"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        if stop and px <= float(stop) * 1.015:
            return "stop"
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return "manual"


def _r_multiple(entry_price, stop, current) -> float:
    """Unrealized gain in R (multiples of the planned per-share risk). At +1R a
    winner has earned the right to a risk-free ride: stop to breakeven."""
    try:
        risk = float(entry_price) - float(stop)
        if risk <= 0:
            return 0.0
        return (float(current) - float(entry_price)) / risk
    except (TypeError, ValueError):
        return 0.0


def _staleness_note(session, log: Emit) -> None:
    """Note when the latest available session lags today by 2+ trading days —
    the analysis is still valid, but prices/levels are not this morning's."""
    import datetime

    try:
        behind = len(pd.bdate_range(pd.Timestamp(session).normalize(),
                                    pd.Timestamp(datetime.date.today()))) - 1
        if behind >= 2:
            log(f"[note] latest available session is {pd.Timestamp(session).date()} "
                f"({behind} trading days behind today) - the data vendor lags at "
                "this hour; entry/stop levels reflect that close, not the "
                "current price.")
    except Exception:
        pass


def _pullback_entry(action: str, pm_entry, ref: float):
    """The PM's pullback limit, when actionable. An 'adjust' decision means the
    thesis is sound but the entry is extended — the PM's suggested entry is the
    most actionable number in the deliberation, so surface it on the ticket
    instead of burying it in prose. Sane-range guarded; None otherwise."""
    try:
        e = float(pm_entry)
    except (TypeError, ValueError):
        return None
    if action == "adjust" and ref > 0 and ref * 0.70 <= e <= ref * 0.995:
        return round(e, 2)
    return None


def _load_memory(cfg: AppConfig, log: Emit):
    """Load the cross-run learning memory if learning is enabled (else None)."""
    if not getattr(cfg, "learn_from_runs", True):
        return None
    mem = load_memory()
    n_les = len(mem.entries)
    n_out = len(mem.outcomes)
    if n_les or n_out:
        log(f"[memory] loaded {n_les} lesson(s) and {n_out} past outcome(s) "
            "to inform the agents.")
    else:
        log("[memory] no prior lessons yet; the desk will learn from trades it closes.")
    return mem


def _save_memory(cfg: AppConfig, mem, log: Emit, before: tuple[int, int]):
    if mem is None:
        return
    added = (len(mem.entries) - before[0], len(mem.outcomes) - before[1])
    if added != (0, 0):
        path = save_memory(mem)
        log(f"[memory] learned {added[1]} new outcome(s)/{added[0]} lesson(s); "
            f"saved to {path}")


def _screen_universe(cfg: AppConfig):
    """(symbols, sector_map, label, key) for the configured screen index."""
    from harness.data.sp500 import screen_universe as sp_u, sector_of
    from harness.data import nasdaq100 as nq
    from harness.data import midsmall as ms
    idx = (getattr(cfg, "screen_index", "sp500") or "sp500").lower()
    cap = cfg.screen_universe or None
    if idx == "qqq":
        return nq.screen_universe(cap), sector_of(), "Nasdaq-100 (QQQ)", "qqq"
    if idx in {"sp400", "sp600", "midsmall"}:
        labels = {"sp400": "S&P 400 (mid-cap)", "sp600": "S&P 600 (small-cap)",
                  "midsmall": "S&P 400+600 (mid/small-cap)"}
        syms, sect = ms.screen_universe(idx, cap)
        return syms, sect, labels[idx], idx
    if idx == "broad":
        # Everything: S&P 500 + 400 + 600 — the full discovery sweep.
        syms_ms, sect_ms = ms.screen_universe("midsmall", None)
        syms = sorted(dict.fromkeys(sp_u(None) + syms_ms))
        sect = {**sect_ms, **sector_of()}
        if cap:
            syms = syms[:cap]
        return syms, sect, "S&P 1500 (broad)", "broad"
    return sp_u(cap), sector_of(), "S&P 500", "sp500"


def _macro_context(cfg: AppConfig, client, log: Emit):
    """Macro snapshot + a single shared MacroAnalyst read for this run (snapshot
    cached daily). Returns (snapshot, macro_read_dict) or (None, None)."""
    import dataclasses
    import datetime
    import json

    from harness.data.macro import fetch_macro_snapshot
    from system.agents.analysts import MacroAnalyst
    from system.config import DEFAULT_CONFIG

    cache = CONFIG_DIR / "data_store" / f"macro_{datetime.date.today().isoformat()}.json"
    snap = None
    if cache.exists():
        try:
            snap = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            snap = None
    if not snap:
        with _redirect(log):
            snap = fetch_macro_snapshot(emit=log)
        if snap.get("available"):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(snap), encoding="utf-8")
    if not snap or not snap.get("available"):
        log("[macro] snapshot unavailable; proceeding without the macro read.")
        return None, None
    log(f"[macro] {snap.get('summary', '')}")
    try:
        read = MacroAnalyst(client, DEFAULT_CONFIG.models.framing).run(
            {"symbol": "_MACRO_", "evidence": {"macro": snap}})
        return snap, dataclasses.asdict(read)
    except Exception as exc:
        log(f"[macro] analyst read failed ({exc}); using the snapshot only.")
        return snap, None


_AGENT_ROLE = {
    "memory": "Recalled lessons & base rates",
    "technical_analyst": "Technical analyst", "fundamental_analyst": "Fundamental analyst",
    "valuation_analyst": "Valuation analyst", "growth_analyst": "Growth analyst",
    "moat_analyst": "Moat & secular-trend analyst",
    "hypothesis": "Strategist (thesis)", "skeptic": "Skeptic (bear case)",
    "hypothesis_rebuttal": "Strategist (rebuttal)",
    "skeptic_rejoinder": "Skeptic (round 2 - contested call)",
    "portfolio_manager": "Portfolio manager (decision)",
}


def _fmt_step_output(out) -> list[str]:
    """An agent's structured output -> readable lines (no raw JSON dumps —
    the deliberation must be scannable by a human, in the app and the logs)."""
    if not isinstance(out, dict):
        return _wrap(out, indent="", width=92)
    W = 92                                                   # caller adds indent
    lines: list[str] = []
    if "stance" in out or "assessment" in out:               # analyst read
        sc = out.get("score")
        head = str(out.get("stance", "?")).upper()
        if isinstance(sc, (int, float)):
            head += f"   (score {sc:.2f})"
        lines.append(head)
        lines += _wrap(out.get("assessment"), indent="", width=W)
        for p in (out.get("positives") or [])[:6]:
            lines += _wrap(p, indent="    ", width=W, first="  + ")
        for c in (out.get("concerns") or [])[:6]:
            lines += _wrap(c, indent="    ", width=W, first="  - ")
    elif "mechanism" in out or "decision" in out:            # hypothesis
        if out.get("decision") != "propose":
            lines.append(f"DECLINE   (conviction {out.get('raw_conviction', 0):.2f})")
        else:
            lines.append(f"PROPOSE   (conviction {out.get('raw_conviction', 0):.2f}, "
                         f"hold ~{out.get('expected_hold_days', '?')} sessions)")
            lines += _wrap(out.get("mechanism"), indent="  ", width=W,
                           first="thesis: ")
            lines += _wrap(out.get("invalidation"), indent="  ", width=W,
                           first="invalidated if: ")
    elif "objections" in out:                                # skeptic
        lines.append(f"verdict: {str(out.get('verdict', '?')).upper()}")
        for o in (out.get("objections") or [])[:8]:
            if isinstance(o, dict):
                lines += _wrap(o.get("detail", ""), indent="        ", width=W,
                               first=f"  - [{float(o.get('severity', 0)):.1f}] ")
    elif "rebuttal" in out:
        lines += _wrap(out["rebuttal"], indent="", width=W)
    elif "stands" in out:                                    # rejoinder (round 2)
        fs = out.get("final_severity")
        sev = f"   (final severity {fs:.1f})" if isinstance(fs, (int, float)) else ""
        lines.append(("objection STANDS" if out.get("stands")
                      else "objection CONCEDED") + sev)
        lines += _wrap(out.get("counter"), indent="", width=W)
    elif "action" in out:                                    # PM / guardian
        act = str(out.get("action", "?")).upper()
        head = act
        if isinstance(out.get("final_conviction"), (int, float)):
            head += f"   (final conviction {out['final_conviction']:.2f})"
        lines.append(head)
        if out.get("entry"):
            lines.append(f"entry ~{out['entry']:.2f}   stop {out.get('stop', 0):.2f}   "
                         f"target {out.get('target', 0):.2f}")
        lines += _wrap(out.get("decisive_factor") or out.get("reason"),
                       indent="  ", width=W, first="decisive: ")
    else:
        lines.append(_short(out, 500))
    return lines


def _print_transcript(log: Emit, symbol: str, transcript: dict, verbose: bool) -> None:
    """Print the deliberation cleanly: the evidence, then each agent's role,
    inputs and output (so it is not a black box) — formatted for scanning."""
    ev = transcript.get("evidence", {})
    log(f"\n   EVIDENCE PROVIDED")
    log(f"   {'-' * 56}")
    log(f"   technicals    {_short(ev.get('technicals', {}), 360)}")
    if ev.get("fundamentals", {}).get("available"):
        log(f"   fundamentals  {_short(ev.get('fundamentals', {}), 460)}")
    fl = dict(ev.get("filings", {}))
    snippet = fl.pop("risk_text_snippet", None)
    log(f"   filings       {_short(fl, 360)}")
    if snippet and verbose:
        log(f"   risk text     {snippet[:300]} ...")
    log(f"   insider       {_short(ev.get('insider', {}), 260)}")
    if ev.get("recent_news"):
        log(f"   news          {ev.get('recent_news')}")
    log(f"\n   AGENT DELIBERATION")
    log(f"   {'-' * 56}")
    for st in transcript.get("steps", []):
        role = _AGENT_ROLE.get(st["agent"], st["agent"])
        log(f"\n   ▸ {role}  ({st.get('model', '-')})")
        if verbose and st.get("system_prompt"):
            log(f"       brief : {_short(st['system_prompt'], 160)}")
            log(f"       input : {_short(st.get('inputs', {}), 300)}")
        for ln in _fmt_step_output(st.get("output", {})):
            log(f"       {ln}")


def _build_ticker_store(cfg: AppConfig, ticker: str, emit: Emit):
    """A focused PIT store for ONE symbol (prices ~2y + EDGAR), for single-ticker
    analysis — even a symbol outside the standard universe."""
    import datetime

    from harness.data.loader import (LIVE_UNIVERSE, fetch_cik_map, fetch_edgar_for_symbol,
                                     fetch_fundamentals_yahoo, fetch_prices_yahoo)
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
    # Fundamentals can be missing on a price cache built before this was added —
    # fetch them if absent so the valuation/growth agents always have data.
    if store.read_table("fundamentals").empty:
        try:
            emit(f"[data] fetching fundamentals for {ticker} ...")
            from harness.data.loader import (available_at_for_session,
                                             fetch_options_positioning)
            last = pd.to_datetime(store.read_table("prices")["date"]).max()
            fund = fetch_fundamentals_yahoo(ticker, available_at=available_at_for_session(last))
            if not fund.empty:
                for k_, v_ in fetch_options_positioning(ticker).items():
                    fund[k_] = v_
                store.write_fundamentals(fund)
        except Exception as exc:
            emit(f"[data] fundamentals skipped: {exc}")
    # Quarterly revenue/margin history (EDGAR XBRL) feeds the moat analyst's
    # TRAJECTORY read — where the business is going, not just where it is.
    if cfg.edgar_user_agent and store.read_table("fundamentals_history").empty:
        try:
            from harness.data.loader import fetch_fundamental_history
            cik = fetch_cik_map(cfg.edgar_user_agent).get(ticker)
            if cik:
                emit(f"[edgar] fetching quarterly revenue/margin history for {ticker} ...")
                hist = fetch_fundamental_history(ticker, cik, cfg.edgar_user_agent)
                if not hist.empty:
                    store.write_fundamentals_history(hist)
        except Exception as exc:
            emit(f"[edgar] quarterly history skipped: {exc}")
    # Recent real headlines so the agents' news evidence is no longer empty
    # on live data.
    if store.read_table("news").empty:
        try:
            from harness.data.loader import fetch_news_yahoo
            n = fetch_news_yahoo(ticker)
            if not n.empty:
                store.write_news(n)
                emit(f"[news] {len(n)} recent headline(s) loaded for {ticker}.")
        except Exception as exc:
            emit(f"[news] skipped: {exc}")
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


def _assess_fundamentals(fu: dict) -> list[str]:
    if not fu.get("available"):
        return ["  (no valuation/growth data available)"]
    v, g = fu.get("valuation", {}), fu.get("growth", {})
    out = []

    def tag_pe(pe):
        return "[cheap]" if 0 < pe < 15 else "[expensive]" if pe > 35 else "[fair]"
    if isinstance(v.get("forward_pe"), (int, float)):
        out.append(f"  forward P/E: {v['forward_pe']:.1f} {tag_pe(v['forward_pe'])}")
    if isinstance(v.get("peg_ratio"), (int, float)):
        peg = v["peg_ratio"]
        out.append(f"  PEG: {peg:.2f} "
                   f"{'[cheap vs growth]' if 0 < peg < 1 else '[expensive vs growth]' if peg > 2 else '[fair]'}")
    if isinstance(v.get("price_to_sales"), (int, float)):
        out.append(f"  price/sales: {v['price_to_sales']:.1f}")
    if isinstance(v.get("analyst_target_price"), (int, float)):
        out.append(f"  analyst target price: ${v['analyst_target_price']:.0f}")
    if isinstance(g.get("revenue_growth_pct"), (int, float)):
        rg = g["revenue_growth_pct"]
        out.append(f"  revenue growth: {rg:+.0f}% "
                   f"{'[GOOD]' if rg > 15 else '[BAD]' if rg < 0 else '[neutral]'}")
    if isinstance(g.get("earnings_growth_pct"), (int, float)):
        eg = g["earnings_growth_pct"]
        out.append(f"  earnings growth: {eg:+.0f}% "
                   f"{'[GOOD]' if eg > 15 else '[BAD]' if eg < 0 else '[neutral]'}")
    if isinstance(g.get("implied_fwd_eps_growth_pct"), (int, float)):
        ig = g["implied_fwd_eps_growth_pct"]
        gtag = "[guided up]" if ig > 5 else "[guided down]" if ig < -5 else "[flat]"
        out.append(f"  forward EPS guidance vs trailing: {ig:+.0f}% {gtag}")
    rev90 = g.get("eps_revision_90d_pct")
    if isinstance(rev90, (int, float)):
        rtag = ("[GOOD: analysts raising]" if rev90 >= 5
                else "[BAD: analysts cutting]" if rev90 <= -5 else "[stable]")
        out.append(f"  est. revisions (fwd EPS, 90d): {rev90:+.0f}% {rtag}")
    eq = (fu or {}).get("earnings_quality", {})
    if eq.get("rating"):
        qtag = {"poor": "[BAD: earnings not converting to cash]",
                "strong": "[GOOD: cash-backed earnings]"}.get(eq["rating"], "[fair]")
        out.append(f"  earnings quality: {eq['rating']} (profit margin "
                   f"{eq.get('profit_margin_pct')}% vs FCF margin "
                   f"{eq.get('fcf_margin_pct')}%) {qtag}")
    if fu.get("analyst_recommendation"):
        out.append(f"  street rating: {fu['analyst_recommendation']}")
    return out or ["  (fundamentals present but sparse)"]


def _assess_moat(fu: dict) -> list[str]:
    m = (fu or {}).get("moat", {}) or {}
    if not fu.get("available") or not any(v is not None for v in m.values()):
        return ["  (no business-quality data available)"]
    out = []
    gm = m.get("gross_margin_pct")
    if isinstance(gm, (int, float)):
        tag = "[pricing power]" if gm >= 55 else "[commodity]" if gm < 25 else "[moderate]"
        out.append(f"  gross margin: {gm:.0f}% {tag}")
    om = m.get("operating_margin_pct")
    if isinstance(om, (int, float)):
        out.append(f"  operating margin: {om:.0f}% "
                   f"{'[strong]' if om >= 25 else '[weak]' if om < 5 else '[ok]'}")
    fcf = m.get("fcf_margin_pct")
    if isinstance(fcf, (int, float)):
        out.append(f"  free-cash-flow margin: {fcf:.0f}% "
                   f"{'[self-funding]' if fcf >= 15 else '[burning cash]' if fcf < 0 else '[ok]'}")
    mc = m.get("market_cap_bn")
    if isinstance(mc, (int, float)):
        size = "mega" if mc >= 200 else "large" if mc >= 10 else "mid/small (more room to re-rate)"
        out.append(f"  market cap: ${mc:,.0f}B [{size}]")
    io = m.get("insider_ownership_pct")
    if isinstance(io, (int, float)) and io >= 3:
        out.append(f"  insider ownership: {io:.0f}% [aligned founders/management]")
    if m.get("industry"):
        out.append(f"  industry: {m['industry']}")
    traj = (fu or {}).get("trajectory", {}) or {}
    if traj.get("available"):
        ys = [q for q in (traj.get("quarters") or []) if q.get("rev_yoy_pct") is not None]
        if ys:
            out.append("  revenue YoY by quarter: " + "   ".join(
                f"{q['q']} {q['rev_yoy_pct']:+.0f}%" for q in ys[-4:]))
        if traj.get("revenue_accelerating") and traj.get("margins_expanding"):
            out.append("  trajectory: revenue ACCELERATING + margins EXPANDING "
                       "[the pre-rally inflection]")
        elif traj.get("revenue_decelerating_2q"):
            out.append("  trajectory: revenue growth decelerating 2 quarters running "
                       "[cooling]")
        elif traj.get("revenue_accelerating"):
            out.append("  trajectory: revenue growth accelerating [improving]")
    return out or ["  (business-quality data sparse)"]


def _analysis_summary(log: Emit, symbol: str, evidence: dict, transcript: dict,
                      rec: dict | None, equity: float) -> None:
    """Readable scorecard: technicals, valuation/growth, filings, what's good/bad,
    the agents' reasoning, the buy/no-buy verdict, and the trade plan if a buy."""
    macro_r = _step_out(transcript, "macro_analyst")
    tech = _step_out(transcript, "technical_analyst")
    fund = _step_out(transcript, "fundamental_analyst")
    val = _step_out(transcript, "valuation_analyst")
    grow = _step_out(transcript, "growth_analyst")
    moat = _step_out(transcript, "moat_analyst")
    recalled = _step_out(transcript, "memory")
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
        sc = r.get("score")
        sc_s = f"{sc:.2f}" if isinstance(sc, (int, float)) else "?"
        log(f"  {label} analyst: {str(r.get('stance', '?')).upper()} (score {sc_s})")
        for ln in _wrap(r.get("assessment"), "    "):
            log(ln)
        for g in (r.get("positives") or [])[:5]:
            for ln in _wrap(g, "      ", first="    + "):
                log(ln)
        for b in (r.get("concerns") or [])[:5]:
            for ln in _wrap(b, "      ", first="    - "):
                log(ln)

    bar = "=" * 64
    log(f"\n{bar}")
    log(f"  {symbol}   {verdict}{conv_s}")
    log(bar)
    dq = evidence.get("technicals", {}).get("data_quality_warning")
    if dq:
        log(f"  !! DATA QUALITY WARNING: {dq}\n")

    def _h(title):
        log(f"\n  {title}")
        log(f"  {'-' * len(title)}")

    _h("TECHNICAL PICTURE")
    for ln in _assess_technicals(evidence.get("technicals", {})):
        log(ln)
    _h("VALUATION & GROWTH")
    for ln in _assess_fundamentals(evidence.get("fundamentals", {})):
        log(ln)
    _h("MOAT & FUTURE GROWTH")
    for ln in _assess_moat(evidence.get("fundamentals", {})):
        log(ln)
    _h("FILINGS & INSIDER")
    for ln in _assess_filings(evidence.get("filings", {}), evidence.get("insider", {})):
        log(ln)
    evts = evidence.get("events", {})
    if evts.get("available"):
        _h("SCHEDULED EVENTS")
        warn = ("  [!] inside the swing window - binary event risk"
                if evts.get("earnings_within_swing_window") else "")
        log(f"  next earnings: {evts.get('next_earnings_date')} "
            f"({evts.get('days_to_earnings')} days away){warn}")
    nsent = evidence.get("news_sentiment", {})
    if nsent.get("available"):
        _h("NEWS TONE")
        ttag = {"bullish": "[GOOD]", "bearish": "[BAD]"}.get(nsent["tone"], "[mixed]")
        log(f"  recent headlines skew {nsent['tone'].upper()} {ttag}  "
            f"({nsent.get('bull_hits')} positive / {nsent.get('bear_hits')} negative "
            f"cues across {nsent.get('headline_count')} headlines)")
        for it in (nsent.get("bearish_items") or [])[:2]:
            log(f"    - flagged: {it}")
    mac = evidence.get("macro", {})
    if mac.get("available"):
        _h("MACRO BACKDROP")
        log(f"  {mac.get('summary', '')}")
    if macro_r or tech or fund or val or grow or moat:
        _h("ANALYST READS")
        _panel("Macro      ", macro_r)
        _panel("Technical  ", tech)
        _panel("Fundamental", fund)
        _panel("Valuation  ", val)
        _panel("Growth     ", grow)
        _panel("Moat/Trend ", moat)
    if recalled:
        stats = recalled.get("setup_stats", {})
        _h("LEARNED FROM PAST TRADES (advisory)")
        if stats.get("count"):
            log(f"    base rate: {stats['count']} past similar trades, win rate "
                f"{stats.get('win_rate_pct', 0):.0f}%, "
                f"avg return {stats.get('avg_return_pct', 0):+.1f}%")
        for les in (recalled.get("lessons") or [])[:3]:
            for ln in _wrap(les, "      ", first="    - "):
                log(ln)
        if not stats.get("count") and not recalled.get("lessons"):
            log("    (no comparable past trades yet)")
    _h("WHY THIS VERDICT")
    if hyp.get("decision") == "propose" and str(hyp.get("mechanism", "")).strip() not in ("", "None"):
        log("    bull thesis:")
        for ln in _wrap(hyp.get("mechanism"), "      "):
            log(ln)
    else:
        log("    bull thesis : the strategist declined to propose a thesis "
            "(the bull case was not strong enough to commit to).")
    log(f"    main risk (skeptic verdict: {crit.get('verdict', '-')}):")
    for ln in _wrap(crit.get("strongest", "-"), "      "):
        log(ln)
    log("    PM decision:")
    for ln in _wrap(pm.get("decisive_factor", "-"), "      "):
        log(ln)
    if action not in {"enter", "adjust"}:
        log("    => PASS is the disciplined default: a BUY needs a real edge that "
            "survives the bear case. The positives did not outweigh the risks here.")
    if rec:
        e, s, tg = rec["entry"], rec["stop"], rec["target"]
        sp = f"{(s / e - 1) * 100:+.1f}%" if (e and s) else "?"
        tp = f"{(tg / e - 1) * 100:+.1f}%" if (e and tg) else "?"
        rr = f"{abs((tg - e) / (e - s)):.1f}" if (e and s and tg and e != s) else "?"
        _h("TRADE PLAN (advisory)")
        log(f"    entry ~${e}   stop ${s} ({sp})   target ${tg} ({tp})   R/R {rr}:1")
        if rec.get("p_win_label"):
            log(f"    win probability: {rec['p_win_label']}")
        if rec.get("suggested_entry"):
            log(f"    PM advises WAITING for a pullback to ~${rec['suggested_entry']} "
                "rather than chasing the current price")
        log(f"    hold ~{rec['hold_days']} sessions  ->  exit by {rec['exit_by']}")
        log(f"    risk-sized: {rec['shares_at_ref_equity']} sh @ ${equity:,.0f} (1% risk)")


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

        equity, buying_power = _resolve_equity(cfg, log)
        # The live "core universe" scan is retired: the screens cover the full
        # indices with the same agent panel and bounded LLM cost.
        if not ticker and cfg.data_source == "live":
            log("[mode] universe scanning now lives in the screens (S&P 500 / "
                "QQQ / 400 / 600 / broad) - run one of those, or enter a ticker "
                "for a single-name deep-dive.")
            return {"session": None, "recommendations": []}

        mem = _load_memory(cfg, log)
        _memb = (len(mem.entries), len(mem.outcomes)) if mem else (0, 0)
        macro, macro_read = _macro_context(cfg, client, log)
        recs, considered = [], []
        with _redirect(log):
            if ticker:
                store, sector_map = _build_ticker_store(cfg, ticker, log)
            else:
                store, sector_map = _build_store(cfg, log)
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity),
                                        edges=list(ALL_FREE_EDGES),   # ungated
                                        memory=mem,
                                        auto_approve_lessons=cfg.auto_approve_lessons,
                                        macro=macro, macro_read=macro_read)
            session = engine._last_session()
            T = available_at_for_session(session)
            gov = RiskGovernor(DEFAULT_CONFIG)
            mode = "real LLM" if real_llm else "deterministic"

            _staleness_note(session, log)
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
                        ctx = GovernorContext(equity=equity,
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
                            "suggested_entry": _pullback_entry(pm.get("action"),
                                                               pm.get("entry"), ref),
                        }
                        if buying_power is not None and ref > 0:
                            rec["affordable_qty"] = int(buying_power // ref)
                        from app import reco_ledger as _rl
                        from system.reflection.calibration import (
                            calibrated_probability, calibration_table, describe)
                        _tbl = calibration_table(_rl.load())
                        rec["p_win"] = calibrated_probability(rec["conviction"], _tbl)
                        rec["p_win_label"] = describe(rec["p_win"], _tbl)
                        recs.append(rec)
                _analysis_summary(log, cand.symbol, transcript.get("evidence", {}),
                                  transcript, rec, equity)

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
            log(f"    size @ ${equity:,.0f}: {r['shares_at_ref_equity']} sh (1% risk)")
            log("    thesis:")
            for ln in _wrap(r["thesis"], "      "):
                log(ln)
            log(f"    skeptic verdict: {r['skeptic']}   |   decisive:")
            for ln in _wrap(r["decisive_factor"], "      "):
                log(ln)
        if considered:
            log("\n  consensus per name:")
            for sym, act, why in considered:
                for ln in _wrap(why, "        ", first=f"    {sym}: {act.upper()} - "):
                    log(ln)
        calls = getattr(client, "calls", 0)
        if real_llm:
            log(f"\n[cost] LLM calls this run: {calls}")
        if recs:
            if cfg.data_source == "synthetic":
                log("[ledger] synthetic run - recommendations NOT recorded (the "
                    "ledger tracks real-market calls only).")
            else:
                from app import reco_ledger
                n = reco_ledger.record(recs, "ticker" if ticker else "scan",
                                       str(session.date()))
                if n:
                    log(f"[ledger] saved {n} recommendation(s) for forward tracking.")
        _save_memory(cfg, mem, log, _memb)
        log(f"\n[done] {len(recs)} recommendation(s).")
    return {"session": str(session.date()), "recommendations": recs}


def _analyze_symbol(cfg: AppConfig, sym: str, client, real_llm: bool, mem, log: Emit,
                    macro=None, macro_read=None, equity: float | None = None,
                    buying_power: float | None = None):
    """Full multi-agent deep-dive on one symbol (data + analysts + trio). Returns
    (rec | None, pm_action, decisive, conviction). Used by the S&P 500 screen for
    each shortlisted name and reused for ad-hoc analysis."""
    from harness.data.loader import available_at_for_session
    from harness.data import calendar as cal
    from harness.signals import ALL_FREE_EDGES
    from system.data_plane.indicators import last_atr
    from system.risk.governor import GovernorContext, RiskGovernor
    from system.run_live import PaperTradingEngine
    from system.config import DEFAULT_CONFIG

    equity = float(equity if equity is not None else cfg.starting_equity)
    store, sector_map = _build_ticker_store(cfg, sym, log)
    engine = PaperTradingEngine(store, sector_map, client=client,
                                starting_equity=float(cfg.starting_equity),
                                edges=list(ALL_FREE_EDGES), memory=mem,
                                auto_approve_lessons=cfg.auto_approve_lessons,
                                macro=macro, macro_read=macro_read)
    session = engine._last_session()
    T = available_at_for_session(session)
    _staleness_note(session, log)
    cand, _decision, transcript = engine.orchestrator.deliberate_symbol(T, sym)
    _print_transcript(log, sym, transcript, bool(cfg.verbose_agents))
    hyp = _step_out(transcript, "hypothesis")
    crit = _step_out(transcript, "skeptic")
    pm = _step_out(transcript, "portfolio_manager")
    moat = _step_out(transcript, "moat_analyst")
    rec = None
    if pm.get("action") in {"enter", "adjust"}:
        px = engine.panels.get(sym)
        if px is not None and session in px.index:
            ref = float(px.loc[session, "close"])
            atr = last_atr(px.loc[px.index <= session].reset_index())
            ctx = GovernorContext(equity=equity, reference_price=ref,
                                  atr_value=atr if atr == atr else 0.0,
                                  sector=sector_map.get(sym, "?"))
            ticket = RiskGovernor(DEFAULT_CONFIG).evaluate(sym, "enter", ctx)
            hold = int(hyp.get("expected_hold_days", 10) or 10)
            sessions = cal.sessions(session, session + pd.Timedelta(days=hold * 2 + 14))
            exit_on = sessions[min(hold, len(sessions) - 1)].date().isoformat()
            rec = {
                "symbol": sym, "families": cand.families, "entry": round(ref, 2),
                "stop": round(float(ticket.stop), 2) if ticket.stop else None,
                "target": round(float(ticket.target), 2) if ticket.target else None,
                "shares_at_ref_equity": ticket.shares if ticket.approved else 0,
                "hold_days": hold, "exit_by": exit_on,
                "conviction": round(float(pm.get("final_conviction", 0)), 2),
                "thesis": hyp.get("mechanism", ""), "skeptic": crit.get("verdict", "?"),
                "decisive_factor": pm.get("decisive_factor", ""),
                "moat_stance": moat.get("stance"),
                "suggested_entry": _pullback_entry(pm.get("action"), pm.get("entry"), ref),
            }
            # What the account could actually buy (caps the GUI ticket default).
            if buying_power is not None and ref > 0:
                rec["affordable_qty"] = int(buying_power // ref)
            # Conviction -> evidence-based win probability (the desk's own record).
            from app import reco_ledger as _rl
            from system.reflection.calibration import (calibrated_probability,
                                                       calibration_table, describe)
            table = calibration_table(_rl.load())
            rec["p_win"] = calibrated_probability(rec["conviction"], table)
            rec["p_win_label"] = describe(rec["p_win"], table)
    _analysis_summary(log, sym, transcript.get("evidence", {}), transcript, rec, equity)
    conv = float(pm.get("final_conviction", 0) or 0)
    return rec, pm.get("action", "pass"), pm.get("decisive_factor", ""), conv


def run_screen(cfg: AppConfig, emit: Emit) -> dict:
    """Screen the S&P 500 to find what to buy, cheaply.

    Funnel: (1) one batched price download for the whole universe; (2) a free,
    deterministic pre-filter ranks every name by relative strength vs the market,
    trend, momentum and overbought/extension; (3) only the top-K names get the
    full, costly multi-agent deep-dive. LLM spend is bounded to K names regardless
    of universe size. Advisory only; no orders are placed."""
    import datetime

    from harness.data.loader import fetch_closes_volumes_batch
    from harness.data.sp500 import SECTOR_ETFS, sector_of
    from app.screen import prescreen, market_regime
    from app import strategy

    cfg.apply_to_env()
    with _run_logger(emit, "screen") as (log, _path):
        client, real_llm = _resolve_client(cfg, log)
        universe, sec_of_map, label, idx_key = _screen_universe(cfg)
        log(f"[mode] SCREEN {label} - free pre-filter over the whole universe, then "
            f"a full AI deep-dive on only the top {cfg.screen_top_k} (advisory; no orders).")
        if real_llm:
            log("[mode] using the real LLM agents for the shortlist (bounded spend).")
        equity, buying_power = _resolve_equity(cfg, log)
        held: set = set()
        if cfg.alpaca_key_id and cfg.alpaca_secret:
            try:
                held = set(_alpaca_broker(cfg).positions())
            except Exception:
                held = set()

        extras = ["SPY"] + list(SECTOR_ETFS.values())     # benchmark + sector ETFs
        fetch_list = list(dict.fromkeys(universe + extras))
        log(f"[universe] {len(universe)} {label} names + benchmark/sector ETFs "
            f"({len(fetch_list)} series).")
        # A suspiciously small universe means the live constituents fetch failed
        # and the static fallback is in use — say so instead of shrinking quietly.
        floors = {"sp500": 480, "qqq": 95, "sp400": 350, "sp600": 500,
                  "midsmall": 850, "broad": 1300}
        if len(universe) < floors.get(idx_key, 0) and not cfg.screen_universe:
            log(f"[universe] WARNING: expected ~{floors[idx_key]}+ names for {label}; "
                "the live constituents fetch likely failed (network/Wikipedia) and a "
                "smaller static fallback is in use. Results still valid, coverage reduced.")

        today = datetime.date.today()
        start = (today - datetime.timedelta(days=420)).isoformat()
        # Cache key includes the index + universe size so different screens never
        # shadow each other (sp500 vs qqq, capped vs full).
        cache = (CONFIG_DIR / "data_store" /
                 f"prefilter_{today.isoformat()}_{idx_key}_{len(fetch_list)}.parquet")
        vcache = cache.with_name(cache.name.replace("prefilter_", "prefilter_vol_"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        closes = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
        volumes = pd.read_parquet(vcache) if vcache.exists() else pd.DataFrame()
        # Refetch if missing or if coverage is well short of what was requested
        # (a stale/partial cache must not silently shrink the screened universe).
        if closes.empty or closes.shape[1] < 0.8 * len(fetch_list) or volumes.empty:
            with _redirect(log):
                closes, volumes = fetch_closes_volumes_batch(
                    fetch_list, start, today.isoformat(), emit=log)
            if not closes.empty:
                closes.to_parquet(cache)
            if not volumes.empty:
                volumes.to_parquet(vcache)
        else:
            log(f"[prefilter] using cached prices ({cache.name}, {closes.shape[1]} series).")
        if closes.empty:
            log("[error] no price data fetched for the pre-filter (check network).")
            return {"recommendations": []}

        # Regime read first, then adapt the factor weights to the regime + what the
        # desk's own past trades say is working.
        mem = _load_memory(cfg, log)
        _memb = (len(mem.entries), len(mem.outcomes)) if mem else (0, 0)

        # Score any matured prior recommendations off the freshly-downloaded prices
        # and feed the outcomes into the learning memory (advice learns too).
        from app import reco_ledger
        ev = reco_ledger.evaluate(closes, today.isoformat(), mem)
        if ev["evaluated"]:
            log(f"[ledger] scored {ev['evaluated']} matured recommendation(s): "
                f"hit rate {ev['win_rate_pct']:.0f}%, avg {ev['avg_return_pct']:+.1f}%. "
                f"({ev['open']} still open)")
            co = ev.get("cohorts", {})
            gem, core = co.get("hidden_gem", {}), co.get("core", {})
            if gem.get("n"):
                log(f"[ledger] lens scoreboard - hidden-gem picks: {gem['n']} scored, "
                    f"hit {gem['win_rate_pct']:.0f}%, avg {gem['avg_return_pct']:+.1f}%  |  "
                    f"core picks: {core.get('n', 0)} scored, "
                    f"hit {core.get('win_rate_pct', 0):.0f}%, "
                    f"avg {core.get('avg_return_pct', 0):+.1f}%")
            mb = co.get("moat_bullish", {})
            if mb.get("n"):
                log(f"[ledger] moat-bullish picks: {mb['n']} scored, "
                    f"hit {mb['win_rate_pct']:.0f}%, avg {mb['avg_return_pct']:+.1f}%")
            if mem is not None:
                _run_curator(mem, client, log)
        regime0 = market_regime(closes)
        # Nightly self-tuning: pick the weight preset currently winning a
        # trailing walk-forward, then stack regime/memory adaptation on top.
        preset_base = None
        if getattr(cfg, "self_tune_weights", True):
            import json as _json
            from app.screen import _metrics as _met
            pcache = (CONFIG_DIR / "data_store" /
                      f"preset_{today.isoformat()}_{idx_key}.json")
            try:
                if pcache.exists():
                    sel = _json.loads(pcache.read_text(encoding="utf-8"))
                else:
                    name, _w, board = strategy.select_preset(closes, _met)
                    sel = {"name": name, "board": board}
                    pcache.write_text(_json.dumps(sel), encoding="utf-8")
                preset_base = strategy.WEIGHT_PRESETS.get(sel["name"])
                if sel["name"] != "base" and preset_base:
                    b = sel["board"]
                    log(f"[tuner] weight preset '{sel['name']}' leads the trailing "
                        f"walk-forward ({b[sel['name']]['total']:+.1f}% vs base "
                        f"{b['base']['total']:+.1f}%) - using it today "
                        "(in-sample selection, advisory).")
            except Exception as exc:
                log(f"[tuner] preset selection skipped ({type(exc).__name__}: {exc}).")
        weights = strategy.factor_weights(regime0, mem, base=preset_base)
        ranked, regime = prescreen(closes, top=max(15, int(cfg.screen_top_k) * 3),
                                   weights=weights, sector_etfs=SECTOR_ETFS,
                                   sector_of=sec_of_map,
                                   volumes=volumes if not volumes.empty else None)

        macro, macro_read = _macro_context(cfg, client, log)
        risk_off = regime.get("available") and not regime.get("above_200dma")
        if macro and macro.get("backdrop") == "hostile":
            risk_off = True
            log("[macro] hostile backdrop -> risk-off: fewer, more selective picks.")
        # SELF-THROTTLE ('shut down if no edge'): when the desk's OWN recent
        # recommendations have been poor or its convictions ran hot, raise the
        # entry bar and cut exposure. Dormant until enough trades are scored;
        # only ever reduces risk.
        from system.reflection.calibration import desk_throttle
        throttle = desk_throttle(reco_ledger.load())
        if throttle["active"]:
            log(f"[throttle] DE-RISKING: {throttle['reason']}. Raising the BUY bar "
                f"(+{throttle['conviction_bump']:.2f} conviction) and capping gross "
                f"exposure to {throttle['gross_scale'] * 100:.0f}% until the record "
                "improves. (The desk stands down when it isn't paying.)")
        dropped = regime.get("dropped_bad_data", 0)
        if dropped:
            log(f"[prefilter] dropped {dropped} name(s) with implausible/corrupt "
                "price data (kept out of the shortlist).")
        illiq = regime.get("dropped_illiquid", 0)
        if illiq:
            log(f"[prefilter] dropped {illiq} name(s) below the liquidity floor "
                "(median dollar volume / price too low to trade cleanly).")
        if regime.get("available"):
            log(f"\n[regime] market is {regime['regime']} (SPY 6mo "
                f"{regime['mom6_pct']:+.0f}%, "
                f"{'above' if regime['above_200dma'] else 'below'} its 200-DMA). "
                "Factor weights adapted accordingly.")
            if risk_off:
                log("[regime] risk-off: deep-diving fewer names, demanding the 200-DMA, "
                    "and the suggested portfolio runs only ~50% invested.")

        # Sector rotation leaderboard.
        sec_rs = regime.get("sector_rs", {})
        if sec_rs:
            lead = sorted(sec_rs.items(), key=lambda kv: kv[1], reverse=True)
            log("\n[sectors] relative strength vs SPY (leaders first):")
            for s, v in lead[:4]:
                log(f"    + {s}: {v * 100:+.0f}%")
            for s, v in lead[-2:]:
                log(f"    - {s}: {v * 100:+.0f}%")

        log(f"\n[leaderboard] top pre-filter names (of {len(closes.columns)} priced):")
        log("  rank  symbol   score   RS_vs_mkt  6mo_mom  vs_high  vs20d  RVOL  RSI  gap   sector")
        for i, m in enumerate(ranked[:15], 1):
            rv = m.get("rvol")
            log(f"  {i:>4}  {m['symbol']:<6}  {m['score']:>5.2f}  "
                f"{m['rs'] * 100:>+8.0f}%  {m['mom6'] * 100:>+6.0f}%  "
                f"{m['dist_high'] * 100:>+6.0f}%  {m.get('ext20', 0) * 100:>+5.0f}%  "
                f"{(f'{rv:.1f}' if isinstance(rv, (int, float)) else '  -'):>4}  "
                f"{m['rsi']:>3.0f}  {m['earnings_gap'] * 100:>+4.0f}  "
                f"{(m.get('sector') or '')[:18]}")
        log("  (vs20d = stretch above the 20-DMA: near 0 is a buyable entry, "
            ">12% is a chase.  RVOL = today's volume vs its 20-day average.)")

        # Regime-conditional shortlist: in risk-off, deep-dive fewer; never waste
        # LLM on negative-score names.
        k = int(cfg.screen_top_k)
        if risk_off:
            k = max(2, k // 2)
        # Top composite scores plus reserved HIDDEN-GEM slots: early-acceleration
        # names a pure momentum ranking would only surface after the rally is
        # consensus. Slot count self-tunes from the lens's REALIZED hit rate.
        cohorts = reco_ledger.cohort_stats()
        gems_n = 0 if risk_off else strategy.gem_slot_count(cohorts)
        gco = cohorts.get("hidden_gem", {})
        if not risk_off and gems_n != 2 and gco.get("n"):
            log(f"[lens] hidden-gem slots tuned to {gems_n} from realized results "
                f"({gco['n']} scored, {gco.get('win_rate_pct', 0):.0f}% hit rate).")
        top = strategy.select_shortlist(ranked, k, gem_slots=gems_n, closes=closes)
        if not top:
            log("\n[deep-dive] no name cleared the pre-filter bar (score > 0) in this "
                "regime; standing down. Default is to do nothing.")
        else:
            log(f"\n[deep-dive] full AI agent panel on the top {len(top)}: "
                f"{', '.join(m['symbol'] + (' (gem)' if m.get('hidden_gem') else '') for m in top)}")
            gems = [m for m in top if m.get("hidden_gem")]
            if gems:
                log("[deep-dive] hidden-gem slots (early acceleration, pre-consensus): "
                    + ", ".join(f"{m['symbol']} (3mo {m['mom3'] * 100:+.0f}% vs 6mo "
                                f"{m['mom6'] * 100:+.0f}%)" for m in gems))

        recs, considered = [], []
        sec_map = sec_of_map
        for m in top:
            sym = m["symbol"]
            gem_tag = "  [HIDDEN GEM - early acceleration]" if m.get("hidden_gem") else ""
            log(f"\n{'#' * 60}\n# DEEP-DIVE: {sym}  (score {m['score']:.2f}, "
                f"RS {m['rs'] * 100:+.0f}%, sector {m.get('sector') or '?'})"
                f"{gem_tag}\n{'#' * 60}")
            try:
                with _redirect(log):
                    rec, action, decisive, conv = _analyze_symbol(
                        cfg, sym, client, real_llm, mem, log, macro, macro_read,
                        equity=equity, buying_power=buying_power)
            except Exception as exc:
                log(f"[deep-dive] {sym} failed: {type(exc).__name__}: {exc}")
                continue
            considered.append({"symbol": sym, "action": action, "decisive": decisive,
                               "conviction": conv, "score": m["score"]})
            if rec:
                rec["sector"] = sec_map.get(sym, "?")
                rec["hidden_gem"] = bool(m.get("hidden_gem"))
                rec["already_held"] = sym in held
                if rec["already_held"]:
                    log(f"[book] note: {sym} is ALREADY an open position - this "
                        "would add to it, not diversify.")
                recs.append(rec)

        # Apply the self-throttle's raised entry bar: when de-risking, only the
        # higher-conviction calls survive (lower ones drop to WATCH).
        if throttle["active"] and throttle["conviction_bump"] > 0:
            bar = 0.55 + throttle["conviction_bump"]
            held_back = [r for r in recs if r["conviction"] < bar]
            recs = [r for r in recs if r["conviction"] >= bar]
            for r in held_back:
                log(f"[throttle] holding back {r['symbol']} (conviction "
                    f"{r['conviction']:.2f} < raised bar {bar:.2f}) - watch, don't buy.")

        log("\n" + "=" * 60)
        log(f"{label} SCREEN RESULTS  ({today})  -- advisory only, no orders")
        log("=" * 60)
        if not recs:
            log("No BUY among the shortlist (the agents passed on all of them).")
        for r in sorted(recs, key=lambda x: x["conviction"], reverse=True):
            pw = f", P(win) {r['p_win'] * 100:.0f}%" if r.get("p_win") else ""
            log(f"\n  RECOMMEND BUY {r['symbol']}  (conviction {r['conviction']}{pw}, "
                f"{r.get('sector', '?')})")
            pb = (f"   << PM: wait for a pullback to ~{r['suggested_entry']}"
                  if r.get("suggested_entry") else "")
            log(f"    entry ~{r['entry']}   stop {r['stop']}   target {r['target']}{pb}")
            log(f"    hold ~{r['hold_days']} sessions  ->  exit by {r['exit_by']}")
            log("    thesis:")
            for ln in _wrap(r["thesis"], "      "):
                log(ln)

        # Portfolio construction: conviction-scaled, capped, regime-budgeted —
        # sized off the REAL account equity when broker keys are present, and
        # further capped by the self-throttle on a cold streak.
        portfolio = strategy.construct_portfolio(
            recs, equity, regime, gross_scale=throttle["gross_scale"])
        if portfolio:
            log(f"\n  SUGGESTED PORTFOLIO (@ ${equity:,.0f} equity, conviction-weighted, "
                f"capped, {'~50% invested - risk-off' if risk_off else 'fully invested'}):")
            for p in portfolio:
                log(f"    {p['symbol']:<6} {p['weight_pct']:>5.1f}%  "
                    f"${p['dollars']:>10,.0f}  {p['shares']:>5} sh   "
                    f"[{p['sector']}]")
            inv = sum(p["weight_pct"] for p in portfolio)
            log(f"    invested {inv:.0f}% / cash {max(0, 100 - inv):.0f}%")

        # Always surface the model's best ideas, ranked by conviction and tiered,
        # so an all-PASS run is still actionable (what to watch, and why not yet).
        log("\n  TOP IDEAS (ranked by the agents' conviction):")
        for c in sorted(considered, key=lambda x: x["conviction"], reverse=True):
            if c["action"] in {"enter", "adjust"} or c["conviction"] >= 0.55:
                tier = "BUY  "
            elif c["conviction"] >= 0.40:
                tier = "WATCH"
            else:
                tier = "PASS "
            log(f"    [{tier}] {c['symbol']:<6} conv {c['conviction']:.2f}")
            for ln in _wrap(c["decisive"], "            "):
                log(ln)
        watch = [c for c in considered if c["action"] not in {"enter", "adjust"}
                 and c["conviction"] >= 0.40]
        if watch and not recs:
            log("\n  No name cleared the BUY bar, but the WATCH names above are close - "
                "they typically need a better entry (a pullback) or one more "
                "confirming signal. Re-run after a dip or set a tighter ticker.")
        if recs:
            n = reco_ledger.record(recs, f"screen-{idx_key}", today.isoformat())
            if n:
                log(f"\n[ledger] saved {n} recommendation(s) to track forward "
                    "performance (scored automatically after the hold window).")

        # WATCHLIST: names whose time hasn't come — WATCH-tier ideas and PM
        # pullback calls persist with trigger levels for the watch check.
        from app import watchlist as wl
        watch_items = []
        by_sym = {m["symbol"]: m for m in top}
        for c in considered:
            m = by_sym.get(c["symbol"], {})
            price = m.get("price")
            if (c["action"] not in {"enter", "adjust"}
                    and 0.40 <= c["conviction"] < 0.55 and price):
                ext = m.get("ext20", 0.0) or 0.0
                dh = m.get("dist_high", 0.0) or 0.0
                watch_items.append({
                    "symbol": c["symbol"], "sector": m.get("sector"),
                    "reason": f"WATCH tier (conviction {c['conviction']:.2f}): "
                              + _sent(c.get("decisive", ""), 300),
                    "pullback_target": round(price / (1 + ext), 2) if ext > 0.02 else None,
                    "breakout_level": round(price / (1 + dh), 2) if dh < -0.02 else None,
                })
        for r in recs:
            if r.get("suggested_entry"):
                watch_items.append({
                    "symbol": r["symbol"], "sector": r.get("sector"),
                    "reason": f"PM advised waiting for a pullback to "
                              f"~{r['suggested_entry']}",
                    "pullback_target": r["suggested_entry"],
                })
        n_watch = wl.add(watch_items, today.isoformat()) if watch_items else 0
        if n_watch:
            log(f"[watchlist] tracking {n_watch} name(s) for entry triggers - "
                "the watch check alerts when price reaches them.")
        if real_llm:
            log(f"\n[cost] LLM calls this screen: {getattr(client, 'calls', 0)}")
        _save_memory(cfg, mem, log, _memb)
        log(f"\n[done] screened {len(closes.columns)} series; deep-dived {len(top)}; "
            f"{len(recs)} BUY recommendation(s).")
    return {"recommendations": recs, "portfolio": portfolio if recs else [],
            "leaderboard": [(m["symbol"], m["score"]) for m in ranked[:15]],
            "regime": {k2: v for k2, v in regime.items() if k2 != "sector_rs"}}


def run_strategy_backtest(cfg: AppConfig, emit: Emit) -> dict:
    """Honest walk-forward backtest of the pre-filter strategy: every hold-period,
    rank the S&P 500 by the composite score using ONLY prior history, buy the
    top-K equal-weight, hold, and compare realized returns to the benchmark
    (CAGR, Sharpe, win rate, drawdown, excess vs SPY). No LLM, no look-ahead."""
    import datetime

    from harness.data.loader import fetch_closes_batch
    from harness.data.sp500 import SECTOR_ETFS
    from app.screen import _metrics
    from app import strategy

    cfg.apply_to_env()
    with _run_logger(emit, "backtest") as (log, _path):
        universe, _sec, label, idx_key = _screen_universe(cfg)
        log(f"[backtest] universe: {label}")
        extras = ["SPY"] + list(SECTOR_ETFS.values())
        fetch_list = list(dict.fromkeys(universe + extras))
        today = datetime.date.today()
        # ~3 years so the walk-forward has many periods.
        start = (today - datetime.timedelta(days=1100)).isoformat()
        cache = (CONFIG_DIR / "data_store" /
                 f"backtest_{idx_key}_{len(fetch_list)}_{today.isoformat()}.parquet")
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            log(f"[data] using cached prices ({cache.name}).")
            closes = pd.read_parquet(cache)
        else:
            log(f"[data] downloading ~3y prices for {len(fetch_list)} series "
                "(first run is slow) ...")
            with _redirect(log):
                closes = fetch_closes_batch(fetch_list, start, today.isoformat(), emit=log)
            if not closes.empty:
                closes.to_parquet(cache)
        if closes.empty:
            log("[error] no price data; cannot backtest.")
            return {}

        hold = int(cfg.momentum_hold_days) or 10
        k = max(int(cfg.screen_top_k), 5)
        weights = strategy.factor_weights(None, None)   # stable base weights
        log(f"[backtest] walk-forward: rank {len(closes.columns)} series, buy top {k} "
            f"equal-weight, hold {hold} sessions, repeat. No look-ahead.")
        with _redirect(log):
            res = strategy.walk_forward_backtest(closes, _metrics, weights,
                                                 top_k=k, hold_days=hold,
                                                 diversify_corr=True)
        if res.get("error"):
            log(f"[backtest] {res['error']}")
            return res
        log("\n" + "=" * 60)
        log("WALK-FORWARD BACKTEST  (strategy vs S&P 500)")
        log("=" * 60)
        log(f"  periods: {res['periods']}  (top {res['top_k']}, hold {res['hold_days']} sessions)")
        log(f"  strategy : total {res['strategy_total_return_pct']:+.1f}%   "
            f"CAGR {res['strategy_cagr_pct']:+.1f}%   Sharpe {res['strategy_sharpe']}")
        log(f"  benchmark: total {res['benchmark_total_return_pct']:+.1f}%   "
            f"CAGR {res['benchmark_cagr_pct']:+.1f}%   Sharpe {res['benchmark_sharpe']}")
        verdict = "BEATS" if res["excess_return_pct"] > 0 else "TRAILS"
        log(f"  --> strategy {verdict} the benchmark by {res['excess_return_pct']:+.1f}% "
            f"total;  win rate {res['win_rate_pct']:.0f}%,  max drawdown "
            f"{res['max_drawdown_pct']:.1f}%")
        if res.get("gem_picks"):
            log(f"  hidden-gem lens (tracked separately): {res['gem_picks']} picks over "
                f"{res['gem_periods']} periods -> total {res['gem_total_return_pct']:+.1f}%, "
                f"win rate {res['gem_win_rate_pct']:.0f}% "
                "(validates the early-acceleration lens before money follows it)")
        log("\n[note] gross of slippage on the pre-filter alone (no agent gate). The "
            "agents + Risk Governor are an additional discipline layer on top.")
        log("[note] the universe is TODAY's index membership, so this backtest has "
            "survivorship bias (delisted/demoted names are missing) - treat the "
            "excess return as an upper bound, not a promise.")
        log("[done] backtest complete.")
    return res


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
    from harness.data.loader import fetch_closes_volumes_batch, fetch_prices_yahoo
    from system.config import DEFAULT_CONFIG
    from system.data_plane.indicators import last_atr
    from system.risk.governor import GovernorContext, RiskGovernor
    from app.momentum import load_positions, save_positions
    from app.screen import prescreen

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

        # The momentum scan covers the SAME universe as the configured screen —
        # the old tiny "core universe" is retired now that screens span the
        # full indices. Batched closes/volumes, cached per day.
        universe, sec_of_map, label, idx_key = _screen_universe(cfg)
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=420)).isoformat()
        fetch_list = list(dict.fromkeys(universe + ["SPY"]))
        cache = (CONFIG_DIR / "data_store" /
                 f"momentum_{today.isoformat()}_{idx_key}_{len(fetch_list)}.parquet")
        vcache = cache.with_name(cache.name.replace("momentum_", "momentum_vol_"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        closes = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
        volumes = pd.read_parquet(vcache) if vcache.exists() else pd.DataFrame()
        if closes.empty or closes.shape[1] < 0.8 * len(fetch_list):
            log(f"[data] downloading prices for {len(fetch_list)} {label} series ...")
            with _redirect(log):
                closes, volumes = fetch_closes_volumes_batch(
                    fetch_list, start, today.isoformat(), emit=log)
            if not closes.empty:
                closes.to_parquet(cache)
            if not volumes.empty:
                volumes.to_parquet(vcache)
        else:
            log(f"[data] using cached prices ({closes.shape[1]} series).")
        if closes.empty:
            log("[error] no price data; aborting.")
            return {}
        log(f"[universe] momentum scan over {label}: {len(universe)} names.")
        last = pd.Timestamp(closes.index.max())
        d_today = last.date().isoformat()
        _staleness_note(last, log)
        days_behind = len(pd.bdate_range(last.normalize(), pd.Timestamp(today))) - 1

        env = cfg.alpaca_env
        state = load_positions()
        tracked = state.get(env, {})
        live_pos = broker.positions()

        mem = _load_memory(cfg, log)
        _memb = (len(mem.entries), len(mem.outcomes)) if mem else (0, 0)

        def _learn_close(sym, info, reason):
            if mem is None:
                return
            from system.agents.llm_client import MockLLMClient
            from system.agents.meta import ReflectionAgent
            from system.reflection.memory import TradeOutcome
            try:
                cur = float(closes[sym].dropna().iloc[-1])
            except Exception:
                return
            entry_px = float(info.get("entry_price", cur) or cur)
            pnl_pct = (cur / entry_px - 1) * 100.0 if entry_px else 0.0
            mem.record_outcome(TradeOutcome("momentum_swing", sym, 0.0,
                                            round(pnl_pct, 2), reason, d_today))
            try:
                les = ReflectionAgent(MockLLMClient(), DEFAULT_CONFIG.models.framing).run(
                    {"trade": {"pnl": pnl_pct, "pnl_pct": pnl_pct, "reason": reason,
                               "symbol": sym, "setup_type": "momentum_swing",
                               "as_of": d_today, "conviction": 0.0}})
                mem.add(les, human_reviewed=cfg.auto_approve_lessons)
            except Exception:
                pass

        # --- 1) EXITS ---
        log("\n[exits] reviewing tracked positions ...")
        for sym in list(tracked):
            info = tracked[sym]
            if sym not in live_pos:
                log(f"  {sym}: already closed at broker (stop hit / manual). Untracking.")
                _learn_close(sym, info, "stop")
                tracked.pop(sym)
                state[env] = tracked
                save_positions(state)            # persist NOW: stop-safe tracker
            elif d_today >= info["exit_on"]:
                log(f"  {sym}: held to {d_today} >= exit date {info['exit_on']} -> close at market.")
                broker.close_position(sym)
                _learn_close(sym, info, "time")
                tracked.pop(sym)
                state[env] = tracked
                save_positions(state)
            else:
                log(f"  {sym}: holding (entered {info['entry_date']}, exit on {info['exit_on']}).")
        if not tracked:
            log("  (no open momentum positions)")

        # --- 2) ENTRY ---
        result_entry = None
        if days_behind >= 2:
            log("\n[enter] price data is stale - not entering on old prices "
                "(exits above still ran).")
        elif len(tracked) < int(cfg.momentum_max_positions):
            rows, _regime = prescreen(closes, top=10 ** 6,
                                      volumes=volumes if not volumes.empty else None,
                                      sector_of=sec_of_map)
            ranked = [m for m in rows if m["above_200dma"] and m["mom6"] > 0]
            ranked.sort(key=lambda m: m["mom6"], reverse=True)
            log("\n[scan] strongest momentum (trend-confirmed, liquid):")
            for m in ranked[:5]:
                log(f"  {m['symbol']}: 6mo {m['mom6'] * 100:+.0f}%  RS "
                    f"{m['rs'] * 100:+.0f}%  vs20d {m.get('ext20', 0) * 100:+.0f}%  "
                    f"[{(m.get('sector') or '?')[:18]}]")

            pick = next((m for m in ranked if m["symbol"] not in tracked
                         and m["symbol"] not in live_pos), None)
            if pick is None:
                log("[enter] no eligible momentum name to enter.")
            else:
                sym, score = pick["symbol"], pick["mom6"]
                px = pd.DataFrame()
                with _redirect(log):
                    try:
                        px, _acts = fetch_prices_yahoo(
                            sym, (today - datetime.timedelta(days=200)).isoformat(),
                            today.isoformat())
                    except Exception as exc:
                        log(f"[enter] {sym}: OHLC fetch failed ({exc}).")
                if px.empty:
                    log(f"[enter] {sym}: no OHLC data - skipping entry.")
                    state[env] = tracked
                    save_positions(state)
                    _save_memory(cfg, mem, log, _memb)
                    return {"entered": None, "open": sorted(tracked)}
                ref = float(px["close"].iloc[-1])
                atr = last_atr(px)
                ctx = GovernorContext(equity=equity, reference_price=ref,
                                      atr_value=atr if atr == atr else 0.0,
                                      sector=sec_of_map.get(sym, "?"))
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
                        state[env] = tracked
                        save_positions(state)    # persist before any log can stop us
                        log(f"\n[enter] {sym} = strongest momentum "
                            f"(6mo {score * 100:+.0f}%). "
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
        _save_memory(cfg, mem, log, _memb)
        log(f"\n[done] momentum cycle complete. Open positions: {sorted(tracked)}.")
    return {"entered": result_entry, "open": sorted(tracked)}


def run_reddit_scan(cfg: AppConfig, emit: Emit) -> dict:
    """Analyze Reddit with our model: fetch recent finance-subreddit posts, count
    universe-ticker mentions (buzz), then have the LLM judge sentiment/conviction
    on the most-mentioned names. Needs Reddit API credentials; uses the model when
    Anthropic credits are present (else reports buzz only)."""
    from harness.data import reddit as rd
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

        # Match mentions against the FULL tradable coverage (S&P 1500 + QQQ) —
        # buzz on a small cap is exactly the early signal worth catching.
        from harness.data import midsmall as ms
        from harness.data import nasdaq100 as nq
        from harness.data.sp500 import sp500_symbols
        ms_syms, _sect = ms.midsmall_universe()
        universe = sorted(set(sp500_symbols()) | set(nq.nasdaq100_symbols())
                          | set(ms_syms))
        log(f"[reddit] matching against {len(universe)} names (S&P 1500 + QQQ).")
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
        mem = _load_memory(cfg, log)
        _memb = (len(mem.entries), len(mem.outcomes)) if mem else (0, 0)
        t0 = time.time()
        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            log("[engine] starting paper-trading cycles (paper-only) ...")
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity),
                                        memory=mem,
                                        auto_approve_lessons=cfg.auto_approve_lessons)
            result = engine.run()
        log(f"\n[done] paper run finished in {time.time() - t0:.1f}s")
        _save_memory(cfg, mem, log, _memb)
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

        mem = _load_memory(cfg, log)
        macro, macro_read = _macro_context(cfg, client, log)
        with _redirect(log):
            store, sector_map = _build_store(cfg, log)
            engine = PaperTradingEngine(store, sector_map, client=client,
                                        starting_equity=float(cfg.starting_equity),
                                        edges=edges, memory=mem,
                                        auto_approve_lessons=cfg.auto_approve_lessons,
                                        macro=macro, macro_read=macro_read)
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

        # Two-key view: what the Risk Governor would actually approve/size today,
        # off the REAL account when broker keys are present.
        equity, _bp = _resolve_equity(cfg, log)
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
            ctx = GovernorContext(equity=equity, reference_price=ref,
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
