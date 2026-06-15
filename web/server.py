"""FastAPI server for the Swing System web UI.

The same desk, in a browser. Every flow is the existing `app.runner` function
(`fn(cfg, emit)`) — they are run on a worker thread and their `emit()` output is
streamed to the browser over Server-Sent Events, so the web app reuses 100% of
the engine. One run at a time, mirroring the desktop.

Run locally:   uvicorn web.server:app --host 0.0.0.0 --port 8000
Then open:     http://localhost:8000
Deploy (free always-on): see docs/WEB.md (Oracle Cloud Always-Free).
"""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import SECRET_FIELDS, AppConfig
from app.runner import (RunStopped, clear_stop, place_manual_order, request_stop,
                        run_curation, run_momentum_trade, run_portfolio_status,
                        run_position_review, run_recommendations, run_reddit_scan,
                        run_screen, run_strategy_backtest, run_trade_history, run_watch)

app = FastAPI(title="Swing System")
_STATIC = Path(__file__).parent / "static"

# The flows a button can trigger: name -> (function, default overrides).
ACTIONS = {
    "screen": run_screen,
    "deepdive": run_recommendations,
    "review": run_position_review,
    "portfolio": run_portfolio_status,
    "watch": run_watch,
    "momentum": run_momentum_trade,
    "backtest": run_strategy_backtest,
    "reddit": run_reddit_scan,
    "curate": run_curation,
    "trades": run_trade_history,
}

# -- single-run registry (one active run, like the desktop) ------------------
_RUNS: dict[str, dict] = {}
_current = {"id": None}
_last_recs: list[dict] = []
_lock = threading.Lock()


def _is_running() -> bool:
    rid = _current["id"]
    return bool(rid and _RUNS.get(rid, {}).get("status") == "running")


def _start(fn, **overrides) -> str | None:
    global _last_recs
    with _lock:
        if _is_running():
            return None
        rid = uuid.uuid4().hex[:12]
        q: queue.Queue = queue.Queue()
        _RUNS[rid] = {"q": q, "status": "running", "result": None}
        _current["id"] = rid
    clear_stop()
    cfg = AppConfig.load()
    for k, v in overrides.items():
        if v is not None:
            cfg = dataclasses.replace(cfg, **{k: v})

    def work():
        global _last_recs
        def emit(line: str):
            q.put(("log", str(line)))
        try:
            res = fn(cfg, emit)
            _RUNS[rid]["result"] = res
            if isinstance(res, dict) and res.get("recommendations"):
                _last_recs = res["recommendations"]
            q.put(("result", res if isinstance(res, dict) else {}))
            _RUNS[rid]["status"] = "done"
        except RunStopped:
            q.put(("log", "[stopped] run cancelled by user."))
            _RUNS[rid]["status"] = "stopped"
        except Exception as exc:                     # surface, never crash the server
            q.put(("log", f"[error] {type(exc).__name__}: {exc}"))
            _RUNS[rid]["status"] = "error"
        finally:
            q.put(("done", None))

    threading.Thread(target=work, daemon=True).start()
    return rid


# -- request models ----------------------------------------------------------
class RunReq(BaseModel):
    action: str
    universe: str | None = None
    ticker: str | None = None


class OrderReq(BaseModel):
    symbol: str
    qty: int
    order_type: str = "limit"
    limit_price: float | None = None
    stop: float | None = None
    target: float | None = None
    attach_bracket: bool = True
    ref_price: float | None = None


# -- run / stream / stop -----------------------------------------------------
@app.post("/api/run")
def api_run(req: RunReq):
    fn = ACTIONS.get(req.action)
    if not fn:
        raise HTTPException(400, f"unknown action {req.action!r}")
    overrides = {}
    if req.action == "screen" and req.universe:
        overrides["screen_index"] = req.universe
    if req.action == "deepdive":
        overrides["ticker"] = (req.ticker or "").strip().upper()
    rid = _start(fn, **overrides)
    if not rid:
        raise HTTPException(409, "a run is already in progress")
    return {"run_id": rid}


@app.get("/api/stream/{run_id}")
def api_stream(run_id: str):
    run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "no such run")
    q = run["q"]

    def gen():
        while True:
            try:
                kind, payload = q.get(timeout=1.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield "data: " + json.dumps({"kind": kind, "payload": payload},
                                        default=str) + "\n\n"
            if kind == "done":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/stop")
def api_stop():
    request_stop()
    return {"stopping": True}


@app.get("/api/status")
def api_status():
    return {"running": _is_running()}


# -- orders ------------------------------------------------------------------
@app.get("/api/orders")
def api_orders():
    return {"recommendations": _last_recs}


@app.post("/api/order")
def api_order(req: OrderReq):
    cfg = AppConfig.load()
    out_lines: list[str] = []
    res = place_manual_order(cfg, req.model_dump(), out_lines.append)
    return {"ok": bool(res.get("ok")), "log": out_lines, "result": res}


# -- config (chips + Settings) ----------------------------------------------
def _masked_config() -> dict:
    cfg = AppConfig.load()
    d = dataclasses.asdict(cfg)
    for k in SECRET_FIELDS:
        if d.get(k):
            d[k] = "********"                         # present-but-hidden marker
    d["_has_alpaca"] = bool(cfg.alpaca_key_id and cfg.alpaca_secret)
    d["_has_anthropic"] = bool(cfg.anthropic_api_key)
    return d


@app.get("/api/config")
def api_config_get():
    return _masked_config()


@app.post("/api/config")
def api_config_set(patch: dict):
    cfg = AppConfig.load()
    fields = {f.name for f in dataclasses.fields(cfg)}
    for k, v in (patch or {}).items():
        if k in fields and not (k in SECRET_FIELDS and v == "********"):
            cfg = dataclasses.replace(cfg, **{k: v})
    cfg.save()
    return _masked_config()


# -- performance / learning (read-only views) --------------------------------
@app.get("/api/performance")
def api_performance():
    from app import reco_ledger
    from system.reflection.calibration import calibration_table
    rows = reco_ledger.load()
    scored = sorted((r for r in rows if r.get("status") == "evaluated"
                     and isinstance(r.get("return_pct"), (int, float))),
                    key=lambda r: str(r.get("evaluated_on") or ""))
    eq, v = [1.0], 1.0
    for r in scored:
        v *= 1 + float(r["return_pct"]) / 100.0
        eq.append(round(v, 4))
    wins = sum(1 for r in scored if r["return_pct"] > 0)
    return {
        "n": len(scored),
        "hit_rate": round(100 * wins / len(scored), 0) if scored else 0,
        "avg_return": round(sum(r["return_pct"] for r in scored) / len(scored), 2)
        if scored else 0,
        "equity_curve": eq,
        "calibration": calibration_table(rows)["bands"],
        "cohorts": reco_ledger.cohort_stats(rows),
        "recent": [{"date": r.get("evaluated_on"), "symbol": r["symbol"],
                    "return_pct": r["return_pct"], "outcome": r.get("outcome"),
                    "conviction": r.get("conviction"), "hidden_gem": r.get("hidden_gem")}
                   for r in scored[-15:][::-1]],
        "open": sum(1 for r in rows if r.get("status") == "open"),
    }


@app.get("/api/learning")
def api_learning():
    from app import reco_ledger
    from app.learning import load_memory, summarize
    return {"lessons": summarize(load_memory()),
            "ledger": reco_ledger.summarize()}


@app.get("/api/watchlist")
def api_watchlist():
    import datetime
    from app import watchlist as wl
    return {"items": wl.active(datetime.date.today().isoformat())}


# -- static SPA --------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


# Assets live under /static/* (matching index.html's references).
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
