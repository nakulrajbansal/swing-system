"""Web UI server: endpoints, run registry, and SSE streaming of a flow's output.

Skipped entirely if FastAPI isn't installed (the web extra is optional)."""

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

from web import server                       # noqa: E402

client = TestClient(server.app)


def test_index_and_static_served():
    r = client.get("/")
    assert r.status_code == 200 and "Swing System" in r.text
    # the CSS/JS the page references must actually resolve.
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_config_is_masked():
    r = client.get("/api/config")
    assert r.status_code == 200
    d = r.json()
    assert "anthropic_api_key" in d and "_has_alpaca" in d
    # secrets are never returned in the clear.
    assert d["anthropic_api_key"] in ("", "********")


def test_unknown_action_rejected():
    r = client.post("/api/run", json={"action": "nope"})
    assert r.status_code == 400


def test_performance_and_learning_endpoints():
    for path in ("/api/performance", "/api/learning", "/api/watchlist", "/api/orders"):
        assert client.get(path).status_code == 200


def test_run_registry_streams_and_guards(monkeypatch):
    # A trivial flow: emit two lines, return a recommendations result.
    def fake(cfg, emit):
        emit("first line")
        emit("second line")
        return {"recommendations": [{"symbol": "AAA"}]}

    monkeypatch.setitem(server.ACTIONS, "faketest", fake)
    server._current["id"] = None
    rid = server._start(fake)
    assert rid
    # Drain the SSE stream and collect the logged lines + the done marker.
    seen = []
    with client.stream("GET", f"/api/stream/{rid}") as resp:
        for raw in resp.iter_lines():
            if raw and raw.startswith("data: "):
                msg = json.loads(raw[6:])
                seen.append(msg["kind"])
                if msg["kind"] == "done":
                    break
    assert "log" in seen and "done" in seen
    # The result's recommendations are exposed for the order tickets.
    time.sleep(0.05)
    assert server._last_recs and server._last_recs[0]["symbol"] == "AAA"


def test_schedule_endpoint_emits_cron_lines():
    s = client.get("/api/schedule").json()
    assert s["jobs"] and s["cron"]
    assert all("* * 1-5" in line for line in s["cron"])      # weekday crons
    assert "app.main --" in s["cron"][0]                     # invokes the CLI


def test_auth_gate_blocks_then_lets_in(monkeypatch):
    monkeypatch.setenv("SWING_WEB_PASSWORD", "hunter2")
    fresh = TestClient(server.app)                           # no cookie yet
    assert fresh.get("/api/config").status_code == 401       # gated
    assert fresh.get("/").status_code == 200                 # the page still loads
    assert fresh.get("/api/me").json()["auth_required"] is True
    assert fresh.post("/api/login", json={"password": "wrong"}).status_code == 401
    ok = fresh.post("/api/login", json={"password": "hunter2"})
    assert ok.status_code == 200
    assert fresh.get("/api/config").status_code == 200       # cookie now admits
    fresh.post("/api/logout")
    assert fresh.get("/api/config").status_code == 401       # logged out again


def test_open_when_no_password(monkeypatch):
    monkeypatch.delenv("SWING_WEB_PASSWORD", raising=False)
    assert TestClient(server.app).get("/api/config").status_code == 200


def test_only_one_run_at_a_time(monkeypatch):
    started = {"go": True}

    def slow(cfg, emit):
        while started["go"]:
            time.sleep(0.01)

    server._current["id"] = None
    rid = server._start(slow)
    assert rid
    assert server._start(slow) is None        # second start refused while running
    started["go"] = False
    time.sleep(0.05)
