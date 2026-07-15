"""Phase A/B/C upgrades: calibration, preset self-tuning, debate escalation,
watchlist triggers."""

import numpy as np
import pandas as pd

from app.config import AppConfig
from system.reflection.calibration import (calibrated_probability,
                                           calibration_table, describe)


def _scored(conviction, ret, n):
    return [{"status": "evaluated", "conviction": conviction, "return_pct": ret,
             "evaluated_on": f"2026-06-{10 + i % 19:02d}"} for i in range(n)]


def test_calibration_shrinks_honestly():
    # No evidence: the desk's claim comes back unchanged, labeled uncalibrated.
    empty = calibration_table([])
    assert calibrated_probability(0.62, empty) == 0.62
    assert "uncalibrated" in describe(0.62, empty)
    # A seasoned band pulls toward the realized rate: 0.6-conviction calls that
    # won only 30% of 20 tries must read well below the stated 0.60.
    rows = _scored(0.60, -3.0, 14) + _scored(0.60, 5.0, 6)     # 30% win rate
    table = calibration_table(rows)
    p = calibrated_probability(0.60, table)
    assert p < 0.45
    assert "calibrated on 20" in describe(p, table)
    # Garbage input never crashes.
    assert calibrated_probability(None, table) is None


def test_calibration_uses_trailing_window():
    old_losses = _scored(0.60, -5.0, 80)        # ancient losing streak
    recent_wins = [{"status": "evaluated", "conviction": 0.60, "return_pct": 4.0,
                    "evaluated_on": "2026-06-30"} for _ in range(20)]
    table = calibration_table(old_losses + recent_wins, trailing=20)
    band = next(b for b in table["bands"] if b["lo"] == 0.55)
    assert band["win_rate_pct"] == 100.0        # only the recent window counts


def test_preset_selector_picks_the_winning_personality():
    from app.screen import _metrics
    from app.strategy import WEIGHT_PRESETS, select_preset

    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-06-01", periods=320, freq="B")
    closes = pd.DataFrame({
        sym: 50 * np.cumprod(1 + rng.normal(d, 0.004, 320))
        for sym, d in [("SPY", 0.0004), ("AAA", 0.0012), ("BBB", 0.0008),
                       ("CCC", 0.0002), ("DDD", -0.0004)]}, index=idx)
    name, weights, board = select_preset(closes, _metrics, top_k=2, hold_days=10)
    assert name in WEIGHT_PRESETS and weights == WEIGHT_PRESETS[name]
    assert set(board) == set(WEIGHT_PRESETS)
    assert all("total" in v for v in board.values())
    # Too-short history degrades to base, never crashes.
    nm, w, _ = select_preset(closes.iloc[:50], _metrics)
    assert nm == "base"


def test_contested_debate_fires_second_round(synth_store):
    from harness.signals import Edge01Filing, Edge06Insider, Edge08Momentum
    from system.agents.core import (HypothesisAgent, PortfolioManagerAgent,
                                    SkepticAgent)
    from system.agents.llm_client import MockLLMClient
    from system.agents.specialists import EdgeSpecialist
    from system.config import SystemConfig
    from system.orchestrator import Orchestrator
    from harness.data.loader import available_at_for_session

    store, sector_map = synth_store
    cfg = SystemConfig()
    client = MockLLMClient()

    class SevereSkeptic(SkepticAgent):
        def deterministic(self, inputs):
            crit = super().deterministic(inputs)
            for o in crit.objections:
                o.severity = 0.65                       # contested, not a kill
            crit.verdict = "caution"
            return crit

    specs = [EdgeSpecialist(E(), client, cfg.models.framing)
             for E in (Edge01Filing, Edge06Insider, Edge08Momentum)]
    orch = Orchestrator(store, specs, HypothesisAgent(client, cfg.models.synthesis),
                        SevereSkeptic(client, cfg.models.adversarial),
                        PortfolioManagerAgent(client, cfg.models.adversarial), cfg)
    sessions = pd.to_datetime(sorted(store.read_table("prices")["date"].unique()))
    saw_rejoinder = False
    for d in sessions[len(sessions) // 2:]:
        res = orch.run_cycle(available_at_for_session(d))
        for rec in res.deliberation.values():
            agents = [s["agent"] for s in rec.get("steps", [])]
            if "skeptic_rejoinder" in agents:
                saw_rejoinder = True
                out = next(s["output"] for s in rec["steps"]
                           if s["agent"] == "skeptic_rejoinder")
                assert out["stands"] is True            # deterministic: unchanged
        if saw_rejoinder:
            break
    assert saw_rejoinder, "high-severity critique should trigger round 2"


def test_pm_weighs_rejoinder_outcome():
    from system.agents.core import PortfolioManagerAgent
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    pm = PortfolioManagerAgent(MockLLMClient(), SystemConfig().models.adversarial)
    base = {"symbol": "AAA",
            "hypothesis": {"decision": "propose", "raw_conviction": 0.85},
            "critique": {"verdict": "caution", "max_severity": 0.6},
            "rebuttal": "addressed", "price": 100.0, "atr": 2.0}
    conceded = pm.run({**base, "rejoinder": {"stands": False, "final_severity": 0.6}})
    stands = pm.run({**base, "rejoinder": {"stands": True, "final_severity": 0.8}})
    # A conceded objection frees conviction; a standing one drags it down.
    assert conceded.final_conviction > stands.final_conviction


def test_missing_protection_requires_a_real_stop():
    from app.runner import _missing_protection

    plan = {"stop": 90.0, "target": 120.0}
    # A take-profit target alone is NOT downside protection: the stop is missing.
    target_only = [{"type": "limit", "limit_price": 120.0}]
    arm_stop, arm_target = _missing_protection(target_only, plan)
    assert arm_stop == 90.0 and arm_target is None
    # A full OCO already resting: nothing to arm.
    full = [{"type": "limit", "limit_price": 120.0},
            {"type": "stop", "stop_price": 90.0}]
    assert _missing_protection(full, plan) == (None, None)
    # Nothing resting: arm both.
    assert _missing_protection([], plan) == (90.0, 120.0)
    # A stop_price on any order counts as a resting stop (type variants).
    assert _missing_protection([{"type": "stop_limit", "stop_price": 90.0}],
                               {"stop": 90.0})[0] is None


class _RecordingBroker:
    """Records the broker calls _arm_protection makes. submit_exit_orders fails
    with the real Alpaca 'insufficient qty' error while the requested qty is
    still held by a resting order — so the test proves the fix frees it first."""

    def __init__(self):
        self.deleted: list[str] = []
        self.held = True            # shares held by the resting exit until cancelled
        self.submitted = None

    def _req(self, method, path):
        if method == "DELETE":
            self.deleted.append(path.rsplit("/", 1)[-1])
            self.held = False       # cancelling the resting order frees the shares
        return {}

    def submit_exit_orders(self, sym, qty, stop=None, target=None):
        if self.held:
            return {"error": "Alpaca 403: insufficient qty available for order"}
        self.submitted = {"sym": sym, "qty": qty, "stop": stop, "target": target}
        return {"id": "oco-1", "status": "accepted"}


def test_arm_protection_replaces_a_lone_target_with_full_oco():
    # Regression: a take-profit target rests holding all the shares, but there is
    # NO protective stop. The old code armed a bare stop for the same qty and
    # Alpaca rejected it ("insufficient qty available"), leaving the downside
    # unprotected. The fix cancels the resting target, then re-places the FULL
    # stop+target so a real stop ends up at the broker.
    from app.runner import _arm_protection

    plan = {"stop": 90.0, "target": 120.0}
    resting = [{"id": "target-123", "type": "limit", "limit_price": 120.0}]
    broker = _RecordingBroker()
    lines: list[str] = []

    armed = _arm_protection(broker, "ABC", 100, resting, plan, lines.append)

    assert armed is True
    assert broker.deleted == ["target-123"]           # the lone target was cancelled
    assert broker.submitted == {"sym": "ABC", "qty": 100,
                                "stop": 90.0, "target": 120.0}  # full OCO re-placed
    assert any("DOWNSIDE UNPROTECTED" in ln for ln in lines)


def test_arm_protection_escalates_when_replace_fails_after_cancel():
    # Hardening: once the resting order is cancelled the shares are naked. If the
    # replacement OCO then fails, the position is UNPROTECTED — that must be
    # flagged CRITICAL, not logged as a routine failure.
    from app.runner import _arm_protection

    class _BrokenBroker(_RecordingBroker):
        def submit_exit_orders(self, sym, qty, stop=None, target=None):
            return {"error": "network down"}            # replace always fails

    plan = {"stop": 90.0, "target": 120.0}
    resting = [{"id": "target-123", "type": "limit", "limit_price": 120.0}]
    broker = _BrokenBroker()
    lines: list[str] = []

    armed = _arm_protection(broker, "ABC", 100, resting, plan, lines.append)

    assert armed is False
    assert broker.deleted == ["target-123"]             # the target WAS cancelled
    assert any("CRITICAL" in ln and "UNPROTECTED" in ln for ln in lines)


def test_arm_protection_noop_when_full_protection_already_rests():
    from app.runner import _arm_protection

    plan = {"stop": 90.0, "target": 120.0}
    resting = [{"id": "t", "type": "limit", "limit_price": 120.0},
               {"id": "s", "type": "stop", "stop_price": 90.0}]
    broker = _RecordingBroker()

    assert _arm_protection(broker, "ABC", 100, resting, plan, lambda _l: None) is False
    assert broker.deleted == [] and broker.submitted is None


def test_arm_protection_arms_directly_when_nothing_rests():
    from app.runner import _arm_protection

    broker = _RecordingBroker()
    broker.held = False             # no resting order, so nothing holds the shares
    armed = _arm_protection(broker, "ABC", 50, [],
                            {"stop": 90.0, "target": 120.0}, lambda _l: None)

    assert armed is True
    assert broker.deleted == []                       # no cancel needed
    assert broker.submitted["qty"] == 50


def test_arm_protection_waits_out_async_cancel_then_arms():
    # Real-world race from the review logs: a DELETE only moves the resting order
    # into pending_cancel, so the shares stay held_for_orders and the FIRST
    # replacement attempts are rejected with 'insufficient qty available'. The fix
    # retries while that transient error persists instead of leaving the position
    # naked (the old code submitted once and gave up -> every name UNPROTECTED).
    from app.runner import _arm_protection

    class _SlowSettleBroker(_RecordingBroker):
        def __init__(self, frees_after):
            super().__init__()
            self.frees_after = frees_after
            self.attempts = 0

        def submit_exit_orders(self, sym, qty, stop=None, target=None):
            self.attempts += 1
            if self.attempts <= self.frees_after:    # cancel not settled yet
                return {"error": "Alpaca 403 code 40310000: insufficient qty "
                                 "available for order"}
            self.submitted = {"sym": sym, "qty": qty, "stop": stop, "target": target}
            return {"id": "oco-1", "status": "accepted"}

    plan = {"stop": 90.0, "target": 120.0}
    resting = [{"id": "target-123", "type": "limit", "limit_price": 120.0}]
    broker = _SlowSettleBroker(frees_after=2)
    slept: list[float] = []

    armed = _arm_protection(broker, "ABC", 100, resting, plan, lambda _l: None,
                            sleep=lambda d: slept.append(d))

    assert armed is True
    assert broker.attempts == 3                       # two rejections, then armed
    assert broker.submitted["stop"] == 90.0           # full OCO finally rests
    assert len(slept) == 2                            # waited between retries


def test_arm_protection_gives_up_and_escalates_if_cancel_never_settles():
    # Bounded: if the held-qty error never clears, don't spin forever — give up
    # and flag CRITICAL so the naked position is surfaced.
    from app.runner import _arm_protection

    class _NeverSettles(_RecordingBroker):
        def submit_exit_orders(self, sym, qty, stop=None, target=None):
            return {"error": "403 code 40310000 insufficient qty available"}

    plan = {"stop": 90.0, "target": 120.0}
    resting = [{"id": "t", "type": "limit", "limit_price": 120.0}]
    broker = _NeverSettles()
    lines: list[str] = []

    armed = _arm_protection(broker, "ABC", 100, resting, plan, lines.append,
                            sleep=lambda _d: None)

    assert armed is False
    assert any("CRITICAL" in ln and "UNPROTECTED" in ln for ln in lines)


def test_watchlist_clear_empties_and_counts(tmp_path):
    from app import watchlist as wl

    p = tmp_path / "wl.json"
    wl.add([{"symbol": "ABC", "pullback_target": 10.0},
            {"symbol": "XYZ", "breakout_level": 50.0}], "2026-06-23", path=p)
    assert len(wl.load(p)) == 2
    assert wl.clear(path=p) == 2
    assert wl.load(p) == []
    assert wl.clear(path=p) == 0                      # already empty -> 0


def test_watchlist_annotate_reports_validity():
    import pandas as pd

    from app import watchlist as wl

    items = [{"symbol": "ABC", "pullback_target": 100.0, "expires": "2026-06-30"},
             {"symbol": "OLD", "breakout_level": 50.0, "expires": "2026-06-01"}]
    closes = pd.DataFrame({"ABC": [110.0, 101.0]})
    ann = {a["symbol"]: a for a in wl.annotate(items, "2026-06-23", closes)}

    assert ann["ABC"]["days_left"] == 7 and ann["ABC"]["expired"] is False
    assert ann["ABC"]["price"] == 101.0
    assert ann["ABC"]["distance_pct"] == 1.0         # 101 is +1% vs the 100 target
    assert ann["OLD"]["expired"] is True             # expiry already in the past
    assert "price" not in ann["OLD"]                 # no price column -> no price


def test_governor_book_maps_the_live_position_into_caps():
    # Regression: the momentum entry path used to size every trade as if flat,
    # bypassing the Governor's portfolio caps. _governor_book turns the broker's
    # open positions into Governor Positions (notional from the latest close) so
    # the gross/sector/open-count caps actually see the book.
    import pandas as pd

    from app.runner import _governor_book, _gross_exposure
    from system.execution.broker import BrokerPosition

    live = {"AAA": BrokerPosition("AAA", 100, 50.0, 0.0, 0.0),
            "BBB": BrokerPosition("BBB", 0, 10.0, 0.0, 0.0)}   # 0-share -> dropped
    closes = pd.DataFrame({"AAA": [55.0, 60.0]})               # mark = latest close
    book = _governor_book(live, closes, {"AAA": "XLK"})

    assert len(book) == 1                                       # the 0-share name is skipped
    p = book[0]
    assert p.symbol == "AAA" and p.shares == 100 and p.sector == "XLK"
    assert p.price == 60.0 and p.notional == 6000.0            # mark x shares
    assert p.open_risk == 0.0                                   # stop set to entry: no phantom heat
    # gross exposure is book notional / equity.
    assert _gross_exposure(book, 60_000.0) == 0.1
    assert _gross_exposure([], 0.0) == 0.0                      # unknown equity -> 0


def test_has_resting_stop_only_counts_real_stops():
    # A take-profit limit is NOT downside protection; only a stop guards it.
    from app.runner import _has_resting_stop

    assert _has_resting_stop([]) is False
    assert _has_resting_stop([{"type": "limit", "limit_price": 120.0}]) is False
    assert _has_resting_stop([{"type": "stop", "stop_price": 90.0}]) is True
    assert _has_resting_stop([{"type": "stop_limit", "stop_price": 90.0}]) is True
    # A bare stop_price field counts even if the type label is missing.
    assert _has_resting_stop([{"stop_price": 90.0}]) is True


def test_open_orders_flattens_bracket_legs_so_the_stop_is_visible(monkeypatch):
    # ROOT CAUSE of the 'stop never persisted' bug: a plain status=open query
    # returns only the OCO PRIMARY (the take-profit limit); the protective stop
    # is a nested child leg. open_orders() must pass nested=true and flatten the
    # legs so the stop the broker is already holding becomes visible — otherwise
    # the review re-arms it on every run, churning the OCO and going naked.
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("key", "secret", env="paper")
    nested = [{"id": "tp", "symbol": "ABC", "side": "sell", "type": "limit",
               "limit_price": 120.0,
               "legs": [{"id": "sl", "symbol": "ABC", "side": "sell",
                         "type": "stop", "stop_price": 90.0}]}]

    def fake(method, path, payload=None):
        assert "nested=true" in path                       # the actual fix
        return nested

    monkeypatch.setattr(b, "_req", fake)
    orders = b.open_orders()
    assert {o["id"] for o in orders} == {"tp", "sl"}        # leg pulled up
    # The desk can now SEE the stop, so it will NOT needlessly re-arm.
    from app.runner import _has_resting_stop, _missing_protection
    assert _has_resting_stop(orders) is True
    assert _missing_protection(orders, {"stop": 90.0, "target": 120.0}) == (None, None)


def test_open_orders_dedupes_and_filters_by_symbol(monkeypatch):
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("key", "secret", env="paper")
    # Same leg appearing both top-level and nested must not double-count.
    raw = [{"id": "a", "symbol": "ABC", "legs": [{"id": "b", "symbol": "ABC"}]},
           {"id": "b", "symbol": "ABC"},
           {"id": "c", "symbol": "XYZ"}]
    monkeypatch.setattr(b, "_req", lambda *a, **k: raw)
    assert {o["id"] for o in b.open_orders()} == {"a", "b", "c"}
    assert {o["id"] for o in b.open_orders("ABC")} == {"a", "b"}


def test_order_role_classifies_entries_stops_and_targets():
    from app.runner import _order_role

    assert _order_role({"side": "buy", "type": "limit"}, set()) == "entry"
    assert _order_role({"side": "sell", "type": "stop", "stop_price": 90}, set()) == "stop"
    assert _order_role({"side": "sell", "type": "limit", "limit_price": 120}, set()) == "target"


class _OrderBroker:
    """Fake broker for the cancel flow: tracks which order ids get DELETEd."""

    def __init__(self, orders, held=()):
        self._orders = orders
        self._held = list(held)
        self.cancelled: list[str] = []

    def open_orders(self, symbol=None):
        return self._orders

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        return {"id": oid, "status": "canceled"}

    def _req(self, method, path, payload=None):
        if "positions" in path:
            return [{"symbol": s} for s in self._held]
        return {}


def test_cancel_orders_entries_scope_frees_buying_power_without_touching_stops(monkeypatch):
    # The buying-power fix: 'entries' scope cancels only unfilled BUY orders, so
    # margin is freed without ever stripping a position's protective stop.
    from app import runner

    orders = [{"id": "buy1", "symbol": "MU", "side": "buy", "type": "limit"},
              {"id": "stop1", "symbol": "AMD", "side": "sell", "type": "stop",
               "stop_price": 400.0}]
    broker = _OrderBroker(orders, held=["AMD"])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")

    out = runner.cancel_orders(cfg, lambda _l: None, scope="entries")
    assert broker.cancelled == ["buy1"]                    # the stop was spared
    assert out["cancelled"] == 1


def test_cancel_orders_by_id_flags_a_naked_stop(monkeypatch):
    from app import runner

    orders = [{"id": "stop1", "symbol": "AMD", "side": "sell", "type": "stop",
               "stop_price": 400.0}]
    broker = _OrderBroker(orders, held=["AMD"])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")
    lines: list[str] = []

    out = runner.cancel_orders(cfg, lines.append, order_ids=["stop1"])
    assert broker.cancelled == ["stop1"] and out["cancelled"] == 1
    assert any("UNPROTECTED" in ln for ln in lines)        # the naked warning


def test_preset_hysteresis_keeps_base_unless_challenger_is_decisively_ahead():
    from app.strategy import PRESET_SWITCH_MARGIN_PP, _choose_preset

    def board(base, timing):
        return {"base": {"total": base, "sharpe": 1.0},
                "timing": {"total": timing, "sharpe": 1.0},
                "discovery": {"total": -5.0, "sharpe": 0.0},
                "defensive": {"total": -5.0, "sharpe": 0.0}}

    # Marginal edge over base -> keep base (don't chase in-sample noise).
    assert _choose_preset(board(5.0, 5.0 + PRESET_SWITCH_MARGIN_PP - 0.5)) == "base"
    # Decisive edge over base -> adopt the challenger.
    assert _choose_preset(board(5.0, 5.0 + PRESET_SWITCH_MARGIN_PP + 1.0)) == "timing"
    # Challenger "wins" but is negative -> keep base (never chase a losing tilt).
    assert _choose_preset(board(-8.0, -1.0)) == "base"


def test_extend_plan_trails_a_winner_once(tmp_path):
    from app import reco_ledger

    p = tmp_path / "ledger.json"
    reco_ledger.record([{"symbol": "AAA", "entry": 100.0, "stop": 90.0,
                         "target": 130.0, "hold_days": 10, "exit_by": "2026-07-01",
                         "conviction": 0.6}], "screen", "2026-06-20", path=p)
    r = reco_ledger.extend_plan("AAA", "2026-07-11", new_stop=100.0, path=p)
    assert r["exit_by"] == "2026-07-11" and r["stop"] == 100.0 and r["extended"]
    # One-shot: a second extend is refused.
    assert reco_ledger.extend_plan("AAA", "2026-07-21", new_stop=105.0, path=p) is None


def test_symbol_normalization_round_trips_class_shares():
    from system.execution.broker import to_broker_symbol, to_data_symbol

    assert to_broker_symbol("MOG-A") == "MOG.A"            # data -> Alpaca
    assert to_data_symbol("MOG.A") == "MOG-A"              # Alpaca -> data
    assert to_broker_symbol("BRK-B") == "BRK.B"
    assert to_broker_symbol("AAPL") == "AAPL"              # plain names untouched
    assert to_data_symbol("AAPL") == "AAPL"
    assert to_broker_symbol("") == "" and to_broker_symbol(None) is None


def test_manual_order_sends_alpaca_class_symbol(monkeypatch):
    # Regression from the 2026-07-13 order log: an order for the data-feed ticker
    # 'MOG-A' was rejected 422 'asset not found' because Alpaca wants 'MOG.A'.
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("k", "s", env="paper")
    seen = {}

    def fake(method, path, payload=None):
        seen["payload"] = payload
        return {"id": "ord-1", "status": "accepted"}

    monkeypatch.setattr(b, "_req", fake)
    b.submit_manual("MOG-A", 10, side="buy", order_type="limit", limit_price=200.0)
    assert seen["payload"]["symbol"] == "MOG.A"            # dot form to the broker


def test_broker_reads_present_class_symbols_in_data_form(monkeypatch):
    # Positions / open orders / fills must map Alpaca 'MOG.A' back to 'MOG-A' so
    # the review matches the position to its 'MOG-A' plan.
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("k", "s", env="paper")

    def fake(method, path, payload=None):
        if "positions" in path:
            return [{"symbol": "MOG.A", "qty": "10", "avg_entry_price": "200"}]
        if "orders" in path:
            return [{"id": "o1", "symbol": "MOG.A", "side": "sell", "type": "stop",
                     "stop_price": 180.0}]
        if "activities" in path:
            return [{"id": "f1", "symbol": "MOG.A", "side": "buy", "price": "200"}]
        return {}

    monkeypatch.setattr(b, "_req", fake)
    assert "MOG-A" in b.positions()                        # keyed in data form
    assert b.open_orders()[0]["symbol"] == "MOG-A"
    assert b.open_orders("MOG-A")[0]["id"] == "o1"         # filter accepts data form
    assert b.fills(1)[0]["symbol"] == "MOG-A"


def test_throttle_bar_keeps_a_name_at_exactly_the_raised_bar():
    # Regression from the 2026-06-30 screen log: a +0.05 bump made the bar
    # 0.55 + 0.05 = 0.6000000000000001 in float, so FFIV at exactly 0.60 was
    # wrongly held back. The bar must be rounded so a name AT 0.60 qualifies.
    from app.runner import _apply_throttle_bar, _throttle_bar

    bar = _throttle_bar(0.05)
    assert bar == 0.60                                     # not 0.6000000000000001
    recs = [{"symbol": "FFIV", "conviction": 0.60},
            {"symbol": "MRVL", "conviction": 0.55},
            {"symbol": "AAA", "conviction": 0.61}]
    kept, held = _apply_throttle_bar(recs, bar)
    kept_syms = {r["symbol"] for r in kept}
    assert kept_syms == {"FFIV", "AAA"}                    # 0.60 meets the 0.60 bar
    assert {r["symbol"] for r in held} == {"MRVL"}         # 0.55 < 0.60 held back


def test_latest_per_symbol_dedupes_so_one_closure_scores_once():
    # Regression from the 2026-07-02 review log: CRWD had TWO open executed recs
    # (bought 06-18 and 06-23), and a single broker stop fill scored BOTH at
    # -0.7%, double-counting the closure in the learning data. The broker-close
    # path must collapse to one rec per symbol — the most recent.
    from app.runner import _latest_per_symbol

    recs = [{"symbol": "CRWD", "date": "2026-06-18"},
            {"symbol": "CRWD", "date": "2026-06-23"},
            {"symbol": "AVT", "date": "2026-06-24"}]
    out = _latest_per_symbol(recs)
    assert len(out) == 2                                   # CRWD collapsed to one
    crwd = [r for r in out if r["symbol"] == "CRWD"]
    assert len(crwd) == 1 and crwd[0]["date"] == "2026-06-23"   # the most recent
    assert {r["symbol"] for r in out} == {"CRWD", "AVT"}


def test_fill_scores_rec_rejects_stale_fills():
    # Regression: CSCO's 06-23 rec was scored against a 06-17 sell fill from a
    # prior position — a fill can't close a rec that did not yet exist.
    from app.runner import _fill_scores_rec

    assert _fill_scores_rec("2026-07-02", "2026-06-23") is True   # fill after rec
    assert _fill_scores_rec("2026-06-23", "2026-06-23") is True   # same day: ok
    assert _fill_scores_rec("2026-06-17", "2026-06-23") is False  # fill predates rec
    assert _fill_scores_rec("2026-07-02", None) is True           # no rec date: ok
    assert _fill_scores_rec(None, "2026-06-23") is True           # no fill date: ok


class _ManualOrderBroker:
    """Mock broker for place_manual_order: an account with equity/buying-power,
    a set of open positions (for the gross-exposure check), and a submit that
    records the last order."""

    def __init__(self, equity, buying_power, positions, legs=None):
        self._acct = {"equity": str(equity), "buying_power": str(buying_power)}
        self._positions = positions
        self._legs = legs                       # child legs the confirmation echoes
        self.submitted = None

    def account(self):
        return self._acct

    def _req(self, method, path, payload=None):
        if "positions" in path:
            return self._positions
        return {}

    def submit_manual(self, sym, qty, **kw):
        self.submitted = {"sym": sym, "qty": qty, **kw}
        resp = {"id": "ord-1", "status": "accepted"}
        if self._legs is not None:
            resp["legs"] = self._legs
        return resp


def test_manual_order_warns_when_it_levers_past_the_ceiling(monkeypatch):
    from app import runner

    # Equity 100k, already 95k gross; buying 20 @ $1000 = 20k -> 1.15x, past 1.0x.
    broker = _ManualOrderBroker(100_000, 300_000,
                                [{"market_value": "95000"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    monkeypatch.setattr("app.reco_ledger.mark_executed", lambda *a, **k: False)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")
    lines: list[str] = []

    res = runner.place_manual_order(
        cfg, {"symbol": "XYZ", "qty": 20, "order_type": "limit",
              "limit_price": 1000.0, "attach_bracket": False}, lines.append)

    assert res["ok"] is True                               # warned, NOT blocked
    assert any("[RISK]" in ln and "gross exposure" in ln for ln in lines)
    assert broker.submitted["qty"] == 20                   # order still placed


def test_manual_order_quiet_when_within_the_ceiling(monkeypatch):
    from app import runner

    # Equity 100k, 20k gross; buying 10 @ $1000 = 10k -> 0.30x, well under 1.0x.
    broker = _ManualOrderBroker(100_000, 300_000,
                                [{"market_value": "20000"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    monkeypatch.setattr("app.reco_ledger.mark_executed", lambda *a, **k: False)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")
    lines: list[str] = []

    res = runner.place_manual_order(
        cfg, {"symbol": "XYZ", "qty": 10, "order_type": "limit",
              "limit_price": 1000.0, "attach_bracket": False}, lines.append)

    assert res["ok"] is True
    assert not any("[RISK]" in ln for ln in lines)         # no false alarm


def test_manual_bracket_warns_when_broker_drops_the_stop_leg(monkeypatch):
    # Regression (CHEF 2026-07-15): a bracket was submitted but only the target
    # leg came to rest - no protective stop. Verify the order path flags a missing
    # stop leg in the broker's confirmation instead of trusting "submitted".
    from app import runner

    # Confirmation echoes ONLY a take-profit leg (limit) - the stop is missing.
    broker = _ManualOrderBroker(100_000, 300_000, [{"market_value": "20000"}],
                                legs=[{"type": "limit", "limit_price": "110.64"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    monkeypatch.setattr("app.reco_ledger.mark_executed", lambda *a, **k: False)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")
    lines: list[str] = []

    res = runner.place_manual_order(
        cfg, {"symbol": "CHEF", "qty": 10, "order_type": "limit",
              "limit_price": 97.44, "stop": 90.84, "target": 110.64,
              "attach_bracket": True}, lines.append)

    assert res["ok"] is True                               # warned, not blocked
    assert any("[RISK]" in ln and "protective stop" in ln for ln in lines)


def test_manual_bracket_quiet_when_stop_leg_is_attached(monkeypatch):
    # The happy path: the confirmation carries both a take-profit AND a stop leg,
    # so no missing-stop warning fires.
    from app import runner

    broker = _ManualOrderBroker(
        100_000, 300_000, [{"market_value": "20000"}],
        legs=[{"type": "limit", "limit_price": "110.64"},
              {"type": "stop", "stop_price": "90.84"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    monkeypatch.setattr("app.reco_ledger.mark_executed", lambda *a, **k: False)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper")
    lines: list[str] = []

    res = runner.place_manual_order(
        cfg, {"symbol": "CHEF", "qty": 10, "order_type": "limit",
              "limit_price": 97.44, "stop": 90.84, "target": 110.64,
              "attach_bracket": True}, lines.append)

    assert res["ok"] is True
    assert not any("protective stop" in ln for ln in lines)   # no false alarm


class _AutoEntryBroker:
    """Mock broker for _maybe_place_orders: submit_entry echoes the child legs
    the confirmation would carry (a bracket always has stop + target)."""

    def __init__(self, legs):
        self._legs = legs
        self.submitted = None

    def submit_entry(self, symbol, shares, **kw):
        self.submitted = {"symbol": symbol, "shares": shares, **kw}
        return {"id": "auto-1", "status": "accepted", "legs": self._legs}


def _auto_ticket(symbol="AMD"):
    import types
    return types.SimpleNamespace(symbol=symbol, shares=10, entry=100.0,
                                 stop=90.0, target=120.0)


def test_auto_entry_warns_when_broker_drops_the_stop_leg(monkeypatch):
    # The momentum auto-entry path is a bracket too: if the confirmation shows
    # only a take-profit leg, flag the missing stop at order time.
    from app import runner

    broker = _AutoEntryBroker(legs=[{"type": "limit", "limit_price": "120.0"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper",
                    place_orders=True)
    lines: list[str] = []

    runner._maybe_place_orders(cfg, [_auto_ticket()], {"AMD": "Tech"}, lines.append)

    assert any("[RISK]" in ln and "protective stop" in ln for ln in lines)


def test_auto_entry_quiet_when_stop_leg_is_attached(monkeypatch):
    from app import runner

    broker = _AutoEntryBroker(legs=[{"type": "limit", "limit_price": "120.0"},
                                    {"type": "stop", "stop_price": "90.0"}])
    monkeypatch.setattr(runner, "_alpaca_broker", lambda cfg: broker)
    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", alpaca_env="paper",
                    place_orders=True)
    lines: list[str] = []

    runner._maybe_place_orders(cfg, [_auto_ticket()], {"AMD": "Tech"}, lines.append)

    assert not any("protective stop" in ln for ln in lines)   # no false alarm


def test_order_qty_is_resilient_to_missing_fields():
    # Regression: a flattened OCO leg can lack a clean `qty`, which printed a bare
    # '?' in the cancel log. Fall back through the alternatives, never '?'.
    from app.runner import _order_qty

    assert _order_qty({"qty": "100"}) == "100"
    assert _order_qty({"qty": None, "filled_qty": "50"}) == "50"
    assert _order_qty({}) == "—"                            # nothing usable
    assert _order_qty({"qty": 80.0}) == "80"               # numeric coerced


def test_equity_curve_is_risk_weighted_not_full_notional(tmp_path):
    # The Performance-chart fix: the naive curve compounded each call's FULL
    # per-trade return as if 100% of capital rode every one — turning a small
    # average edge into a fictional huge curve. The curve must instead compound
    # at the desk's ~1%-risk position weight (risk / stop-distance).
    from app import reco_ledger

    p = tmp_path / "ledger.json"
    # Ten +10% calls, each with a 10% stop distance -> ~10% notional weight, so a
    # weighted per-call contribution of ~+1% (not +10%).
    rows = []
    for i in range(10):
        rows.append({"id": f"r{i}", "symbol": f"S{i}", "status": "evaluated",
                     "return_pct": 10.0, "entry": 100.0, "stop": 90.0,
                     "date": f"2026-03-{i + 1:02d}", "evaluated_on": "2026-04-01"})
    reco_ledger.save(rows, p)
    ec = reco_ledger.equity_curve(reco_ledger.load(p))

    assert ec["n"] == 10
    assert 9 <= ec["avg_weight_pct"] <= 11                  # ~10% notional/trade
    # Full-notional would be 1.10^10-1 = +159%; risk-weighted is ~1.01^10 ≈ +10%.
    assert 8 <= ec["total_return_pct"] <= 13
    assert ec["curve"][0] == 1.0 and ec["curve"][-1] > 1.0


def test_equity_curve_caps_weight_at_single_name_limit(tmp_path):
    # A very tight stop would imply a huge position for 1% risk; the weight must
    # be capped at the single-name limit (15%), never larger.
    from app import reco_ledger
    from system.config import DEFAULT_CONFIG

    p = tmp_path / "ledger.json"
    # 1% stop distance -> uncapped weight would be 100%; must cap at 15%.
    reco_ledger.save([{"id": "r", "symbol": "S", "status": "evaluated",
                       "return_pct": 5.0, "entry": 100.0, "stop": 99.0,
                       "date": "2026-03-01"}], p)
    ec = reco_ledger.equity_curve(reco_ledger.load(p))
    assert ec["avg_weight_pct"] <= DEFAULT_CONFIG.limits.max_single_name * 100 + 0.01


def test_governor_book_falls_back_to_entry_when_no_close():
    from app.runner import _governor_book
    from system.execution.broker import BrokerPosition

    import pandas as pd
    live = {"ZZZ": BrokerPosition("ZZZ", 10, 42.0, 0.0, 0.0)}
    book = _governor_book(live, pd.DataFrame(), {})             # no price column
    assert book[0].price == 42.0 and book[0].sector == "?"     # mark falls back to entry


def test_infer_exit_reason_names_the_trigger():
    from app.runner import _infer_exit_reason

    # AMAT-style plan: stop 443.79, target 588.92.
    assert _infer_exit_reason(588.10, 443.79, 588.92) == "target"   # ~target fill
    assert _infer_exit_reason(443.00, 443.79, 588.92) == "stop"     # stopped out
    assert _infer_exit_reason(440.00, 443.79, 588.92) == "stop"     # gapped through
    assert _infer_exit_reason(520.00, 443.79, 588.92) == "manual"   # neither level
    assert _infer_exit_reason(None, 443.79, 588.92) == "manual"
    assert _infer_exit_reason(100.0, None, None) == "manual"


def test_broker_fills_pages_within_alpaca_cap(monkeypatch):
    from system.execution.broker import AlpacaBroker

    b = AlpacaBroker("key", "secret", env="paper")
    calls = []
    # 150 fills across two pages; Alpaca caps page_size at 100.
    full = [{"id": f"f{i}", "symbol": "AMAT", "side": "sell"} for i in range(150)]

    def fake(method, path, payload=None):
        calls.append(path)
        assert "page_size=100" in path or "page_size=50" in path   # never >100
        if "page_token=" in path:
            return full[100:]
        return full[:100]

    monkeypatch.setattr(b, "_req", fake)
    fills = b.fills(200)
    assert len(fills) == 150
    assert "activities?activity_types=FILL" in calls[0]
    assert any("page_token=f99" in c for c in calls)               # cursor advanced


def test_executed_link_and_full_open_list(tmp_path):
    from app import reco_ledger

    p = tmp_path / "ledger.json"
    recs = [{"symbol": s, "entry": 100.0, "hold_days": 10,
             "exit_by": f"2026-06-{17 + i}", "conviction": 0.5}
            for i, s in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE",
                                   "FFF", "GGG", "HHH", "III", "JJJ"])]
    reco_ledger.record(recs, "screen-sp500", "2026-06-11", path=p)
    assert reco_ledger.mark_executed("CCC", qty=80, path=p)
    assert not reco_ledger.mark_executed("ZZZ", path=p)        # unknown: no-op
    text = reco_ledger.summarize(path=p)
    # ALL ten open entries are listed (the old view truncated to 8).
    assert all(s in text for s in ("AAA", "JJJ"))
    assert "10 open (1 executed in your account)" in text
    assert "● 2026-06-11  CCC" in text and "x80" in text
    assert "○ 2026-06-11  AAA" in text                          # advisory-only


def test_wrap_never_truncates():
    from app.runner import _wrap

    long = ("The fundamental engine is real and durable: revenue re-accelerating "
            "for two consecutive quarters while operating margins inflect, and "
            "the moat read confirms a defensible niche position. " * 3)
    lines = _wrap(long, indent="      ")
    assert "".join(lines).replace(" ", "") == ("      " + long).replace(" ", "")[:0] \
        or " ".join(ln.strip() for ln in lines) == " ".join(long.split())
    assert all(len(ln) <= 100 for ln in lines)
    assert not any(ln.rstrip().endswith("...") for ln in lines)
    assert _wrap("") == [] and _wrap(None) == []


def test_weekly_trend_quality_rewards_confirmed_uptrends():
    from app.strategy import composite_score

    base = {"rs": 0.2, "mom6": 0.2, "mom3": 0.1, "above_200dma": True,
            "dist_high": -0.1, "earnings_gap": 0.0, "accel": 0.0, "ext20": 0.01}
    confirmed = composite_score({**base, "trend_quality": 1})
    bounce = composite_score({**base, "trend_quality": -1})
    assert confirmed > bounce          # confirmed structure beats a fragile bounce


def test_news_sentiment_scores_tone_and_flags_bearish():
    from system.data_plane.evidence import _news_sentiment

    bull = _news_sentiment(["Company beats earnings, raises guidance",
                            "Analyst upgrade as orders surge to record"])
    assert bull["tone"] == "bullish" and bull["bear_hits"] == 0
    bear = _news_sentiment(["SEC probe over accounting; shares plunge",
                            "Analyst downgrade after guidance cut"])
    assert bear["tone"] == "bearish" and bear["bearish_items"]
    assert _news_sentiment([])["available"] is False


def test_guardian_exits_on_thesis_contradicting_news():
    from system.agents.llm_client import MockLLMClient
    from system.agents.meta import GuardianAgent
    from system.config import SystemConfig

    g = GuardianAgent(MockLLMClient(), SystemConfig().models.framing)
    d = g.run({"symbol": "AAA", "evidence": {"news_sentiment": {
        "available": True, "tone": "bearish",
        "bearish_items": ["recall and federal probe announced"]}}})
    assert d.action == "exit" and "probe" in d.reason
    # Bullish/mixed news never forces an exit.
    assert g.run({"symbol": "AAA", "evidence": {"news_sentiment": {
        "available": True, "tone": "bullish", "bearish_items": []}}}).action == "hold"


def test_self_throttle_dormant_until_enough_data_then_derisks():
    from system.reflection.calibration import desk_throttle

    def calls(n, ret, conv=0.6):
        return [{"status": "evaluated", "return_pct": ret, "conviction": conv,
                 "evaluated_on": f"2026-06-{10 + i % 19:02d}"} for i in range(n)]

    # Thin record: dormant no matter how bad.
    assert desk_throttle(calls(5, -8.0))["active"] is False
    # Cold streak with enough evidence: de-risk (raise bar, cut gross).
    cold = desk_throttle(calls(12, -6.0))
    assert cold["active"] and cold["conviction_bump"] > 0 and cold["gross_scale"] < 1.0
    # Healthy record: never throttles (and never ADDS risk).
    warm = desk_throttle(calls(12, 7.0))
    assert warm["active"] is False and warm["gross_scale"] == 1.0


def test_correlation_diversify_drops_near_duplicates_only():
    import numpy as np
    from app.strategy import correlation_diversify

    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    rng = np.random.default_rng(3)
    base = np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    twin = base * (1 + rng.normal(0, 0.0005, 120))      # ~identical to base
    indep = np.cumprod(1 + rng.normal(0.001, 0.01, 120))
    closes = pd.DataFrame({"AAA": base, "ATWIN": twin, "BBB": indep}, index=idx)
    # AAA ranked first, its twin second, independent BBB third; k=2 should skip
    # the twin and take the independent name instead.
    out = correlation_diversify(["AAA", "ATWIN", "BBB"], closes, k=2)
    assert out == ["AAA", "BBB"]
    # k>=len always returns everyone (redundant ones just pushed to the back).
    assert set(correlation_diversify(["AAA", "ATWIN", "BBB"], closes, k=3)) == {
        "AAA", "ATWIN", "BBB"}
    # No data / single slot: graceful passthrough.
    assert correlation_diversify(["AAA", "BBB"], None, k=2) == ["AAA", "BBB"]


def test_growth_analyst_reads_estimate_revisions():
    from system.agents.analysts import GrowthAnalyst
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    m = SystemConfig().models.framing
    up = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 18.0, "eps_revision_90d_pct": 12.0, "num_analysts": 20}}}
    r = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "X", "evidence": up})
    assert any("REVISED UP" in p for p in r.positives)
    down = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 5.0, "eps_revision_90d_pct": -10.0, "num_analysts": 15}}}
    r2 = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "Y", "evidence": down})
    assert any("CUT" in c for c in r2.concerns) and r2.score < 0.5
    # Thin coverage (no analysts) -> the revision is ignored, not trusted.
    thin = {"fundamentals": {"available": True, "growth": {
        "revenue_growth_pct": 5.0, "eps_revision_90d_pct": 30.0, "num_analysts": 1}}}
    r3 = GrowthAnalyst(MockLLMClient(), m).run({"symbol": "Z", "evidence": thin})
    assert not any("REVISED UP" in p for p in r3.positives)


def test_skeptic_flags_poor_earnings_quality():
    from system.agents.core import SkepticAgent
    from system.agents.llm_client import MockLLMClient
    from system.config import SystemConfig

    crit = SkepticAgent(MockLLMClient(), SystemConfig().models.adversarial).run(
        {"symbol": "AAA",
         "evidence": {"earnings_quality": {"rating": "poor", "profit_margin_pct": 22.0,
                                           "fcf_margin_pct": 3.0, "accrual_gap_pp": 19.0}},
         "max_corr_to_book": 0.0, "min_read_confidence": 0.9, "priced_in": 0.0})
    assert any(o.kind == "earnings_quality" for o in crit.objections)


def test_watchlist_lifecycle_and_triggers(tmp_path):
    from app import watchlist as wl

    p = tmp_path / "watch.json"
    n = wl.add([{"symbol": "SITM", "reason": "PM pullback", "pullback_target": 630.0},
                {"symbol": "ARCB", "reason": "WATCH tier", "breakout_level": 150.0},
                {"symbol": "NOLEVELS", "reason": "ignored"}],     # no triggers: skipped
               "2026-06-11", path=p)
    assert n == 2
    assert len(wl.active("2026-06-12", path=p)) == 2
    assert wl.active("2026-08-01", path=p) == []                  # expired + pruned

    wl.add([{"symbol": "SITM", "reason": "PM pullback", "pullback_target": 630.0},
            {"symbol": "ARCB", "reason": "WATCH tier", "breakout_level": 150.0}],
           "2026-06-11", path=p)
    idx = pd.date_range("2026-05-01", periods=30, freq="B")
    closes = pd.DataFrame({"SITM": 660.0, "ARCB": 140.0}, index=idx)
    closes.loc[idx[-1], "SITM"] = 633.0                           # entered the window
    vols = pd.DataFrame(1_000_000.0, index=idx, columns=closes.columns)
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert [h["kind"] for h in hits] == ["pullback"]
    # Breakout requires the level AND unusual volume.
    closes.loc[idx[-1], "ARCB"] = 151.0
    vols.loc[idx[-1], "ARCB"] = 900_000.0                         # rvol < 1.5: no hit
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert not any(h["symbol"] == "ARCB" for h in hits)
    vols.loc[idx[-1], "ARCB"] = 2_000_000.0                       # rvol 2.0: hit
    hits = wl.watch_hits(wl.active("2026-06-12", path=p), closes, vols)
    assert any(h["symbol"] == "ARCB" and h["kind"] == "breakout" for h in hits)
    wl.remove({"ARCB"}, path=p)
    assert [it["symbol"] for it in wl.active("2026-06-12", path=p)] == ["SITM"]


def test_factor_weights_stacks_on_a_preset_base():
    from app.strategy import WEIGHT_PRESETS, factor_weights

    w = factor_weights(None, None, base=WEIGHT_PRESETS["timing"])
    assert w["timing"] == WEIGHT_PRESETS["timing"]["timing"]
    risk_off = factor_weights({"available": True, "above_200dma": False}, None,
                              base=WEIGHT_PRESETS["timing"])
    assert risk_off["rs"] < WEIGHT_PRESETS["timing"]["rs"]   # regime still applies
