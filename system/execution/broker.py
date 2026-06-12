"""Brokers (master §11).

`PaperBroker` is the default and is fully deterministic: it fills bracket orders
against the bars it is fed, with **gap-aware** stop/target fills (a session that
gaps through a resting order fills at the OPEN, not the order price). The
protective stop rests at the broker immediately after fill.

`AlpacaBroker` is a real-broker adapter that is GATED: it refuses to construct
unless live trading is explicitly enabled AND credentials are present (master
invariant 5, asymmetric autonomy — automation may never turn real-money trading
on by itself). Its order methods are left to be wired by a human operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from harness.study.costs import CostModel, gap_aware_exit


@dataclass
class BrokerPosition:
    symbol: str
    shares: int
    entry: float
    stop: float
    target: float
    sector: str = "?"
    bars_held: int = 0


@dataclass
class Fill:
    symbol: str
    side: str            # "buy" | "sell"
    shares: int
    price: float
    reason: str          # "entry" | "stop" | "target" | "time" | "guardian"


@dataclass
class _PendingEntry:
    symbol: str
    shares: int
    band_low: float
    band_high: float      # marketable-limit ceiling
    stop: float
    target: float
    sector: str


class Broker(ABC):
    @abstractmethod
    def submit_entry(self, symbol, shares, band_low, band_high, stop, target, sector): ...
    @abstractmethod
    def positions(self) -> dict[str, BrokerPosition]: ...
    @abstractmethod
    def cancel_pending(self, symbol): ...


class PaperBroker(Broker):
    def __init__(self, starting_equity: float = 100_000.0, costs: CostModel | None = None):
        self.cash = starting_equity
        self.costs = costs or CostModel()
        self._positions: dict[str, BrokerPosition] = {}
        self._pending: dict[str, _PendingEntry] = {}
        self.fills: list[Fill] = []
        self.closed_trades: list[dict] = []

    # -- order entry -------------------------------------------------------
    def submit_entry(self, symbol, shares, band_low, band_high, stop, target, sector="?"):
        if shares <= 0 or symbol in self._positions:
            return
        self._pending[symbol] = _PendingEntry(symbol, shares, band_low, band_high,
                                              stop, target, sector)

    def cancel_pending(self, symbol):
        self._pending.pop(symbol, None)

    def positions(self) -> dict[str, BrokerPosition]:
        return self._positions

    # -- the daily fill engine --------------------------------------------
    def on_session_open(self, bars: dict[str, dict]) -> list[Fill]:
        """Stage-2 entry validation: fill pending entries against today's open.

        `bars[symbol]` = {open, high, low, close}. Marketable-limit semantics:
        fill only if the open is at/below the band ceiling; size against the
        ACTUAL fill; rest the protective stop immediately. Otherwise cancel.
        """
        session_fills = []
        for symbol, pend in list(self._pending.items()):
            bar = bars.get(symbol)
            self._pending.pop(symbol, None)
            if bar is None:
                continue
            open_ = bar["open"]
            if open_ > pend.band_high:                  # gapped above the band -> cancel
                continue
            fill_price = self.costs.fill_price(open_, "buy")
            cost = self.costs.commission(pend.shares)
            self.cash -= fill_price * pend.shares + cost
            self._positions[symbol] = BrokerPosition(
                symbol, pend.shares, fill_price, pend.stop, pend.target, pend.sector)
            f = Fill(symbol, "buy", pend.shares, fill_price, "entry")
            self.fills.append(f)
            session_fills.append(f)
        return session_fills

    def on_session(self, bars: dict[str, dict], time_stop_bars: int = 20) -> list[Fill]:
        """Manage open positions for one session: gap-aware stop/target/time exits."""
        exits = []
        for symbol, pos in list(self._positions.items()):
            bar = bars.get(symbol)
            if bar is None:
                continue
            pos.bars_held += 1
            exit_price, reason = None, ""
            if bar["low"] <= pos.stop:                       # stop touched
                exit_price = gap_aware_exit(pos.stop, bar["open"], "stop")
                reason = "stop"
            elif bar["high"] >= pos.target:                  # target touched
                exit_price = gap_aware_exit(pos.target, bar["open"], "target")
                reason = "target"
            elif pos.bars_held >= time_stop_bars:            # time stop
                exit_price, reason = bar["close"], "time"
            if exit_price is not None:
                self._close(pos, exit_price, reason)
                exits.append(Fill(symbol, "sell", pos.shares, exit_price, reason))
        return exits

    def force_exit(self, symbol: str, price: float, reason: str = "guardian") -> Fill | None:
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        self._close(pos, price, reason)
        return Fill(symbol, "sell", pos.shares, price, reason)

    def _close(self, pos: BrokerPosition, ref_price: float, reason: str):
        fill_price = self.costs.fill_price(ref_price, "sell")
        cost = self.costs.commission(pos.shares)
        self.cash += fill_price * pos.shares - cost
        self.closed_trades.append({
            "symbol": pos.symbol, "shares": pos.shares, "entry": pos.entry,
            "exit": fill_price, "reason": reason,
            "pnl": (fill_price - pos.entry) * pos.shares - cost,
            "bars_held": pos.bars_held,
        })
        self.fills.append(Fill(pos.symbol, "sell", pos.shares, fill_price, reason))
        self._positions.pop(pos.symbol, None)

    def equity(self, marks: dict[str, float]) -> float:
        mtm = sum(p.shares * marks.get(p.symbol, p.entry) for p in self._positions.values())
        return self.cash + mtm


class AlpacaBroker(Broker):
    """Real Alpaca adapter (REST). Two environments:

      * ``env="paper"`` -> paper-api.alpaca.markets (fake money; default, safe).
      * ``env="live"``  -> api.alpaca.markets (REAL money) and requires
        ``confirm_live=True`` to construct (asymmetric-autonomy invariant: a human
        must deliberately opt into real-money execution).

    Entries are submitted as marketable-limit BRACKET orders (attached stop +
    target), mirroring the deterministic plane's two-stage entry. Uses `requests`
    so nothing extra needs bundling.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, key_id: str | None, secret: str | None, *,
                 env: str = "paper", confirm_live: bool = False):
        if env not in {"paper", "live"}:
            raise ValueError(f"env must be 'paper' or 'live', got {env!r}")
        if env == "live" and not confirm_live:
            raise RuntimeError(
                "Refusing to construct a LIVE (real-money) Alpaca broker without "
                "confirm_live=True. Use env='paper', or explicitly confirm live."
            )
        if not key_id or not secret:
            raise RuntimeError("Alpaca credentials missing (key_id/secret required).")
        self.env = env
        self.base = self.LIVE_URL if env == "live" else self.PAPER_URL
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}

    # -- REST plumbing -----------------------------------------------------
    def _req(self, method: str, path: str, payload: dict | None = None):
        import requests  # lazy
        r = requests.request(method, self.base + path, headers=self._headers,
                             json=payload, timeout=30)
        if r.status_code >= 300:
            raise RuntimeError(f"Alpaca {method} {path} -> {r.status_code}: {r.text[:200]}")
        return r.json() if r.text else {}

    def account(self) -> dict:
        """Raw /v2/account (status, buying_power, etc.). Read-only."""
        return self._req("GET", "/v2/account")

    def is_tradable(self) -> tuple[bool, str]:
        a = self.account()
        ok = (a.get("status") == "ACTIVE" and not a.get("trading_blocked")
              and not a.get("account_blocked"))
        reason = (f"status={a.get('status')} trading_blocked={a.get('trading_blocked')} "
                  f"account_blocked={a.get('account_blocked')} "
                  f"buying_power={a.get('buying_power')}")
        return ok, reason

    # -- orders ------------------------------------------------------------
    def submit_entry(self, symbol, shares, band_low, band_high, stop, target, sector="?"):
        if shares <= 0:
            return None
        payload = {
            "symbol": symbol, "qty": str(int(shares)), "side": "buy",
            "type": "limit", "limit_price": round(float(band_high), 2),
            "time_in_force": "day", "order_class": "bracket",
            "take_profit": {"limit_price": round(float(target), 2)},
            "stop_loss": {"stop_price": round(float(stop), 2)},
        }
        return self._req("POST", "/v2/orders", payload)

    def positions(self) -> dict[str, BrokerPosition]:
        out: dict[str, BrokerPosition] = {}
        for p in self._req("GET", "/v2/positions"):
            out[p["symbol"]] = BrokerPosition(
                p["symbol"], int(float(p["qty"])), float(p["avg_entry_price"]),
                stop=0.0, target=0.0)
        return out

    def cancel_pending(self, symbol):
        for o in self._req("GET", "/v2/orders?status=open"):
            if o.get("symbol") == symbol:
                self._req("DELETE", f"/v2/orders/{o['id']}")

    def submit_buy_with_stop(self, symbol, shares, limit, stop):
        """Marketable-limit BUY with an attached protective STOP (no profit
        target) — for a momentum ride exited on time rather than at a target."""
        if shares <= 0:
            return None
        payload = {
            "symbol": symbol, "qty": str(int(shares)), "side": "buy",
            "type": "limit", "limit_price": round(float(limit), 2),
            "time_in_force": "day", "order_class": "oto",
            "stop_loss": {"stop_price": round(float(stop), 2)},
        }
        return self._req("POST", "/v2/orders", payload)

    def submit_manual(self, symbol, qty, side="buy", order_type="market",
                      limit_price=None, stop=None, target=None, tif="day"):
        """A flexible single order for the GUI's selective-execution panel: market
        or limit, with an optional protective stop and/or take-profit (bracket/OTO).
        Returns the Alpaca order dict (or an {'error': ...})."""
        qty = int(qty)
        if qty <= 0:
            return {"error": "quantity must be > 0"}
        payload = {"symbol": symbol, "qty": str(qty), "side": side,
                   "type": order_type, "time_in_force": tif}
        if order_type == "limit":
            if not limit_price:
                return {"error": "limit order needs a limit price"}
            payload["limit_price"] = round(float(limit_price), 2)
        if stop and target:
            payload.update({"order_class": "bracket", "time_in_force": "gtc",
                            "take_profit": {"limit_price": round(float(target), 2)},
                            "stop_loss": {"stop_price": round(float(stop), 2)}})
        elif stop:
            payload.update({"order_class": "oto", "time_in_force": "gtc",
                            "stop_loss": {"stop_price": round(float(stop), 2)}})
        return self._req("POST", "/v2/orders", payload)

    def fills(self, limit: int = 100) -> list[dict]:
        """Recent FILL activities (both sides), newest first — the account's
        actual trade history."""
        out = self._req("GET",
                        f"/v2/account/activities?activity_types=FILL&page_size={int(limit)}")
        return out or []

    def submit_exit_orders(self, symbol, qty, stop=None, target=None):
        """Protective exit orders for an EXISTING long position: an OCO pair
        (take-profit limit + stop-loss) when both levels exist, else a single
        GTC sell stop / sell limit. Risk-reducing only — sells what is already
        held, never opens exposure."""
        qty = int(qty)
        if qty <= 0 or not (stop or target):
            return {"error": "nothing to place"}
        if stop and target:
            payload = {"symbol": symbol, "qty": str(qty), "side": "sell",
                       "type": "limit", "time_in_force": "gtc",
                       "order_class": "oco",
                       "take_profit": {"limit_price": round(float(target), 2)},
                       "stop_loss": {"stop_price": round(float(stop), 2)}}
        elif stop:
            payload = {"symbol": symbol, "qty": str(qty), "side": "sell",
                       "type": "stop", "stop_price": round(float(stop), 2),
                       "time_in_force": "gtc"}
        else:
            payload = {"symbol": symbol, "qty": str(qty), "side": "sell",
                       "type": "limit", "limit_price": round(float(target), 2),
                       "time_in_force": "gtc"}
        return self._req("POST", "/v2/orders", payload)

    def close_position(self, symbol):
        """Cancel any open orders for the symbol, then liquidate the position at
        market (Alpaca DELETE /v2/positions/{symbol})."""
        self.cancel_pending(symbol)
        try:
            return self._req("DELETE", f"/v2/positions/{symbol}")
        except Exception:
            return None
