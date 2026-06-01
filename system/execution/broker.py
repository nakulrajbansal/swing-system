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
    """Real-broker adapter — GATED. Live trading requires explicit human enable.

    Construction fails unless ``enable_live=True`` AND both API keys are present.
    Order methods are intentionally not implemented here: wiring real-money
    execution is a deliberate human act (master invariants 3 & 5).
    """

    def __init__(self, key_id: str | None, secret: str | None, *, enable_live: bool = False):
        if not enable_live:
            raise RuntimeError(
                "AlpacaBroker is disabled. Live trading is opt-in only: pass "
                "enable_live=True and supply credentials. Default to PaperBroker."
            )
        if not key_id or not secret:
            raise RuntimeError("Alpaca credentials missing (key_id/secret required).")
        self.key_id, self.secret = key_id, secret  # wiring left to a human operator

    def submit_entry(self, *a, **k):
        raise NotImplementedError("Wire Alpaca order submission deliberately (human gate).")

    def positions(self):
        raise NotImplementedError

    def cancel_pending(self, symbol):
        raise NotImplementedError
