"""Realistic, gap-aware trading costs (harness spec §7, master §11).

Applied to every simulated entry and exit:
  * commission (per-share, with a per-order minimum),
  * half the bid/ask spread per side,
  * slippage.

Gap-aware fills: when a session gaps through a resting stop or target, the fill
is the session OPEN, not the order price (master G16/G17). The backtest mirrors
live execution exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    half_spread_bps: float = 2.0     # half the round-trip spread, per side
    slippage_bps: float = 2.0        # adverse fill, per side

    def _frac(self) -> float:
        return (self.half_spread_bps + self.slippage_bps) / 10_000.0

    def commission(self, shares: int) -> float:
        return max(self.min_commission, abs(shares) * self.commission_per_share)

    def fill_price(self, ref_price: float, side: str) -> float:
        """Effective fill: pay the spread+slippage in the adverse direction.

        side='buy' fills above ref; side='sell' fills below ref.
        """
        f = self._frac()
        if side == "buy":
            return ref_price * (1 + f)
        if side == "sell":
            return ref_price * (1 - f)
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def round_trip_bps(self) -> float:
        """Total round-trip cost in bps of notional (spread+slippage, both sides)."""
        return 2 * (self.half_spread_bps + self.slippage_bps)


def gap_aware_exit(order_price: float, session_open: float, kind: str) -> float:
    """Return the realistic exit fill for a resting stop/target.

    kind='stop'   : a long stop. If the open gapped below the stop, fill at open.
    kind='target' : a long target. If the open gapped above the target, fill at open.
    Otherwise the order fills at its resting price (touched intrabar).
    """
    if kind == "stop":
        return min(order_price, session_open)      # gap down -> worse fill
    if kind == "target":
        return max(order_price, session_open)       # gap up -> better fill
    raise ValueError(f"kind must be 'stop' or 'target', got {kind!r}")


def net_pnl(entry_fill: float, exit_fill: float, shares: int, costs: CostModel) -> float:
    """P&L for a long round trip, net of commissions (spread already in fills)."""
    gross = (exit_fill - entry_fill) * shares
    return gross - costs.commission(shares) * 2
