"""The Risk Governor (master §10, invariants 1 & 5).

The second of the two keys: a trade opens only if the agents choose it AND the
Governor permits it. The Governor treats the PM's entry/stop/target as *requests*,
recomputes the stop from ATR, sizes by risk, then TRIMS to the most binding hard
cap. It may reject or trim; it may NEVER add risk, raise a limit, or scale up
beyond what sizing proposes (asymmetric autonomy). Loss halts block new entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from system.config import SystemConfig
from system.risk import clusters as cl
from system.risk.sizing import atr_stop, position_size


@dataclass
class Position:
    symbol: str
    shares: int
    entry: float
    stop: float
    sector: str
    price: float            # current mark

    @property
    def notional(self) -> float:
        return self.shares * self.price

    @property
    def open_risk(self) -> float:
        """Dollars at risk to the stop (never negative once stop > entry)."""
        return max(0.0, self.shares * (self.entry - self.stop))


@dataclass
class GovernorContext:
    equity: float
    reference_price: float            # entry request / actual fill
    atr_value: float
    sector: str
    positions: list[Position] = field(default_factory=list)
    sector_map: dict[str, str] = field(default_factory=dict)
    clusters: list[set] = field(default_factory=list)
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    new_entries_this_cycle: int = 0


@dataclass
class OrderTicket:
    symbol: str
    approved: bool
    shares: int = 0
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str = ""
    binding_cap: str = ""


class RiskGovernor:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.limits = config.limits
        self.sizing = config.sizing

    # -- the gate ----------------------------------------------------------
    def evaluate(self, symbol: str, action: str, ctx: GovernorContext) -> OrderTicket:
        if action == "pass":
            return OrderTicket(symbol, approved=False, reason="PM passed")

        # Loss halts block ALL new entries (reducing risk is always allowed
        # elsewhere; the Governor only ever gates *adding* risk).
        if ctx.daily_pnl_pct <= self.limits.daily_loss_halt:
            return OrderTicket(symbol, approved=False, reason="daily loss halt",
                               binding_cap="daily_loss_halt")
        if ctx.weekly_pnl_pct <= self.limits.weekly_loss_halt:
            return OrderTicket(symbol, approved=False, reason="weekly loss halt",
                               binding_cap="weekly_loss_halt")

        held = {p.symbol for p in ctx.positions}
        if symbol not in held and len(ctx.positions) >= self.limits.max_open_positions:
            return OrderTicket(symbol, approved=False, reason="max open positions",
                               binding_cap="max_open_positions")
        if ctx.new_entries_this_cycle >= self.limits.max_new_entries_per_cycle:
            return OrderTicket(symbol, approved=False, reason="max new entries/cycle",
                               binding_cap="max_new_entries_per_cycle")

        entry = ctx.reference_price
        if entry <= 0 or ctx.atr_value <= 0:
            return OrderTicket(symbol, approved=False, reason="bad reference/atr")

        # Governor recomputes stop + target from the actual reference.
        stop = atr_stop(entry, ctx.atr_value, self.sizing.atr_stop_mult)
        target = entry + self.sizing.reward_risk * (entry - stop)

        proposed = position_size(ctx.equity, entry, stop, self.limits.risk_per_trade)
        shares, binding = self._trim_to_caps(symbol, entry, stop, proposed, ctx)

        if shares < self.sizing.min_viable_shares:
            return OrderTicket(symbol, approved=False,
                               reason=f"below minimum viable size (cap: {binding})",
                               binding_cap=binding)
        return OrderTicket(symbol, approved=True, shares=shares, entry=entry,
                           stop=stop, target=target, binding_cap=binding,
                           reason="approved")

    # -- cap trimming (only ever reduces shares) ---------------------------
    def _trim_to_caps(self, symbol, entry, stop, shares, ctx) -> tuple[int, str]:
        eq = ctx.equity
        per_share_risk = entry - stop
        binding = "risk_per_trade"
        caps = {"risk_per_trade": shares}

        # Single-name notional.
        caps["max_single_name"] = int(self.limits.max_single_name * eq // entry)

        # Gross notional headroom (book-wide leverage ceiling). Without this the
        # per-trade and per-sector caps still let many names stack the book past
        # 1.0x equity — the leverage the live review kept flagging. Only trims.
        gross_now = sum(p.notional for p in ctx.positions if p.symbol != symbol)
        gross_head = max(0.0, self.limits.max_gross_exposure * eq - gross_now)
        caps["max_gross_exposure"] = int(gross_head // entry)

        # Sector notional headroom.
        sector_now = sum(p.notional for p in ctx.positions
                         if p.sector == ctx.sector and p.symbol != symbol)
        sector_head = max(0.0, self.limits.max_sector_exposure * eq - sector_now)
        caps["max_sector_exposure"] = int(sector_head // entry)

        # Portfolio heat headroom (open risk).
        heat_now = sum(p.open_risk for p in ctx.positions if p.symbol != symbol)
        heat_head = max(0.0, self.limits.max_portfolio_heat * eq - heat_now)
        caps["max_portfolio_heat"] = int(heat_head // per_share_risk) if per_share_risk > 0 else 0

        # Cluster heat headroom.
        cluster = cl.cluster_of(symbol, ctx.clusters)
        cluster_now = sum(p.open_risk for p in ctx.positions
                          if p.symbol in cluster and p.symbol != symbol)
        cluster_head = max(0.0, self.limits.max_cluster_heat * eq - cluster_now)
        caps["max_cluster_heat"] = int(cluster_head // per_share_risk) if per_share_risk > 0 else 0

        final = shares
        for name, cap in caps.items():
            if cap < final:
                final, binding = cap, name
        return max(0, final), binding
