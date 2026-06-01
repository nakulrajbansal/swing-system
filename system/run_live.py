"""End-to-end paper-trading engine (master Parts II-VII wired together).

Daily cadence: decide post-close on session d, execute at the d+1 open. Each
cycle:
    data plane (PIT)  ->  agent plane (specialists -> confluence -> trio)
                      ->  deterministic plane (Risk Governor -> Paper execution)
                      ->  fast loop (health scorecard -> kill switch, reduce-only)

Paper-only by default. The two keys are both enforced: a trade needs the PM to
choose ENTER *and* the Risk Governor to approve+size it. The orchestrator's
fail-safe means agent failures default to PASS, never into a trade.

    python -m system.run_live        # offline, deterministic (MockLLMClient)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from harness.data.loader import available_at_for_session
from harness.data.pit_store import PITStore
from system.agents.core import HypothesisAgent, PortfolioManagerAgent, SkepticAgent
from system.agents.llm_client import LLMClient, MockLLMClient
from system.agents.specialists import EdgeSpecialist
from system.config import DEFAULT_CONFIG, SystemConfig
from system.data_plane.indicators import last_atr
from system.execution.broker import PaperBroker
from system.monitoring.kill_switch import FLATTEN, HALT_NEW, KillSwitch
from system.monitoring.scorecard import HealthScorecard
from system.orchestrator import Orchestrator
from system.risk import clusters as cl
from system.risk.governor import GovernorContext, Position, RiskGovernor
from harness.signals import ALL_FREE_EDGES


@dataclass
class LiveResult:
    final_equity: float
    starting_equity: float
    closed_trades: list[dict]
    scorecard: dict
    n_cycles: int
    kill_events: list[str]

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.starting_equity - 1.0) * 100.0


class PaperTradingEngine:
    def __init__(self, store: PITStore, sector_map: dict, config: SystemConfig | None = None,
                 client: LLMClient | None = None, starting_equity: float = 100_000.0):
        self.store = store
        self.sector_map = sector_map
        self.cfg = config or DEFAULT_CONFIG
        self.client = client or MockLLMClient()
        self.broker = PaperBroker(starting_equity)
        self.governor = RiskGovernor(self.cfg)
        self.scorecard = HealthScorecard()
        self.killswitch = KillSwitch(self.cfg.kill)
        self.start_equity = starting_equity
        self._pending_conviction: dict = {}

        # Agent roster (model tiering per master §6).
        m = self.cfg.models
        self.specialists = [EdgeSpecialist(E(), self.client, m.framing) for E in ALL_FREE_EDGES]
        self.orchestrator = Orchestrator(
            store, self.specialists,
            HypothesisAgent(self.client, m.synthesis),
            SkepticAgent(self.client, m.adversarial),
            PortfolioManagerAgent(self.client, m.adversarial),
            self.cfg, price_lookup=self._price_lookup)

        # Fully-adjusted panels (outcomes/fills replay realized bars).
        last = available_at_for_session(self._last_session()) + pd.Timedelta(days=1)
        view = store.as_of(last)
        self.universe = view.universe()
        self.panels = {s: view.prices(s, adjust=True).set_index("date")
                       for s in self.universe}
        self.panels = {s: df for s, df in self.panels.items() if not df.empty}

    # -- helpers -----------------------------------------------------------
    def _last_session(self) -> pd.Timestamp:
        return pd.to_datetime(self.store.read_table("prices")["date"]).max()

    def _price_lookup(self, symbol, view):
        return view.prices(symbol)

    def _bars(self, date) -> dict[str, dict]:
        out = {}
        for s, df in self.panels.items():
            if date in df.index:
                r = df.loc[date]
                out[s] = {"open": float(r["open"]), "high": float(r["high"]),
                          "low": float(r["low"]), "close": float(r["close"])}
        return out

    def _marks(self, date) -> dict[str, float]:
        return {s: b["close"] for s, b in self._bars(date).items()}

    def _clusters(self, date) -> list[set]:
        truncated = {s: df.loc[df.index <= date].reset_index()
                     for s, df in self.panels.items()}
        mat = cl.returns_matrix(truncated, self.cfg.correlation_lookback)
        if mat.empty:
            return [{s} for s in self.panels]
        return cl.correlation_clusters(mat, self.cfg.correlation_threshold)

    # -- the cycle loop ----------------------------------------------------
    def run(self, start=None, end=None) -> LiveResult:
        sessions = self._sessions(start, end)
        equity_hist: list[float] = []
        kill_events: list[str] = []
        halted = False

        for i, d in enumerate(sessions):
            bars = self._bars(d)

            # 1. Execute orders submitted last close, then manage open positions.
            if i > 0:
                self.broker.on_session_open(bars)
                exits = self.broker.on_session(bars, time_stop_bars=20)
                for f in exits:
                    self._record_exit(f)

            # 2. Mark equity for the fast loop.
            equity = self.broker.equity(self._marks(d))
            equity_hist.append(equity)
            self.scorecard.mark_equity(equity)

            # 3. Fast loop: kill switch (reduce-only).
            daily = _pct(equity_hist, 1)
            weekly = _pct(equity_hist, 5)
            verdict = self.killswitch.evaluate(self.scorecard, daily_pnl_pct=daily,
                                               weekly_pnl_pct=weekly)
            if verdict.action == FLATTEN:
                kill_events.append(f"{d.date()}: FLATTEN {verdict.reasons}")
                self._flatten(d)
                halted = True
            elif verdict.action == HALT_NEW:
                halted = True

            # 4. Decide post-close; submit entries for the d+1 open (two keys).
            if not halted and i < len(sessions) - 1:
                self._decide_and_submit(d, equity, daily, weekly)

        final_equity = self.broker.equity(self._marks(sessions[-1])) if len(sessions) else self.start_equity
        return LiveResult(final_equity, self.start_equity, self.broker.closed_trades,
                          self.scorecard.summary(), len(sessions), kill_events)

    def _decide_and_submit(self, d, equity, daily, weekly):
        T = available_at_for_session(d)
        cycle = self.orchestrator.run_cycle(T)
        clusters = self._clusters(d)
        new_entries = 0
        for decision in cycle.decisions:
            if decision.action not in {"enter", "adjust"}:
                continue
            sym = decision.symbol
            panel = self.panels.get(sym)
            if panel is None or d not in panel.index:
                continue
            ref = float(panel.loc[d, "close"])           # stage-1 reference (post-close)
            atr = last_atr(panel.loc[panel.index <= d].reset_index(), self.cfg.sizing.atr_len)
            if atr != atr or atr <= 0:
                continue
            ctx = GovernorContext(
                equity=equity, reference_price=ref, atr_value=atr,
                sector=self.sector_map.get(sym, "?"),
                positions=self._positions(d), sector_map=self.sector_map,
                clusters=clusters, daily_pnl_pct=daily, weekly_pnl_pct=weekly,
                new_entries_this_cycle=new_entries)
            ticket = self.governor.evaluate(sym, "enter", ctx)
            if ticket.approved:
                band = self.cfg.sizing.entry_band_bps / 10_000.0
                self.broker.submit_entry(
                    sym, ticket.shares, band_low=ticket.entry * (1 - band),
                    band_high=ticket.entry * (1 + band), stop=ticket.stop,
                    target=ticket.target, sector=self.sector_map.get(sym, "?"))
                # remember conviction for calibration when the trade closes
                self._pending_conviction[sym] = decision.final_conviction
                new_entries += 1

    # -- position / trade bookkeeping -------------------------------------
    def _positions(self, d) -> list[Position]:
        marks = self._marks(d)
        out = []
        for sym, p in self.broker.positions().items():
            out.append(Position(sym, p.shares, p.entry, p.stop,
                                self.sector_map.get(sym, "?"), marks.get(sym, p.entry)))
        return out

    def _record_exit(self, fill):
        trade = next((t for t in reversed(self.broker.closed_trades)
                      if t["symbol"] == fill.symbol), None)
        if trade:
            conv = self._pending_conviction.pop(fill.symbol, None)
            self.scorecard.record_trade(trade["pnl"], conv)

    def _flatten(self, d):
        marks = self._marks(d)
        for sym in list(self.broker.positions()):
            f = self.broker.force_exit(sym, marks.get(sym, self.broker.positions()[sym].entry))
            if f:
                self._record_exit(f)

    def _sessions(self, start, end) -> pd.DatetimeIndex:
        idx = None
        for df in self.panels.values():
            idx = df.index if idx is None else idx.union(df.index)
        idx = idx if idx is not None else pd.DatetimeIndex([])
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        return idx


def _pct(hist: list[float], lag: int) -> float:
    if len(hist) <= lag or hist[-1 - lag] == 0:
        return 0.0
    return hist[-1] / hist[-1 - lag] - 1.0


def main():
    from harness.run import build_synthetic_store
    store, sector_map = build_synthetic_store()
    engine = PaperTradingEngine(store, sector_map)
    result = engine.run()
    print("\n" + "=" * 60)
    print("PAPER-TRADING ENGINE — end-to-end (synthetic, offline, MockLLM)")
    print("=" * 60)
    print(f"  cycles run      : {result.n_cycles}")
    print(f"  starting equity : {result.starting_equity:,.0f}")
    print(f"  final equity    : {result.final_equity:,.0f}  ({result.total_return_pct:+.1f}%)")
    print(f"  closed trades   : {len(result.closed_trades)}")
    print(f"  scorecard       : {result.scorecard}")
    if result.kill_events:
        print(f"  kill events     : {result.kill_events[:3]} ...")


if __name__ == "__main__":
    main()
