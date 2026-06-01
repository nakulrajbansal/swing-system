"""
swing_bot.py
A modular swing-trading framework: data -> indicators -> signals -> risk -> backtest -> execution.

DESIGN PHILOSOPHY
-----------------
- Trade WITH the long-term trend (only longs when price > 200-day SMA).
- ENTER on a momentum pullback (RSI dips into oversold-within-uptrend, then turns up).
- SIZE by risk, not by gut (fixed % of equity at risk per trade).
- EXIT on an ATR stop, an R-multiple target, or a time stop. Whichever hits first.
- The execution layer defaults to PAPER. Live trading is a deliberate, separate step.

This is an educational scaffold, NOT investment advice. Backtested edges decay.
Account for fees and slippage (modeled below) before believing any result.

Requires: pandas, numpy, yfinance  ->  pip install pandas numpy yfinance
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. DATA LAYER
# ----------------------------------------------------------------------------
def fetch_data(symbol: str, start: str, end: Optional[str] = None,
               interval: str = "1d") -> pd.DataFrame:
    """Pull OHLCV history. For live trading, swap this for a broker feed."""
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.dropna()


# ----------------------------------------------------------------------------
# 2. INDICATORS
# ----------------------------------------------------------------------------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range: our volatility unit for stops and sizing."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ----------------------------------------------------------------------------
# 3. STRATEGY CONFIG + SIGNAL ENGINE
# ----------------------------------------------------------------------------
@dataclass
class Config:
    trend_len: int = 200       # long-term trend filter (SMA)
    pullback_ema: int = 20     # short EMA the pullback respects
    rsi_len: int = 14
    rsi_buy: float = 40.0      # "oversold within an uptrend"
    atr_len: int = 14
    atr_stop_mult: float = 2.0   # stop = entry - 2*ATR
    reward_risk: float = 2.0     # target = entry + (2 * risk)
    time_stop_bars: int = 15     # bail if the trade hasn't worked in N bars
    risk_per_trade: float = 0.01  # 1% of equity at risk per position
    starting_equity: float = 10_000.0
    fee_bps: float = 5.0          # round-trip cost assumption (5 bps/side)
    slippage_bps: float = 5.0     # adverse fill assumption


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["sma_trend"] = sma(out["close"], cfg.trend_len)
    out["ema_fast"] = ema(out["close"], cfg.pullback_ema)
    out["rsi"] = rsi(out["close"], cfg.rsi_len)
    out["atr"] = atr(out, cfg.atr_len)
    return out


def entry_signal(row, prev) -> bool:
    """
    Long entry: uptrend confirmed, momentum was oversold and is turning back up,
    price holding above the fast EMA (a healthy pullback, not a breakdown).
    """
    uptrend = row["close"] > row["sma_trend"]
    rsi_turning_up = (prev["rsi"] < 40) and (row["rsi"] > prev["rsi"])
    holding_ema = row["close"] > row["ema_fast"]
    return bool(uptrend and rsi_turning_up and holding_ema)


# ----------------------------------------------------------------------------
# 4. RISK MANAGER
# ----------------------------------------------------------------------------
def position_size(equity: float, entry: float, stop: float,
                  risk_per_trade: float) -> int:
    """Shares such that (entry - stop) * shares ~= risk_per_trade * equity."""
    per_share_risk = max(entry - stop, 1e-9)
    dollars_at_risk = equity * risk_per_trade
    return int(dollars_at_risk // per_share_risk)


# ----------------------------------------------------------------------------
# 5. BACKTESTER
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry: float
    stop: float
    target: float
    shares: int
    exit_date: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    reason: str = ""

    @property
    def pnl(self) -> float:
        return (self.exit - self.entry) * self.shares if self.exit else 0.0


def backtest(df: pd.DataFrame, cfg: Config):
    data = build_features(df, cfg).dropna()
    equity = cfg.starting_equity
    cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    open_trade: Optional[Trade] = None
    bars_held = 0
    trades: list[Trade] = []
    curve = []

    idx = data.index
    for i in range(1, len(data)):
        row, prev = data.iloc[i], data.iloc[i - 1]
        date, price = idx[i], row["close"]

        if open_trade is None:
            if entry_signal(row, prev):
                entry = price * (1 + cost)               # pay the spread on entry
                stop = entry - cfg.atr_stop_mult * row["atr"]
                target = entry + cfg.reward_risk * (entry - stop)
                shares = position_size(equity, entry, stop, cfg.risk_per_trade)
                if shares > 0:
                    open_trade = Trade(date, entry, stop, target, shares)
                    bars_held = 0
        else:
            bars_held += 1
            exit_price, reason = None, ""
            if row["low"] <= open_trade.stop:            # stop hit (intrabar)
                exit_price, reason = open_trade.stop, "stop"
            elif row["high"] >= open_trade.target:       # target hit
                exit_price, reason = open_trade.target, "target"
            elif bars_held >= cfg.time_stop_bars:        # time stop
                exit_price, reason = price, "time"

            if exit_price is not None:
                open_trade.exit = exit_price * (1 - cost)  # pay the spread on exit
                open_trade.exit_date = date
                open_trade.reason = reason
                equity += open_trade.pnl
                trades.append(open_trade)
                open_trade = None

        mark = equity + (open_trade.pnl if open_trade else 0.0)
        curve.append((date, mark))

    return trades, pd.Series(dict(curve), name="equity")


# ----------------------------------------------------------------------------
# 6. METRICS (the honest scorecard)
# ----------------------------------------------------------------------------
def report(trades: list[Trade], curve: pd.Series, cfg: Config) -> dict:
    if not trades:
        return {"trades": 0, "note": "No trades triggered."}
    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    ret = curve.pct_change().dropna()
    peak = curve.cummax()
    max_dd = ((curve - peak) / peak).min()
    sharpe = (ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    return {
        "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "avg_win": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(pnls[pnls <= 0].mean(), 2) if (pnls <= 0).any() else 0.0,
        "expectancy_per_trade": round(pnls.mean(), 2),
        "total_pnl": round(pnls.sum(), 2),
        "final_equity": round(curve.iloc[-1], 2),
        "total_return_pct": round((curve.iloc[-1] / cfg.starting_equity - 1) * 100, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe_annualized": round(sharpe, 2),
    }


# ----------------------------------------------------------------------------
# 7. EXECUTION LAYER (abstract; defaults to paper)
# ----------------------------------------------------------------------------
class Broker:
    """Subclass and implement these for live trading (Alpaca, IBKR, etc.)."""
    def buy(self, symbol, shares, stop, target): raise NotImplementedError
    def positions(self): raise NotImplementedError


class PaperBroker(Broker):
    def __init__(self): self.log = []
    def buy(self, symbol, shares, stop, target):
        self.log.append(("BUY", symbol, shares, stop, target))
        print(f"[PAPER] BUY {shares} {symbol} stop={stop:.2f} target={target:.2f}")
    def positions(self): return self.log


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    df = fetch_data("SPY", start="2015-01-01")
    trades, curve = backtest(df, cfg)
    print("\n=== Backtest report (SPY, daily) ===")
    for k, v in report(trades, curve, cfg).items():
        print(f"{k:>22}: {v}")
    print("\nReminder: in-sample results overstate live performance. "
          "Validate out-of-sample and on multiple symbols before trusting this.")
