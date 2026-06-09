"""Persistent app configuration (API keys + run parameters).

Stored as JSON under the user's home (NOT in the repo, NOT in the bundle), so
secrets never get committed or shipped:  ~/.swing_system/config.json

Keys are applied to the process environment only when a run starts, so the rest
of the system (which reads ANTHROPIC_API_KEY etc.) needs no app-specific wiring.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".swing_system"
CONFIG_PATH = CONFIG_DIR / "config.json"

# Fields that are secrets — masked in the UI and never logged.
SECRET_FIELDS = {"anthropic_api_key", "alpaca_key_id", "alpaca_secret",
                 "reddit_client_id", "reddit_client_secret", "reddit_password"}


@dataclass
class AppConfig:
    # --- credentials (stored locally; used only if their feature is enabled) ---
    anthropic_api_key: str = ""          # real LLM agents (experimental, off by default)
    alpaca_key_id: str = ""              # live broker (gated)
    alpaca_secret: str = ""
    edgar_user_agent: str = ""           # required by SEC for EDGAR pulls
    reddit_client_id: str = ""           # Reddit app (reddit.com/prefs/apps) for social scan
    reddit_client_secret: str = ""
    reddit_username: str = ""            # optional (enables the password grant)
    reddit_password: str = ""

    # --- run parameters (the deterministic/synthetic pipeline that is wired) ---
    data_source: str = "synthetic"       # "synthetic" (wired) | "live" (reserved)
    ticker: str = ""                     # if set, recommendations analyze ONLY this stock
    verbose_agents: bool = True          # show each agent's prompt/inputs/output + consensus
    n_symbols: int = 8
    start_date: str = "2019-01-02"
    end_date: str = "2023-12-29"
    seed: int = 7
    starting_equity: float = 100_000.0
    oos_start: str = "2022-01-01"
    insider_history_quarters: int = 12   # SEC bulk insider quarters for historical validation
    filing_history_count: int = 12       # periodic 10-K/10-Q filings per stock for edge-1 validation
    momentum_hold_days: int = 10         # momentum swing: trading days to hold before exit
    momentum_max_positions: int = 1      # momentum swing: max concurrent positions
    reddit_top_k: int = 10               # top mentioned tickers to analyze with the model

    # --- feature toggles (default safe) ---
    use_llm_agents: bool = False         # experimental; spends tokens if a key is set
    only_validated_edges: bool = True    # live deliberation trades only edges that PASSED
    alpaca_env: str = "paper"            # "paper" (fake money) | "live" (REAL money)
    place_orders: bool = False           # submit approved deliberation orders to Alpaca
    enable_live_trading: bool = False    # extra gate: must be on to use env="live"
    learn_from_runs: bool = True         # reflect on closed trades + recall lessons across runs
    auto_approve_lessons: bool = True    # new lessons are active immediately (else review first)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                known = {k: data[k] for k in asdict(cls()) if k in data}
                return cls(**known)
            except Exception:
                pass
        return cls()

    def save(self) -> Path:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        try:
            os.chmod(CONFIG_PATH, 0o600)         # best-effort: owner-only (POSIX)
        except Exception:
            pass
        return CONFIG_PATH

    def apply_to_env(self) -> None:
        """Export credentials so the system's existing env-based wiring sees them."""
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.alpaca_key_id:
            os.environ["ALPACA_KEY_ID"] = self.alpaca_key_id
        if self.alpaca_secret:
            os.environ["ALPACA_SECRET"] = self.alpaca_secret
        if self.edgar_user_agent:
            os.environ["EDGAR_USER_AGENT"] = self.edgar_user_agent

    def redacted(self) -> dict:
        out = asdict(self)
        for k in SECRET_FIELDS:
            if out.get(k):
                out[k] = "****" + str(out[k])[-4:]
        return out
