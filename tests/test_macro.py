"""Macro analyst + its injection into the deliberation."""

from system.agents.analysts import MacroAnalyst
from system.agents.llm_client import MockLLMClient
from system.config import SystemConfig

M = SystemConfig().models
CLIENT = MockLLMClient()

_SUPPORTIVE = {"available": True, "equity_regime": "uptrend", "vix": 13, "vix_state": "calm",
               "rates_trend": "falling", "credit": "risk-on", "usd_trend": "flat",
               "cyclical_vs_defensive": "cyclicals leading (risk-on)", "score": 2.4,
               "summary": "SUPPORTIVE backdrop."}
_HOSTILE = {"available": True, "equity_regime": "downtrend", "vix": 32, "vix_state": "stressed",
            "rates_trend": "rising", "credit": "risk-off", "usd_trend": "strengthening",
            "cyclical_vs_defensive": "defensives leading (risk-off)", "score": -2.5,
            "summary": "HOSTILE backdrop."}


def test_macro_analyst_reads_backdrop():
    a = MacroAnalyst(CLIENT, M.framing)
    up = a.run({"symbol": "_M_", "evidence": {"macro": _SUPPORTIVE}})
    down = a.run({"symbol": "_M_", "evidence": {"macro": _HOSTILE}})
    assert up.domain == "macro" and up.stance == "bullish" and up.score > down.score
    assert down.stance == "bearish"
    assert any("credit risk-on" in p for p in up.positives)
    assert any("downtrend" in c for c in down.concerns)
    # No snapshot -> neutral, never crashes.
    none = a.run({"symbol": "_M_", "evidence": {"macro": {"available": False}}})
    assert none.stance == "neutral"


def test_macro_injected_into_deliberation(synth_store):
    import dataclasses
    from harness.data.loader import available_at_for_session
    from system.run_live import PaperTradingEngine
    import pandas as pd

    store, sector_map = synth_store
    macro = _SUPPORTIVE
    read = dataclasses.asdict(MacroAnalyst(CLIENT, M.framing).run(
        {"symbol": "_M_", "evidence": {"macro": macro}}))
    eng = PaperTradingEngine(store, sector_map, macro=macro, macro_read=read)
    sess = pd.to_datetime(store.read_table("prices")["date"]).max()
    sym = eng.universe[0]
    _cand, _dec, tr = eng.orchestrator.deliberate_symbol(available_at_for_session(sess), sym)
    agents = [s["agent"] for s in tr["steps"]]
    assert "macro_analyst" in agents                 # macro read shown
    assert tr["evidence"].get("macro", {}).get("available")  # snapshot in evidence
