"""Synthetic point-in-time store fixtures for the leakage and corp-action tests.

A deterministic, hand-built timeline with known event/ex dates placed both
before and after the decision times the tests probe. No network, no real data
(harness spec §9).

Timeline (2026 business days):
  * AAA  : full price ramp; 2-for-1 split ex 2026-03-16; $1.00 dividend ex 2026-04-15.
           close[i] = 100 + i (raw) so prior closes are exactly known.
  * BBB  : member 2026-01-01 .. delisted 2026-04-01 (survivorship-free check).
  * CCC  : not listed until 2026-05-01 (not-yet-listed check).
  * filings / form4 / news / links with available_at straddling the probes.
"""

from __future__ import annotations

import pandas as pd
import pytest

START = pd.Timestamp("2026-01-02")
END = pd.Timestamp("2026-05-29")
SESSIONS = pd.bdate_range(START, END)

SPLIT_EX = pd.Timestamp("2026-03-16")     # Monday; ratio 2.0 (2-for-1)
SPLIT_RATIO = 2.0
DIV_EX = pd.Timestamp("2026-04-15")       # Wednesday; $1.00 cash dividend
DIV_AMOUNT = 1.0


def _price_ramp(symbol: str, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    close = [100.0 + i for i in range(len(sessions))]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": sessions,
            "open": [c - 0.5 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1000 + i for i in range(len(sessions))],
        }
    )


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def store(tmp_path):
    from harness.data import PITStore

    st = PITStore(tmp_path / "pit")

    aaa = _price_ramp("AAA", SESSIONS)
    bbb = _price_ramp("BBB", SESSIONS[SESSIONS < "2026-04-01"])
    st.write_prices(pd.concat([aaa, bbb], ignore_index=True))

    st.write_corp_actions(
        pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "ex_date": [SPLIT_EX, DIV_EX],
                "type": ["split", "dividend"],
                "ratio_or_amount": [SPLIT_RATIO, DIV_AMOUNT],
            }
        )
    )

    st.write_constituents(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "start_date": [START, START, pd.Timestamp("2026-05-01")],
                "end_date": [pd.NaT, pd.Timestamp("2026-04-01"), pd.NaT],
            }
        )
    )

    st.write_filings(
        pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "cik": ["0001", "0001"],
                "form_type": ["10-K", "10-Q"],
                "available_at": [_utc("2026-02-10 14:00"), _utc("2026-04-20 21:00")],
                "accession": ["acc-1", "acc-2"],
                "doc_uri": ["uri-1", "uri-2"],
                "section_text_riskfactors": ["risk old", "risk new"],
                "section_text_mdna": ["mdna old", "mdna new"],
            }
        )
    )

    st.write_form4(
        pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "cik": ["0001", "0001"],
                "available_at": [_utc("2026-03-01 13:30"), _utc("2026-05-04 13:30")],
                "insider_role": ["CEO", "CFO"],
                "txn_code": ["P", "P"],
                "shares": [10000, 5000],
                "value": [1_000_000.0, 500_000.0],
            }
        )
    )

    st.write_news(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "available_at": [_utc("2026-02-20 18:00"), _utc("2026-03-25 12:00")],
                "headline": ["AAA news", "BBB news"],
                "body_uri": ["news-1", "news-2"],
                "source": ["wire", "wire"],
            }
        )
    )

    st.write_links(
        pd.DataFrame(
            {
                "focal_symbol": ["AAA"],
                "linked_symbol": ["BBB"],
                "link_type": ["supplier"],
                "source_filing": ["acc-1"],
                "available_at": [_utc("2026-02-15 14:00")],
            }
        )
    )

    return st


@pytest.fixture
def synth_store(tmp_path):
    """A small synthetic universe for event-study / system / e2e tests (fast)."""
    from harness.data.loader import SyntheticConfig, SyntheticLoader
    from harness.data.pit_store import PITStore

    st = PITStore(tmp_path / "synth")
    cfg = SyntheticConfig(n_symbols=6, start="2020-01-02", end="2022-12-30", seed=3)
    loader = SyntheticLoader(st, cfg)
    loader.load_all()
    return st, loader.sector_map()
