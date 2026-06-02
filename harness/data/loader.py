"""Data loader: populate the PIT store with prices, filings, Form 4, links, news.

Two paths:

* :class:`SyntheticLoader` — deterministic, offline, reproducible. Generates a
  small universe with planted structure so the whole harness/system runs e2e in
  tests with no network or keys. It plants a *modest* post-event drift on Edge-1
  trigger days so validation has something real to find (and noise elsewhere so
  the pass/kill bar is meaningful). Clearly synthetic; not market data.

* Real adapters (:func:`fetch_prices_stooq`, :func:`fetch_edgar_submissions`) —
  network-only, used by the live data plane. Each sets ``available_at`` from the
  event time plus a conservative latency buffer, mapping post-close information
  to the next session (master §9). They are import-safe; network deps are
  imported lazily so offline runs never touch them.

The cardinal PIT rule is unchanged: every written row carries a tz-aware UTC
``available_at`` (events) or a session ``date``/``ex_date``; the store enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from harness.data import calendar as cal
from harness.data.pit_store import PITStore

# Conservative buffer: information observed in a session is treated as available
# only at that session's close (post-close decision cadence, master §9/§11).
SESSION_CLOSE_ET = "16:00"


def available_at_for_session(session: pd.Timestamp) -> pd.Timestamp:
    """UTC instant at which session-d information is usable: that session's close."""
    local = pd.Timestamp(f"{pd.Timestamp(session).date()} {SESSION_CLOSE_ET}",
                         tz="America/New_York")
    return local.tz_convert("UTC")


# --------------------------------------------------------------------------
# Synthetic, offline, reproducible universe
# --------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    n_symbols: int = 8
    start: str = "2019-01-02"
    end: str = "2023-12-29"
    seed: int = 7
    planted_drift: float = 0.10    # window drift after an Edge-1 event, x text-change
    filing_every: int = 63         # ~quarterly filing cadence (sessions)


SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY"]


class SyntheticLoader:
    def __init__(self, store: PITStore, cfg: SyntheticConfig | None = None):
        self.store = store
        self.cfg = cfg or SyntheticConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.sessions = cal.sessions(self.cfg.start, self.cfg.end)
        self.symbols = [f"SYN{i:02d}" for i in range(self.cfg.n_symbols)]

    # -- public ------------------------------------------------------------
    def load_all(self) -> None:
        filing_days = self._generate_prices_and_filings()
        self._generate_form4(filing_days)
        self._generate_links_and_news(filing_days)
        self._generate_constituents()
        # Sector ETF benchmarks (flat-ish, low vol) for abnormal-return adjustment.
        self._generate_benchmarks()

    # -- internals ---------------------------------------------------------
    def _generate_prices_and_filings(self) -> dict[str, list[pd.Timestamp]]:
        sess = self.sessions
        n = len(sess)
        all_prices, all_filings = [], []
        filing_days: dict[str, list[pd.Timestamp]] = {}

        for s_i, sym in enumerate(self.symbols):
            # Filing trigger sessions for this symbol (quarterly, phase-shifted).
            f_idx = list(range(20 + (s_i * 7) % self.cfg.filing_every, n, self.cfg.filing_every))
            filing_days[sym] = [sess[i] for i in f_idx]

            # Daily returns: small drift + noise. The PLANTED edge: the 10 sessions
            # after a filing drift up in proportion to how much the filing's text
            # *changed* vs the prior filing (continuous materiality). A faithful
            # deterministic Edge 1 should recover this; noise edges should not.
            mu, sigma = 0.0002, 0.018
            rets = self.rng.normal(mu, sigma, n)
            materiality = {i: float(self.rng.uniform(0.0, 1.0)) for i in f_idx}
            text_change = {}                          # |Δ materiality| vs prior filing
            prev_m = None
            for i in f_idx:
                m = materiality[i]
                change = 0.0 if prev_m is None else abs(m - prev_m)
                text_change[i] = change
                prev_m = m
                end = min(i + 10, n)
                rets[i + 1:end] += self.cfg.planted_drift * change / 9.0
            price = 50.0 * np.exp(np.cumsum(rets))
            close = np.round(price, 2)
            opn = np.round(close * (1 + self.rng.normal(0, 0.002, n)), 2)
            high = np.round(np.maximum(opn, close) * (1 + np.abs(self.rng.normal(0, 0.004, n))), 2)
            low = np.round(np.minimum(opn, close) * (1 - np.abs(self.rng.normal(0, 0.004, n))), 2)
            vol = (1_000_000 + self.rng.integers(0, 500_000, n)).astype(int)

            all_prices.append(pd.DataFrame({
                "symbol": sym, "date": sess, "open": opn, "high": high,
                "low": low, "close": close, "volume": vol,
            }))

            for i in f_idx:
                sess_day = sess[i]
                m = materiality[i]
                # Text length encodes materiality; Edge 1 sees only the text and
                # recovers the change from the length delta vs the prior filing.
                n_risk = 20 + int(round(m * 80))
                n_mdna = 20 + int(round(m * 80))
                all_filings.append({
                    "symbol": sym, "cik": f"{s_i:010d}", "form_type": "10-Q",
                    "available_at": available_at_for_session(sess_day),
                    "accession": f"{sym}-{sess_day.date()}",
                    "doc_uri": f"synthetic://{sym}/{sess_day.date()}",
                    "section_text_riskfactors": ("risk " * n_risk),
                    "section_text_mdna": ("mdna " * n_mdna),
                })

        self.store.write_prices(pd.concat(all_prices, ignore_index=True))
        self.store.write_filings(pd.DataFrame(all_filings))
        return filing_days

    def _generate_form4(self, filing_days: dict[str, list[pd.Timestamp]]) -> None:
        rows = []
        for s_i, sym in enumerate(self.symbols):
            for sess_day in filing_days[sym][::2]:   # cluster buys near material filings
                for role in ("CEO", "CFO", "Director"):
                    rows.append({
                        "symbol": sym, "cik": f"{s_i:010d}",
                        "available_at": available_at_for_session(sess_day),
                        "insider_role": role, "txn_code": "P",
                        "shares": int(self.rng.integers(5_000, 50_000)),
                        "value": float(self.rng.integers(200_000, 2_000_000)),
                    })
        self.store.write_form4(pd.DataFrame(rows))

    def _generate_links_and_news(self, filing_days: dict[str, list[pd.Timestamp]]) -> None:
        links, news = [], []
        for i, sym in enumerate(self.symbols):
            linked = self.symbols[(i + 1) % len(self.symbols)]
            first_filing = filing_days[sym][0]
            links.append({
                "focal_symbol": sym, "linked_symbol": linked,
                "link_type": "supplier", "source_filing": f"{sym}-{first_filing.date()}",
                "available_at": available_at_for_session(first_filing),
            })
            for sess_day in filing_days[sym]:
                news.append({
                    "symbol": sym, "available_at": available_at_for_session(sess_day),
                    "headline": f"{sym} quarterly update", "body_uri": f"news://{sym}",
                    "source": "synthetic",
                })
        self.store.write_links(pd.DataFrame(links))
        self.store.write_news(pd.DataFrame(news))

    def _generate_constituents(self) -> None:
        start = self.sessions[0]
        self.store.write_constituents(pd.DataFrame({
            "symbol": self.symbols,
            "start_date": start,
            "end_date": pd.NaT,
        }))

    def _generate_benchmarks(self) -> None:
        sess = self.sessions
        n = len(sess)
        rows = []
        for etf in SECTORS:
            rets = self.rng.normal(0.0003, 0.009, n)
            close = np.round(100.0 * np.exp(np.cumsum(rets)), 2)
            rows.append(pd.DataFrame({
                "symbol": etf, "date": sess, "open": close, "high": close * 1.005,
                "low": close * 0.995, "close": close,
                "volume": np.full(n, 5_000_000, dtype=int),
            }))
        self.store.write_prices(pd.concat(rows, ignore_index=True))

    def sector_map(self) -> dict[str, str]:
        return {sym: SECTORS[i % len(SECTORS)] for i, sym in enumerate(self.symbols)}


# --------------------------------------------------------------------------
# Real adapters (network; imported lazily so offline runs are unaffected)
# --------------------------------------------------------------------------
def fetch_prices_yahoo(symbol: str, start: str, end: str | None = None):
    """Free daily data from Yahoo via yfinance. No API key.

    Returns (prices_df, corp_actions_df) ready for the PIT store:
      * prices: RAW (unadjusted) OHLCV + session date — the store keeps raw prices
        as ground truth and back-adjusts as-of-T (master §9), so we must NOT use
        Yahoo's pre-adjusted 'Adj Close'.
      * corp_actions: splits and cash dividends with their ex-dates, which drive
        the as-of-T adjustment.
    """
    import yfinance as yf  # lazy

    h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False, actions=True)
    if h.empty:
        cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols), pd.DataFrame(
            columns=["symbol", "ex_date", "type", "ratio_or_amount"])

    idx = h.index.tz_localize(None).normalize() if h.index.tz is not None else h.index.normalize()
    prices = pd.DataFrame({
        "symbol": symbol, "date": idx,
        "open": h["Open"].to_numpy(), "high": h["High"].to_numpy(),
        "low": h["Low"].to_numpy(), "close": h["Close"].to_numpy(),
        "volume": h["Volume"].astype("int64").to_numpy(),
    }).dropna()

    actions = []
    if "Stock Splits" in h:
        for d, r in h["Stock Splits"][h["Stock Splits"] > 0].items():
            actions.append({"symbol": symbol, "ex_date": pd.Timestamp(d).tz_localize(None).normalize(),
                            "type": "split", "ratio_or_amount": float(r)})
    if "Dividends" in h:
        for d, a in h["Dividends"][h["Dividends"] > 0].items():
            actions.append({"symbol": symbol, "ex_date": pd.Timestamp(d).tz_localize(None).normalize(),
                            "type": "dividend", "ratio_or_amount": float(a)})
    corp = pd.DataFrame(actions, columns=["symbol", "ex_date", "type", "ratio_or_amount"])
    return prices, corp


def fetch_edgar_submissions(cik: str, user_agent: str) -> pd.DataFrame:
    """EDGAR submissions index -> store-ready filings rows (metadata only).

    `user_agent` is required by SEC fair-access policy (e.g. "you@example.com").
    Document text fetching is left to the live data plane's document retriever.
    """
    import requests  # lazy

    cik10 = f"{int(cik):010d}"
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    df = pd.DataFrame(recent)
    filed = pd.to_datetime(df["filingDate"])
    # Conservative availability: close of the filing date's session.
    df["available_at"] = filed.map(lambda d: available_at_for_session(d))
    df["symbol"] = None
    df["cik"] = cik10
    df = df.rename(columns={"form": "form_type", "accessionNumber": "accession",
                            "primaryDocument": "doc_uri"})
    df["section_text_riskfactors"] = None
    df["section_text_mdna"] = None
    return df[["symbol", "cik", "form_type", "available_at", "accession",
               "doc_uri", "section_text_riskfactors", "section_text_mdna"]]


# --------------------------------------------------------------------------
# Real universe loader (free data: yfinance prices + corp actions)
# --------------------------------------------------------------------------
# A curated, liquid, multi-sector universe mapped to matching sector ETFs (used
# as the abnormal-return benchmark). Tunable; the app exposes a size knob.
LIVE_UNIVERSE: dict[str, list[str]] = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS"],
    "XLE": ["XOM", "CVX", "COP", "SLB"],
    "XLV": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
}


def live_symbols(n: int | None = None) -> list[str]:
    """Flatten the universe to a ticker list, optionally capped at n."""
    syms = [s for tickers in LIVE_UNIVERSE.values() for s in tickers]
    return syms[:n] if n else syms


class LiveLoader:
    """Populate the PIT store with REAL free data (yfinance prices + corp actions).

    Prices are stored RAW with their splits/dividends, so the point-in-time
    as-of-T adjustment still holds on real data. Per-symbol fetch failures are
    skipped with a note (network is best-effort). Filing/insider/news tables are
    not populated here yet — those EDGAR document/XML parsers are a further step,
    so on real data only the price-based edge (and the paper engine) have inputs.
    """

    def __init__(self, store: PITStore, symbols: list[str] | None = None,
                 start: str = "2016-01-01", end: str | None = None, emit=None):
        self.store = store
        self.symbols = symbols or live_symbols()
        self.start = start
        self.end = end
        self.emit = emit or (lambda _m: None)
        self._sector = {sym: etf for etf, tickers in LIVE_UNIVERSE.items()
                        for sym in tickers}

    def sector_map(self) -> dict[str, str]:
        return dict(self._sector)

    def load_all(self) -> None:
        benchmarks = sorted({self._sector[s] for s in self.symbols if s in self._sector})
        to_fetch = list(self.symbols) + benchmarks
        all_prices, all_actions, loaded = [], [], []
        for i, sym in enumerate(to_fetch, 1):
            self.emit(f"[live] fetching {sym} ({i}/{len(to_fetch)}) ...")
            try:
                prices, actions = fetch_prices_yahoo(sym, self.start, self.end)
            except Exception as exc:
                self.emit(f"[live] skip {sym}: {type(exc).__name__}: {exc}")
                continue
            if prices.empty:
                self.emit(f"[live] skip {sym}: no data returned")
                continue
            all_prices.append(prices)
            if not actions.empty:
                all_actions.append(actions)
            loaded.append(sym)

        if not all_prices:
            raise RuntimeError("live load failed: no price data fetched (check network).")
        self.store.write_prices(pd.concat(all_prices, ignore_index=True))
        if all_actions:
            self.store.write_corp_actions(pd.concat(all_actions, ignore_index=True))

        stocks = [s for s in loaded if s in self._sector]
        first_session = pd.concat(all_prices)["date"].min()
        self.store.write_constituents(pd.DataFrame({
            "symbol": stocks, "start_date": first_session, "end_date": pd.NaT,
        }))
        self.emit(f"[live] loaded {len(stocks)} stocks + "
                  f"{len(loaded) - len(stocks) + len(benchmarks)} benchmarks.")
