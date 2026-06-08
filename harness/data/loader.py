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

import re
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
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "XLV": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ"],
    "XLI": ["CAT", "BA", "HON", "UPS", "GE", "RTX"],
    "XLC": ["GOOGL", "META", "NFLX", "DIS", "TMUS"],
}


def live_symbols(n: int | None = None) -> list[str]:
    """Flatten the universe to a ticker list, optionally capped at n.

    Round-robin across sectors so a small cap still spans sectors (diversity
    matters for confluence and the sector-exposure cap).
    """
    cols = [list(t) for t in LIVE_UNIVERSE.values()]
    out: list[str] = []
    for i in range(max((len(c) for c in cols), default=0)):
        for c in cols:
            if i < len(c):
                out.append(c[i])
    return out[:n] if n else out


# SEC quarterly bulk insider-transactions datasets (Form 3/4/5) — the source for
# validating the insider edge on YEARS of history without per-filing fetches.
INSIDER_DATASET_BASE = ("https://www.sec.gov/files/structureddata/data/"
                        "insider-transactions-data-sets")

_SENIOR_KEYS = {"CEO": "CEO", "CHIEF EXECUTIVE": "CEO", "CFO": "CFO",
                "CHIEF FINANCIAL": "CFO", "COO": "COO", "CHIEF OPERATING": "COO",
                "PRESIDENT": "President", "CHAIR": "Chairman"}


def _canon_role(title: str | None, relationship: str | None) -> str:
    """Map a Form 4 owner title/relationship to edge 6's role vocabulary."""
    t = title.upper() if isinstance(title, str) else ""
    for key, canon in _SENIOR_KEYS.items():
        if key in t:
            return canon
    rel = relationship.lower() if isinstance(relationship, str) else ""
    if "director" in rel:
        return "Director"
    if "officer" in rel:
        return "Officer"
    return "Insider"


def recent_quarters(n: int, asof: pd.Timestamp | None = None) -> list[tuple[int, int]]:
    """The n most recent COMPLETED calendar quarters as (year, quarter)."""
    asof = (asof or pd.Timestamp.now()).normalize()
    q = (asof.month - 1) // 3 + 1
    y = asof.year
    out = []
    # step back one quarter first (current quarter's dataset isn't complete).
    for _ in range(n):
        q -= 1
        if q == 0:
            q, y = 4, y - 1
        out.append((y, q))
    return out


def fetch_insider_quarter(year: int, quarter: int, user_agent: str,
                          tickers: set[str]) -> pd.DataFrame:
    """Parse one quarterly bulk dataset into edge-6 purchase rows for `tickers`.

    Aggregates each Form 4's non-derivative PURCHASES (code 'P') to one row:
    symbol, available_at (filing-date close), insider_role, shares, value.
    """
    import io
    import zipfile
    import requests  # lazy

    url = f"{INSIDER_DATASET_BASE}/{year}q{quarter}_form345.zip"
    z = zipfile.ZipFile(io.BytesIO(
        requests.get(url, headers={"User-Agent": user_agent}, timeout=120).content))

    def tsv(name, cols):
        return pd.read_csv(z.open(name), sep="\t", dtype=str, encoding="latin-1",
                           usecols=cols)

    sub = tsv("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["ISSUERTRADINGSYMBOL"].isin(tickers)]
    if sub.empty:
        return pd.DataFrame(columns=["symbol", "cik", "available_at", "insider_role",
                                     "txn_code", "shares", "value"])
    nd = tsv("NONDERIV_TRANS.tsv",
             ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    nd = nd[(nd["TRANS_CODE"] == "P") & nd["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])]
    if nd.empty:
        return pd.DataFrame(columns=["symbol", "cik", "available_at", "insider_role",
                                     "txn_code", "shares", "value"])
    ro = tsv("REPORTINGOWNER.tsv",
             ["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    ro = ro.groupby("ACCESSION_NUMBER", as_index=False).first()

    nd["sh"] = pd.to_numeric(nd["TRANS_SHARES"], errors="coerce")
    nd["px"] = pd.to_numeric(nd["TRANS_PRICEPERSHARE"], errors="coerce")
    nd["val"] = nd["sh"] * nd["px"]
    agg = nd.groupby("ACCESSION_NUMBER", as_index=False).agg(shares=("sh", "sum"),
                                                             value=("val", "sum"))
    m = sub.merge(agg, on="ACCESSION_NUMBER").merge(ro, on="ACCESSION_NUMBER", how="left")

    rows = []
    for r in m.itertuples(index=False):
        fdate = pd.to_datetime(r.FILING_DATE, format="%d-%b-%Y", errors="coerce")
        if pd.isna(fdate):
            fdate = pd.to_datetime(r.FILING_DATE, errors="coerce")
        if pd.isna(fdate) or not r.shares or r.shares <= 0:
            continue
        rows.append({
            "symbol": r.ISSUERTRADINGSYMBOL, "cik": "",
            "available_at": available_at_for_session(fdate),
            "insider_role": _canon_role(getattr(r, "RPTOWNER_TITLE", None),
                                        getattr(r, "RPTOWNER_RELATIONSHIP", None)),
            "txn_code": "P", "shares": int(r.shares), "value": float(r.value or 0),
        })
    return pd.DataFrame(rows)


_CIK_CACHE: dict[str, str] | None = None


def fetch_cik_map(user_agent: str) -> dict[str, str]:
    """Ticker -> zero-padded CIK from SEC's company_tickers.json (cached)."""
    global _CIK_CACHE
    if _CIK_CACHE is None:
        import requests  # lazy
        data = requests.get("https://www.sec.gov/files/company_tickers.json",
                            headers={"User-Agent": user_agent}, timeout=30).json()
        _CIK_CACHE = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    return _CIK_CACHE


def _parse_form4(xml_text: str) -> dict | None:
    """Extract role / net purchase from a raw Form 4 XML. Returns None if not
    a (net) purchase by an insider (edge 6 cares about cluster BUYING)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return None
    rel = root.find("reportingOwner/reportingOwnerRelationship")
    role = "Insider"
    if rel is not None:
        ot = rel.findtext("officerTitle")
        if ot:
            role = ot
        elif (rel.findtext("isDirector") in ("1", "true")):
            role = "Director"
        elif (rel.findtext("isTenPercentOwner") in ("1", "true")):
            role = "10%Owner"
    bought_sh = bought_val = 0.0
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = txn.findtext(".//transactionCoding/transactionCode")
        if code != "P":                       # P = open-market purchase
            continue
        sh = float(txn.findtext(".//transactionShares/value") or 0)
        px = float(txn.findtext(".//transactionPricePerShare/value") or 0)
        bought_sh += sh
        bought_val += sh * px
    if bought_sh <= 0:
        return None
    return {"insider_role": role, "txn_code": "P", "shares": int(bought_sh),
            "value": float(bought_val)}


_XBRL_WORDS = {"xbrli", "iso4217", "us-gaap", "dei", "utr", "srt", "false", "true",
               "membergroup", "domain", "axis", "member"}


def _looks_like_noise(tok: str) -> bool:
    """True for inline-XBRL / metadata tokens that aren't readable prose."""
    if ":" in tok:                                  # ns:name (us-gaap:Revenue, iso4217:USD)
        return True
    low = tok.lower().strip(".,;()[]")
    if low in _XBRL_WORDS:
        return True
    alpha = sum(c.isalpha() for c in tok)
    digit = sum(c.isdigit() for c in tok)
    if digit and alpha == 0:                          # 0001734722, 2026-04-30, 189 (no letters)
        return True
    if len(tok) > 24 and alpha < len(tok) * 0.6:     # long identifier blobs
        return True
    return False


def _clean_filing_text(html: str, cap_chars: int = 400_000) -> str:
    """Strip a filing to readable narrative prose (capped).

    SEC filings are inline-XBRL: tag-stripping alone leaves a header soup of
    namespace tokens, contexts, dates and identifiers. We drop those so edge 1
    and the Filings evidence see actual sentences, and start the text at the
    first real narrative anchor when we can find one.
    """
    t = re.sub(r"(?is)<[^>]+>", " ", html)
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = " ".join(tok for tok in t.split() if not _looks_like_noise(tok))
    t = re.sub(r"\s+", " ", t).strip()
    # Prefer to begin at the substantive body (skip cover-page boilerplate).
    for anchor in (r"Risk Factors", r"Management.s Discussion",
                   r"PART I", r"Item\s*1A", r"Item\s*2"):
        m = re.search(anchor, t, re.I)
        if m and m.start() < len(t) * 0.5:
            t = t[m.start():]
            break
    return t[:cap_chars]


def _fetch_filing_text(cik: str, acc: str, doc: str, user_agent: str) -> str:
    import requests  # lazy
    raw = doc.split("/")[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{raw}"
    return _clean_filing_text(requests.get(url, headers={"User-Agent": user_agent},
                                           timeout=60).text)


def fetch_edgar_for_symbol(symbol: str, cik: str, user_agent: str,
                           since_days: int = 60, max_form4: int = 80,
                           periodic_text: int = 2):
    """Recent EDGAR data for a symbol:
      * 8-K (edge 2) in the recent window (metadata only),
      * insider PURCHASES from Form 4 (edge 6) in the window (raw-XML parsed),
      * the most recent `periodic_text` 10-K/10-Q filings WITH cleaned full text
        (edge 1 needs the latest + the prior comparable filing to diff, even if
        the prior one predates the window).
    """
    import requests  # lazy
    h = {"User-Agent": user_agent}
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=since_days)
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                       headers=h, timeout=30).json()
    rec = sub["filings"]["recent"]
    forms, accs = rec["form"], rec["accessionNumber"]
    docs, dates = rec["primaryDocument"], rec["filingDate"]

    filings, form4 = [], []
    f4_count, periodic_count = 0, 0
    for form, acc, doc, d in zip(forms, accs, docs, dates):
        fdate = pd.Timestamp(d)
        avail = available_at_for_session(fdate)
        if form in ("10-K", "10-Q") and periodic_count < periodic_text:
            periodic_count += 1
            try:
                text = _fetch_filing_text(cik, acc, doc, user_agent)
            except Exception:
                text = None
            filings.append({
                "symbol": symbol, "cik": cik, "form_type": form, "available_at": avail,
                "accession": acc, "doc_uri": doc,
                "section_text_riskfactors": text, "section_text_mdna": None,
            })
        elif form == "8-K" and fdate >= cutoff:
            filings.append({
                "symbol": symbol, "cik": cik, "form_type": form, "available_at": avail,
                "accession": acc, "doc_uri": doc,
                "section_text_riskfactors": None, "section_text_mdna": None,
            })
        elif form == "4" and fdate >= cutoff and f4_count < max_form4:
            f4_count += 1
            raw = doc.split("/")[-1]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{raw}"
            try:
                parsed = _parse_form4(requests.get(url, headers=h, timeout=30).text)
            except Exception:
                parsed = None
            if parsed:
                parsed.update({"symbol": symbol, "cik": cik, "available_at": avail})
                form4.append(parsed)
    return filings, form4


class LiveLoader:
    """Populate the PIT store with REAL free data (yfinance prices + corp actions),
    and — when an EDGAR user-agent is supplied — recent 8-K filings (edge 2) and
    insider purchases from Form 4 (edge 6).

    Prices are stored RAW with their splits/dividends, so the point-in-time
    as-of-T adjustment still holds on real data. EDGAR is fetched for a recent
    window (8-K, Form 4) plus the two latest 10-K/10-Q filings WITH cleaned full
    text (edge 1 = whole-filing change). The economic-link graph (edge 7) remains
    future work (no free structured supplier/customer source), so it has no live
    inputs yet.
    """

    def __init__(self, store: PITStore, symbols: list[str] | None = None,
                 start: str = "2016-01-01", end: str | None = None, emit=None,
                 edgar_user_agent: str | None = None, edgar_since_days: int = 60):
        self.store = store
        self.symbols = symbols or live_symbols()
        self.start = start
        self.end = end
        self.emit = emit or (lambda _m: None)
        self.edgar_user_agent = edgar_user_agent
        self.edgar_since_days = edgar_since_days
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
        bench_loaded = [s for s in loaded if s not in self._sector]
        self.emit(f"[live] loaded {len(stocks)} stocks + {len(bench_loaded)} benchmarks.")

        if self.edgar_user_agent:
            self._load_edgar(stocks)

    def _load_edgar(self, stocks: list[str]) -> None:
        """Fetch recent 8-K (edge 2) + insider purchases (edge 6) from EDGAR."""
        try:
            cik_map = fetch_cik_map(self.edgar_user_agent)
        except Exception as exc:
            self.emit(f"[edgar] skipped (CIK map fetch failed: {exc})")
            return
        all_filings, all_form4, n_buys = [], [], 0
        for i, sym in enumerate(stocks, 1):
            cik = cik_map.get(sym)
            if not cik:
                continue
            self.emit(f"[edgar] {sym} ({i}/{len(stocks)}) ...")
            try:
                filings, form4 = fetch_edgar_for_symbol(
                    sym, cik, self.edgar_user_agent, since_days=self.edgar_since_days)
            except Exception as exc:
                self.emit(f"[edgar] skip {sym}: {type(exc).__name__}: {exc}")
                continue
            all_filings.extend(filings)
            all_form4.extend(form4)
            n_buys += len(form4)
        if all_filings:
            self.store.write_filings(pd.DataFrame(all_filings))
        if all_form4:
            self.store.write_form4(pd.DataFrame(all_form4))
        self.emit(f"[edgar] loaded {len(all_filings)} filings (8-K/10-K/10-Q) and "
                  f"{n_buys} insider-purchase records (last {self.edgar_since_days}d).")
