"""S&P 500 screening universe + GICS sector map.

Tries to fetch the current constituents (and their sectors) from Wikipedia so it
stays current; falls back to a broad static set bundled with the app. This is a
LIVE screening universe only — not survivorship-free, and never used by the
validation harness / backtests (those keep their own point-in-time membership).
"""

from __future__ import annotations

import functools

# Broad static fallback, grouped by GICS sector (this doubles as the sector map).
SECTORS: dict[str, list[str]] = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "ACN", "CSCO",
        "INTC", "IBM", "QCOM", "TXN", "INTU", "NOW", "AMAT", "MU", "ADI", "LRCX",
        "KLAC", "SNPS", "CDNS", "PANW", "ANET", "ROP", "FTNT", "MSI", "ADSK", "NXPI",
        "MCHP", "APH", "TEL", "CTSH", "IT", "GLW", "HPQ", "HPE", "DELL", "WDC",
        "STX", "ON", "MPWR", "CDW", "KEYS", "NTAP", "ZBRA", "TDY", "TER", "GEN",
        "JBL", "AKAM", "FFIV", "EPAM", "PTC", "TYL", "ANSS", "SMCI",
    ],
    "Communication Services": [
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "EA", "TTWO", "OMC", "IPG", "LYV", "WBD", "FOXA", "FOX", "PARA", "NWSA", "NWS",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
        "SCHW", "BLK", "SPGI", "CB", "MMC", "PGR", "PNC", "AON", "ICE", "CME",
        "USB", "TFC", "BX", "KKR", "APO", "AIG", "MET", "PRU", "AFL", "TRV",
        "ALL", "MSCI", "MCO", "COF", "DFS", "FIS", "GPN", "PYPL", "AMP",
        "BK", "STT", "NTRS", "RJF", "HBAN", "RF", "CFG", "KEY", "FITB", "MTB",
        "WRB", "ACGL", "HIG", "CINF", "L", "BRO", "GL", "MKTX", "NDAQ", "CBOE",
        "WTW", "SYF", "FDS", "JKHY", "BEN", "IVZ", "TROW", "PFG",
    ],
    "Health Care": [
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
        "ISRG", "SYK", "BSX", "MDT", "GILD", "VRTX", "REGN", "CI", "ELV", "CVS",
        "ZTS", "BDX", "HCA", "MCK", "COR", "CNC", "HUM", "BMY", "DXCM", "EW",
        "IDXX", "IQV", "A", "GEHC", "RMD", "MTD", "WST", "BIIB", "WAT", "ZBH",
        "STE", "BAX", "HOLX", "COO", "PODD", "ALGN", "MOH", "DGX", "LH", "RVTY",
        "CAH", "TECH", "VTRS", "INCY", "CRL", "DVA", "UHS", "HSIC",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "BKNG", "LOW", "TJX", "SBUX", "NKE", "ORLY",
        "CMG", "MAR", "GM", "F", "HLT", "AZO", "ROST", "YUM", "DHI", "LEN",
        "NVR", "PHM", "GRMN", "APTV", "LULU", "EBAY", "TSCO", "ULTA", "DRI", "EXPE",
        "RCL", "CCL", "NCLH", "MGM", "LVS", "WYNN", "POOL", "DPZ", "BBY", "KMX",
        "GPC", "TPR", "RL", "DECK", "LKQ", "BWA", "MHK", "HAS", "WHR", "CZR", "ABNB",
    ],
    "Consumer Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT",
        "KMB", "GIS", "SYY", "KHC", "STZ", "MNST", "KDP", "ADM", "KR", "HSY",
        "KVUE", "DG", "DLTR", "MKC", "CHD", "CLX", "K", "HRL", "SJM", "CAG",
        "TSN", "CPB", "TAP", "BG", "LW", "BF-B", "EL", "WBA",
    ],
    "Industrials": [
        "GE", "CAT", "RTX", "UNP", "HON", "BA", "DE", "LMT", "UPS", "ETN",
        "ADP", "GD", "NOC", "EMR", "ITW", "CSX", "FDX", "NSC", "WM", "PH",
        "TT", "CARR", "PCAR", "JCI", "CMI", "GWW", "PAYX", "FAST", "URI", "OTIS",
        "AME", "RSG", "CTAS", "ROK", "DOV", "IR", "EFX", "VRSK", "XYL", "FTV",
        "HWM", "WAB", "PWR", "AXON", "ODFL", "LHX", "BR", "DAL", "UAL", "LUV",
        "TXT", "SNA", "SWK", "PNR", "ALLE", "MAS", "JBHT", "CHRW", "EXPD", "NDSN",
        "ROL", "GNRC", "HUBB", "IEX", "PAYC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "WMB", "OKE", "VLO",
        "KMI", "OXY", "HES", "FANG", "BKR", "HAL", "DVN", "TRGP", "CTRA", "EQT",
        "APA", "OVV",
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC",
        "MLM", "PPG", "CTVA", "IFF", "ALB", "STLD", "LYB", "PKG", "IP", "AMCR",
        "BALL", "AVY", "CF", "MOS", "FMC", "EMN", "CE",
    ],
    "Utilities": [
        "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC", "XEL", "PEG",
        "ED", "PCG", "WEC", "EIX", "AWK", "DTE", "ETR", "AEE", "PPL", "FE",
        "ES", "CMS", "ATO", "CNP", "NI", "LNT", "EVRG", "AES", "PNW",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "CCI", "DLR", "CBRE",
        "VICI", "EXR", "AVB", "EQR", "VTR", "IRM", "WY", "INVH", "ARE", "SBAC",
        "MAA", "ESS", "UDR", "KIM", "REG", "HST", "BXP", "FRT", "CPT",
    ],
}

# Sector ETF proxies (for sector relative-strength when a name->sector map is thin).
SECTOR_ETFS: dict[str, str] = {
    "Information Technology": "XLK", "Communication Services": "XLC",
    "Financials": "XLF", "Health Care": "XLV", "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP", "Industrials": "XLI", "Energy": "XLE",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
}

_STATIC: list[str] = [s for syms in SECTORS.values() for s in syms]
_STATIC_SECTOR_OF: dict[str, str] = {s: sec for sec, syms in SECTORS.items() for s in syms}


@functools.lru_cache(maxsize=1)
def _live() -> tuple[tuple[str, ...], dict]:
    """(symbols, sector_of) from Wikipedia, or ((), {}) on any failure."""
    try:
        from harness.data.wiki import constituents_table
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df, col = constituents_table(url, min_rows=400)
        if df is None:
            return (), {}
        syms, sect = [], {}
        for _, r in df.iterrows():
            s = str(r.get(col, "")).strip().upper().replace(".", "-")
            if not s or s == "NAN":
                continue
            syms.append(s)
            sect[s] = str(r.get("GICS Sector", "")).strip() or _STATIC_SECTOR_OF.get(s, "")
        if len(syms) >= 400:
            return tuple(dict.fromkeys(syms)), sect
    except Exception:
        pass
    return (), {}


@functools.lru_cache(maxsize=1)
def sp500_symbols() -> list[str]:
    """Current S&P 500 tickers (live if possible, else the static fallback)."""
    live, _ = _live()
    return sorted(live) if live else sorted(dict.fromkeys(_STATIC))


def sector_of() -> dict[str, str]:
    """Ticker -> GICS sector (live if possible, else static)."""
    _, sect = _live()
    return dict(sect) if sect else dict(_STATIC_SECTOR_OF)


def screen_universe(limit: int | None = None) -> list[str]:
    """The screening universe, optionally capped to the first `limit` (sorted)."""
    syms = sp500_symbols()
    return syms[:limit] if limit else syms
