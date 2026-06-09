"""S&P 500 screening universe.

Tries to fetch the current constituent list from Wikipedia (so it stays current);
falls back to a broad static list bundled with the app. This is a LIVE screening
universe only — it is not survivorship-free and must never be used for the
validation harness / backtests (those keep their own point-in-time membership).
"""

from __future__ import annotations

import functools

# A broad, diversified static fallback (the bulk of S&P 500 index weight across
# all sectors). Used when the live fetch is unavailable (e.g. offline / bundled).
_STATIC: list[str] = [
    # Mega-cap tech / comms
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "AVGO", "TSLA", "ORCL",
    "ADBE", "CRM", "AMD", "ACN", "CSCO", "INTC", "IBM", "QCOM", "TXN", "INTU",
    "NOW", "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "ANET",
    "ROP", "FTNT", "MSI", "ADSK", "NXPI", "MCHP", "APH", "TEL", "CTSH", "IT",
    "GLW", "HPQ", "HPE", "DELL", "WDC", "STX", "ON", "MPWR", "CDW", "KEYS",
    "NTAP", "ZBRA", "TDY", "TER", "GEN", "JBL", "AKAM", "FFIV", "EPAM", "PTC",
    "TYL", "ANSS", "SMCI", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "OMC", "IPG", "LYV", "WBD", "FOXA", "FOX", "PARA", "NWSA", "NWS",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C",
    "SCHW", "BLK", "SPGI", "CB", "MMC", "PGR", "PNC", "AON", "ICE", "CME",
    "USB", "TFC", "BX", "KKR", "APO", "AIG", "MET", "PRU", "AFL", "TRV",
    "ALL", "MSCI", "MCO", "COF", "DFS", "FIS", "FISV", "GPN", "PYPL", "AMP",
    "BK", "STT", "NTRS", "RJF", "HBAN", "RF", "CFG", "KEY", "FITB", "MTB",
    "WRB", "ACGL", "HIG", "CINF", "L", "BRO", "GL", "MKTX", "NDAQ", "CBOE",
    "WTW", "SYF", "FDS", "JKHY", "BEN", "IVZ", "TROW", "EG", "PFG",
    # Health care
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "SYK", "BSX", "MDT", "GILD", "VRTX", "REGN", "CI", "ELV", "CVS",
    "ZTS", "BDX", "HCA", "MCK", "COR", "CNC", "HUM", "BMY", "DXCM", "EW",
    "IDXX", "IQV", "A", "GEHC", "RMD", "MTD", "WST", "BIIB", "WAT", "ZBH",
    "STE", "BAX", "HOLX", "COO", "PODD", "ALGN", "MOH", "DGX", "LH", "RVTY",
    "CAH", "TECH", "VTRS", "INCY", "CRL", "SOLV", "DVA", "UHS", "HSIC",
    # Consumer discretionary
    "HD", "MCD", "BKNG", "LOW", "TJX", "SBUX", "NKE", "ORLY", "CMG", "MAR",
    "GM", "F", "HLT", "AZO", "ROST", "YUM", "DHI", "LEN", "NVR", "PHM",
    "GRMN", "APTV", "LULU", "EBAY", "TSCO", "ULTA", "DRI", "EXPE", "RCL", "CCL",
    "NCLH", "MGM", "LVS", "WYNN", "POOL", "DPZ", "BBY", "KMX", "GPC", "TPR",
    "RL", "DECK", "LKQ", "BWA", "MHK", "HAS", "WHR", "CZR", "ABNB", "TPX",
    # Consumer staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT",
    "KMB", "GIS", "SYY", "KHC", "STZ", "MNST", "KDP", "ADM", "KR", "HSY",
    "KVUE", "DG", "DLTR", "MKC", "CHD", "CLX", "K", "HRL", "SJM", "CAG",
    "TSN", "CPB", "TAP", "BG", "LW", "BF-B", "EL", "WBA",
    # Industrials
    "GE", "CAT", "RTX", "UNP", "HON", "BA", "DE", "LMT", "UPS", "ETN",
    "ADP", "GD", "NOC", "EMR", "ITW", "CSX", "FDX", "NSC", "WM", "PH",
    "TT", "CARR", "PCAR", "JCI", "CMI", "GWW", "PAYX", "FAST", "URI", "OTIS",
    "AME", "RSG", "CTAS", "ROK", "DOV", "IR", "EFX", "VRSK", "XYL", "FTV",
    "HWM", "WAB", "PWR", "AXON", "ODFL", "LHX", "BR", "DAL", "UAL", "LUV",
    "TXT", "SNA", "SWK", "PNR", "ALLE", "MAS", "JBHT", "CHRW", "EXPD", "NDSN",
    "ROL", "DAY", "GNRC", "HUBB", "IEX", "PAYC", "VLTO",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "WMB", "OKE", "VLO",
    "KMI", "OXY", "HES", "FANG", "BKR", "HAL", "DVN", "TRGP", "CTRA", "EQT",
    "MRO", "APA", "OVV",
    # Materials
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC",
    "MLM", "PPG", "CTVA", "IFF", "ALB", "STLD", "LYB", "PKG", "IP", "AMCR",
    "BALL", "AVY", "CF", "MOS", "FMC", "EMN", "CE",
    # Utilities
    "NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC", "XEL", "PEG",
    "ED", "PCG", "WEC", "EIX", "AWK", "DTE", "ETR", "AEE", "PPL", "FE",
    "ES", "CMS", "ATO", "CNP", "NI", "LNT", "EVRG", "AES", "PNW",
    # Real estate
    "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "CCI", "DLR", "CBRE",
    "VICI", "EXR", "AVB", "EQR", "VTR", "IRM", "WY", "INVH", "ARE", "SBAC",
    "MAA", "ESS", "UDR", "KIM", "REG", "HST", "BXP", "FRT", "CPT",
]


@functools.lru_cache(maxsize=1)
def sp500_symbols() -> list[str]:
    """Current S&P 500 tickers (live if possible, else the static fallback)."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        col = tables[0]["Symbol"]
        syms = [str(s).strip().upper().replace(".", "-") for s in col if str(s).strip()]
        if len(syms) >= 400:
            return sorted(dict.fromkeys(syms))
    except Exception:
        pass
    return sorted(dict.fromkeys(_STATIC))


def screen_universe(limit: int | None = None) -> list[str]:
    """The screening universe, optionally capped to the first `limit` (sorted)."""
    syms = sp500_symbols()
    return syms[:limit] if limit else syms
