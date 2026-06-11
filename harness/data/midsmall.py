"""S&P MidCap 400 + SmallCap 600 screening universes + sector maps.

This is the hidden-gem hunting ground: names NOT yet in the S&P 500, where
discovery is still possible. Tries to fetch current constituents (and sectors)
from Wikipedia; falls back to a static set of well-known liquid mid/small caps.
The fallback is approximate by design (membership drifts) — this is a LIVE
screening universe only, never used by the validation harness/backtests, and
every name still has to clear the screen's own liquidity and data-quality bars.
"""

from __future__ import annotations

import functools

# Static fallbacks, grouped by GICS sector (doubles as the sector map).
MID_SECTORS: dict[str, list[str]] = {
    "Information Technology": [
        "ONTO", "LSCC", "QLYS", "NOVT", "POWI", "SYNA", "CGNX", "ITRI", "PEGA",
        "BMI", "CIEN", "COHR", "ALTR", "MKSI", "OLED",
    ],
    "Industrials": [
        "AAON", "ATKR", "AYI", "FIX", "LECO", "MLI", "RBC", "WTS", "XPO",
        "FLR", "MIDD", "GTLS", "KBR", "MOG-A", "WWD", "CW", "TTC",
    ],
    "Health Care": [
        "CHE", "ENSG", "HALO", "ITGR", "LNTH", "MEDP", "NEOG", "PEN",
        "GKOS", "EXEL", "MMSI", "TMDX",
    ],
    "Financials": [
        "AX", "CADE", "EWBC", "FNB", "GBCI", "OMF", "PB", "SNV", "UMBF",
        "WAL", "ESNT", "PRI", "RLI", "AFG",
    ],
    "Consumer Discretionary": [
        "BOOT", "CROX", "FND", "LAD", "WING", "YETI", "OLLI", "PLNT", "THO",
        "GME", "TPX", "COLM", "MUSA",
    ],
    "Consumer Staples": ["ELF", "LANC", "FLO", "CASY", "POST", "DAR"],
    "Energy": ["AR", "CHRD", "MTDR", "NOG", "PR", "RRC", "CIVI"],
    "Materials": ["CRS", "ESI", "WOR", "EXP", "RGLD", "CLF", "RYAM"],
    "Utilities": ["OGE", "POR", "IDA", "NWE", "ALE"],
    "Real Estate": ["FR", "STAG", "TRNO", "RYN", "EGP", "COLD"],
    "Communication Services": ["NXST", "TGNA", "IAC", "CARG"],
}

SMALL_SECTORS: dict[str, list[str]] = {
    "Information Technology": [
        "ACLS", "PLXS", "DIOD", "FORM", "VECO", "SANM", "AGYS", "BHE",
        "PRGS", "CTS", "RAMP",
    ],
    "Industrials": [
        "AROC", "ANDE", "DY", "GVA", "MYRG", "HUBG", "MWA", "ARCB",
        "AIT", "GMS", "REZI", "TILE",
    ],
    "Health Care": [
        "CORT", "CPRX", "AMPH", "ICUI", "LMAT", "OMCL", "NHC", "USPH", "IRMD",
    ],
    "Financials": [
        "ENVA", "BANF", "HOMB", "ABCB", "CASH", "STBA", "WSFS", "TBBK",
        "PIPR", "VRTS",
    ],
    "Consumer Discretionary": [
        "SHOO", "PLAY", "URBN", "AEO", "CRI", "SCVL",
        "WGO", "PATK", "MTH", "IBP",
    ],
    "Consumer Staples": ["CALM", "JJSF", "WDFC", "UVV", "SPTN"],
    "Energy": ["SM", "CRC", "TALO", "HP", "LBRT"],
    "Materials": ["BCPC", "HWKN", "KALU", "MTX", "SXT"],
    "Utilities": ["AWR", "CWT", "MSEX", "OTTR"],
    "Real Estate": ["AAT", "LXP", "GTY", "UE", "ELME"],
    "Communication Services": ["SATS", "GOGO", "IDT"],
}


def _static(sectors: dict[str, list[str]]) -> tuple[list[str], dict[str, str]]:
    syms = [s for lst in sectors.values() for s in lst]
    sect = {s: sec for sec, lst in sectors.items() for s in lst}
    return syms, sect


@functools.lru_cache(maxsize=4)
def _live(index: str) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """(symbols, sector pairs) from Wikipedia, or ((), ()) on any failure."""
    urls = {
        "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    }
    try:
        from harness.data.wiki import constituents_table
        df, col = constituents_table(urls[index], min_rows=250)
        if df is None:
            return (), ()
        syms, sect = [], {}
        for _, r in df.iterrows():
            s = str(r.get(col, "")).strip().upper().replace(".", "-")
            if not s or s == "NAN":
                continue
            syms.append(s)
            sec = str(r.get("GICS Sector", "")).strip()
            if sec and sec != "nan":
                sect[s] = sec
        if len(syms) >= 250:
            return tuple(dict.fromkeys(syms)), tuple(sect.items())
    except Exception:
        pass
    return (), ()


def _index(index: str, sectors_static: dict) -> tuple[list[str], dict[str, str]]:
    live_syms, live_sect = _live(index)
    if live_syms:
        return sorted(live_syms), dict(live_sect)
    syms, sect = _static(sectors_static)
    return sorted(dict.fromkeys(syms)), sect


def sp400_universe() -> tuple[list[str], dict[str, str]]:
    """(symbols, sector_of) for the S&P MidCap 400 (live if possible)."""
    return _index("sp400", MID_SECTORS)


def sp600_universe() -> tuple[list[str], dict[str, str]]:
    """(symbols, sector_of) for the S&P SmallCap 600 (live if possible)."""
    return _index("sp600", SMALL_SECTORS)


def midsmall_universe() -> tuple[list[str], dict[str, str]]:
    """Combined S&P 400 + 600: the off-the-beaten-path screening universe."""
    s4, m4 = sp400_universe()
    s6, m6 = sp600_universe()
    return sorted(dict.fromkeys(s4 + s6)), {**m6, **m4}


def screen_universe(index: str = "midsmall", limit: int | None = None
                    ) -> tuple[list[str], dict[str, str]]:
    fn = {"sp400": sp400_universe, "sp600": sp600_universe,
          "midsmall": midsmall_universe}[index]
    syms, sect = fn()
    return (syms[:limit] if limit else syms), sect
