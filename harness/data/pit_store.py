"""Point-in-time data store (parquet) with as-of queries.

This is the integrity foundation of the harness (harness spec §4, master design
§9 and invariant 6). The cardinal rule:

    At decision time T, nothing timestamped after T is ever visible, and no
    corporate action with ex_date > T ever touches the prices returned.

All point-in-time filtering lives in one place — the :class:`AsOfView` — so the
discipline is auditable and the leakage CI test has a single surface to attack.

Time model
----------
* Event tables (``filings``, ``form4``, ``news``, ``links``) carry an
  ``available_at`` timestamp that MUST be timezone-aware UTC. The loader is
  responsible for setting it with a conservative latency buffer (so e.g.
  after-close information already carries the next session's availability). The
  store enforces the exact instant rule ``available_at <= T``.
* Prices, corporate actions, and constituent membership are keyed by a calendar
  *trading session* date (tz-naive). They are compared against ``T``'s
  US/Eastern calendar date — the exchange-local session date. Because the
  system only ever decides post-close, the bar/action/membership for session D
  is visible at a decision stamped on session D, and anything dated after D is
  withheld. (Mapping after-close info to the next session is the loader's job,
  baked into ``available_at``; the store stays simple and conservative.)

Raw prices are immutable ground truth; corporate-action adjustment is computed
at query time (see :mod:`harness.data.corp_actions`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from harness.data import corp_actions as ca

EASTERN = "America/New_York"


@dataclass(frozen=True)
class _TableSpec:
    required: tuple[str, ...]
    instant_cols: tuple[str, ...] = ()   # must be tz-aware UTC
    date_cols: tuple[str, ...] = ()      # tz-naive session dates
    dedupe: tuple[str, ...] = ()


# Schema per harness spec §4. `links` additionally carries `available_at` (the
# time its source filing became available) so every event row is PIT-filterable
# uniformly (master §9: every record carries available_at).
TABLES: dict[str, _TableSpec] = {
    "prices": _TableSpec(
        required=("symbol", "date", "open", "high", "low", "close", "volume"),
        date_cols=("date",),
        dedupe=("symbol", "date"),
    ),
    "corp_actions": _TableSpec(
        required=("symbol", "ex_date", "type", "ratio_or_amount"),
        date_cols=("ex_date",),
        dedupe=("symbol", "ex_date", "type"),
    ),
    "constituents": _TableSpec(
        required=("symbol", "start_date", "end_date"),
        date_cols=("start_date", "end_date"),
        dedupe=("symbol", "start_date"),
    ),
    "filings": _TableSpec(
        required=(
            "symbol", "cik", "form_type", "available_at", "accession",
            "doc_uri", "section_text_riskfactors", "section_text_mdna",
        ),
        instant_cols=("available_at",),
        dedupe=("symbol", "accession"),
    ),
    "form4": _TableSpec(
        required=(
            "symbol", "cik", "available_at", "insider_role",
            "txn_code", "shares", "value",
        ),
        instant_cols=("available_at",),
        dedupe=("symbol", "available_at", "insider_role", "txn_code", "shares"),
    ),
    "links": _TableSpec(
        required=(
            "focal_symbol", "linked_symbol", "link_type",
            "source_filing", "available_at",
        ),
        instant_cols=("available_at",),
        dedupe=("focal_symbol", "linked_symbol", "link_type", "source_filing"),
    ),
    "news": _TableSpec(
        required=("symbol", "available_at", "headline", "body_uri", "source"),
        instant_cols=("available_at",),
        dedupe=("symbol", "available_at", "headline"),
    ),
    # A contemporaneous fundamentals snapshot (valuation multiples + growth +
    # forward/guidance estimates). available_at gates it like any event row, so a
    # snapshot fetched today is invisible to a historical backtest (no leak); a
    # live decision about "today" sees it. Extra numeric columns are optional.
    "fundamentals": _TableSpec(
        required=("symbol", "available_at"),
        instant_cols=("available_at",),
        dedupe=("symbol", "available_at"),
    ),
}

# Event tables (governed by `available_at <= T`) and their PIT time column.
EVENT_TABLES = {
    name: spec.instant_cols[0]
    for name, spec in TABLES.items()
    if spec.instant_cols
}


def _as_utc(series: pd.Series, col: str) -> pd.Series:
    """Coerce a column to tz-aware UTC; reject tz-naive values (leak risk)."""
    s = pd.to_datetime(series)
    tz = getattr(s.dtype, "tz", None)
    if tz is None:
        raise ValueError(
            f"column {col!r} must be timezone-aware (UTC); got tz-naive values. "
            "available_at timestamps must carry a timezone so the as-of filter "
            "is unambiguous."
        )
    return s.dt.tz_convert("UTC")


def _as_date(series: pd.Series) -> pd.Series:
    """Coerce a column to tz-naive, midnight-normalised session dates."""
    s = pd.to_datetime(series)
    tz = getattr(s.dtype, "tz", None)
    if tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def _eastern_date(T: pd.Timestamp) -> pd.Timestamp:
    """The US/Eastern calendar (trading-session) date of decision instant T."""
    return pd.Timestamp(T.tz_convert(EASTERN).date())


class PITStore:
    """A local parquet store with point-in-time as-of queries.

    One parquet file per table lives under ``root``. Writes append-merge
    (read + concat + dedupe + rewrite), which is fine at this universe size.
    Reads only ever go through :meth:`as_of`.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}

    # -- paths / raw IO ----------------------------------------------------
    def _path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def read_table(self, table: str) -> pd.DataFrame:
        """Full raw table (no PIT filter). For writes/introspection/tests.

        Cached in memory (cleared on write) so the many as-of queries an event
        study issues do not re-read parquet from disk each time. Callers receive
        a defensive copy, so the cache is never mutated through a view.
        """
        if table in self._cache:
            return self._cache[table].copy()
        spec = TABLES[table]
        path = self._path(table)
        if not path.exists():
            return pd.DataFrame(columns=list(spec.required))
        df = pd.read_parquet(path)
        self._cache[table] = df
        return df.copy()

    def _raw(self, table: str) -> pd.DataFrame:
        """Internal, non-copying cached read for the as-of accessors.

        Data is already dtype-normalised on write, so accessors need not
        re-coerce; they only ever boolean-index (which returns fresh frames),
        never mutating the cached object in place. Hot-path only.
        """
        if table not in self._cache:
            self.read_table(table)              # populates cache (or returns empty)
        return self._cache.get(table, pd.DataFrame(columns=list(TABLES[table].required)))

    # -- writes ------------------------------------------------------------
    def _write(self, table: str, df: pd.DataFrame) -> None:
        spec = TABLES[table]
        missing = [c for c in spec.required if c not in df.columns]
        if missing:
            raise ValueError(f"{table}: missing required columns {missing}")
        df = df.copy()
        for col in spec.instant_cols:
            df[col] = _as_utc(df[col], col)
        for col in spec.date_cols:
            df[col] = _as_date(df[col])
        existing = self.read_table(table)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
        if spec.dedupe:
            df = df.drop_duplicates(subset=list(spec.dedupe), keep="last")
        df = df.reset_index(drop=True)
        df.to_parquet(self._path(table), index=False)
        self._cache.pop(table, None)

    def write_prices(self, df: pd.DataFrame) -> None:
        self._write("prices", df)

    def write_corp_actions(self, df: pd.DataFrame) -> None:
        self._write("corp_actions", df)

    def write_constituents(self, df: pd.DataFrame) -> None:
        self._write("constituents", df)

    def write_filings(self, df: pd.DataFrame) -> None:
        self._write("filings", df)

    def write_form4(self, df: pd.DataFrame) -> None:
        self._write("form4", df)

    def write_links(self, df: pd.DataFrame) -> None:
        self._write("links", df)

    def write_news(self, df: pd.DataFrame) -> None:
        self._write("news", df)

    def write_fundamentals(self, df: pd.DataFrame) -> None:
        self._write("fundamentals", df)

    # -- the single PIT entry point ---------------------------------------
    def as_of(self, T) -> "AsOfView":
        ts = pd.Timestamp(T)
        if ts.tz is None:
            raise ValueError(
                "as_of(T) requires a timezone-aware decision instant T "
                "(e.g. pd.Timestamp('2026-03-10 21:00', tz='UTC')). "
                "A tz-naive T is ambiguous and forbidden."
            )
        return AsOfView(self, ts.tz_convert("UTC"))


class AsOfView:
    """A read-only view of the store as of decision instant ``T``.

    Every accessor applies the correct PIT filter. Outside of :meth:`as_of`,
    nothing else in the system is permitted to read raw tables for trading
    logic — this class is the integrity choke point.
    """

    def __init__(self, store: PITStore, T: pd.Timestamp):
        self.store = store
        self.T = T                          # tz-aware UTC decision instant
        self.asof_date = _eastern_date(T)   # exchange-local session date

    # -- prices ------------------------------------------------------------
    def prices(self, symbol: str, adjust: bool = True) -> pd.DataFrame:
        raw = self.store._raw("prices")
        if raw.empty:
            return raw.copy()
        df = raw[(raw["symbol"] == symbol) & (raw["date"] <= self.asof_date)]
        df = df.sort_values("date").reset_index(drop=True)
        if df.empty or not adjust:
            return df
        actions = self.store._raw("corp_actions")
        actions = actions[actions["symbol"] == symbol] if not actions.empty else actions
        if actions.empty:
            return df                          # no actions -> raw is already adjusted
        price_f = ca.adjustment_factors(actions, df, self.asof_date)
        vol_f = ca.volume_adjustment_factors(actions, df, self.asof_date)
        return ca.apply_adjustment(df, price_f, vol_f)

    # -- event tables ------------------------------------------------------
    def _event(self, table: str, col: str, value: str | None) -> pd.DataFrame:
        raw = self.store._raw(table)
        if raw.empty:
            return raw.copy()
        mask = raw["available_at"] <= self.T
        if value is not None:
            mask &= raw[col] == value
        return raw[mask].sort_values("available_at").reset_index(drop=True)

    def filings(self, symbol: str | None = None) -> pd.DataFrame:
        return self._event("filings", "symbol", symbol)

    def form4(self, symbol: str | None = None) -> pd.DataFrame:
        return self._event("form4", "symbol", symbol)

    def news(self, symbol: str | None = None) -> pd.DataFrame:
        return self._event("news", "symbol", symbol)

    def links(self, focal_symbol: str | None = None) -> pd.DataFrame:
        return self._event("links", "focal_symbol", focal_symbol)

    def fundamentals(self, symbol: str | None = None) -> pd.DataFrame:
        return self._event("fundamentals", "symbol", symbol)

    # -- universe / membership --------------------------------------------
    def constituents(self) -> pd.DataFrame:
        df = self.store._raw("constituents")
        if df.empty:
            return df.copy()
        started = df["start_date"] <= self.asof_date
        # end_date is exclusive; NaT means still active (no leakage either way).
        active = df["end_date"].isna() | (df["end_date"] > self.asof_date)
        return df[started & active].reset_index(drop=True)

    def universe(self) -> list[str]:
        members = self.constituents()
        if members.empty:
            return []
        return sorted(members["symbol"].unique().tolist())


def assert_no_leakage(view: AsOfView) -> bool:
    """Assert the view leaks nothing from after its decision time.

    The generic point-in-time canary used by the CI test across many T:

    * no event row returned has ``available_at > T``;
    * no price bar returned has ``date > T``'s session date;
    * no constituent membership starts after T's session date.

    Raises AssertionError on any violation; returns True if clean.
    """
    T, asof_date = view.T, view.asof_date

    for table, time_col in EVENT_TABLES.items():
        accessor = getattr(view, table)
        df = accessor()
        if not df.empty:
            latest = pd.to_datetime(df[time_col]).max()
            assert latest <= T, (
                f"LEAKAGE in {table!r}: row available_at {latest} > decision time {T}"
            )

    prices = view.store.read_table("prices")
    for symbol in prices["symbol"].unique() if not prices.empty else []:
        df = view.prices(symbol, adjust=False)
        if not df.empty:
            latest = pd.to_datetime(df["date"]).max()
            assert latest <= asof_date, (
                f"LEAKAGE in prices for {symbol!r}: bar {latest} > session {asof_date}"
            )

    members = view.constituents()
    if not members.empty:
        latest_start = pd.to_datetime(members["start_date"]).max()
        assert latest_start <= asof_date, (
            f"LEAKAGE in constituents: membership starts {latest_start} > "
            f"session {asof_date}"
        )

    return True
