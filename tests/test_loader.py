"""Live price-loader behaviour (network calls are monkeypatched)."""

from __future__ import annotations

import pandas as pd

from harness.data import loader


def _multi(symbols, dates):
    """A yfinance-style multi-ticker frame: column MultiIndex (field, ticker)."""
    cols, data = [], {}
    for s in symbols:
        cols += [("Close", s), ("Volume", s)]
        data[("Close", s)] = [10.0, 11.0, 12.0]
        data[("Volume", s)] = [100, 100, 100]
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df


def test_batch_summarises_missing_names_cleanly(monkeypatch):
    # A name Yahoo can't serve (e.g. the thinly-covered class-A line CWEN-A) must
    # be dropped with ONE tidy summary line, not yfinance's per-name spam.
    import yfinance as yf
    dates = pd.date_range("2026-07-01", periods=3)

    def fake_download(part, **kw):
        part = part if isinstance(part, list) else [part]
        served = [s for s in part if s != "CWEN-A"]      # Yahoo has no CWEN-A
        return _multi(served, dates) if served else pd.DataFrame()

    monkeypatch.setattr(yf, "download", fake_download)
    lines: list[str] = []
    closes, vols = loader.fetch_closes_volumes_batch(
        ["GOOD", "CWEN-A", "ALSO"], "2026-07-01", emit=lines.append)

    assert "GOOD" in closes.columns and "ALSO" in closes.columns
    assert "CWEN-A" not in closes.columns                 # dropped, not crashed
    summary = [ln for ln in lines if "no Yahoo price data" in ln]
    assert summary and "CWEN-A" in summary[0] and "1 name" in summary[0]


def test_batch_quiet_when_every_name_resolves(monkeypatch):
    import yfinance as yf
    dates = pd.date_range("2026-07-01", periods=3)
    monkeypatch.setattr(yf, "download",
                        lambda part, **kw: _multi(
                            part if isinstance(part, list) else [part], dates))
    lines: list[str] = []
    closes, _ = loader.fetch_closes_volumes_batch(
        ["AAA", "BBB"], "2026-07-01", emit=lines.append)

    assert list(closes.columns) == ["AAA", "BBB"]
    assert not any("no Yahoo price data" in ln for ln in lines)  # no false alarm
