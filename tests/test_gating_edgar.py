"""Validation gating, Form 4 parsing, and universe diversity (offline)."""

from harness.data import loader


def _sector_of(sym):
    return next(etf for etf, tickers in loader.LIVE_UNIVERSE.items() if sym in tickers)


def test_universe_roundrobin_spans_sectors():
    syms = loader.live_symbols(8)
    assert len(syms) == 8
    assert len({_sector_of(s) for s in syms}) >= 5   # small cap still spans sectors


_FORM4_BUY = """<ownershipDocument>
  <reportingOwner><reportingOwnerRelationship>
    <isDirector>1</isDirector></reportingOwnerRelationship></reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>100</value></transactionShares>
      <transactionPricePerShare><value>50</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""

_FORM4_SELL = _FORM4_BUY.replace("<transactionCode>P</transactionCode>",
                                 "<transactionCode>S</transactionCode>")


def test_parse_form4_purchase():
    r = loader._parse_form4(_FORM4_BUY)
    assert r is not None
    assert r["txn_code"] == "P" and r["shares"] == 100 and r["value"] == 5000.0
    assert r["insider_role"] == "Director"


def test_parse_form4_ignores_sales():
    assert loader._parse_form4(_FORM4_SELL) is None
    assert loader._parse_form4("<not-xml") is None


def test_gating_roundtrip_and_filter(tmp_path, monkeypatch):
    import app.gating as gating
    from app.config import AppConfig
    from app.runner import _gated_edges

    monkeypatch.setattr(gating, "VALIDATED_PATH", tmp_path / "validated.json")

    gating.save_validated(["edge01_filing", "edge08_momo"], "synthetic")
    loaded = gating.load_validated()
    assert set(loaded["passed"]) == {"edge01_filing", "edge08_momo"}
    assert loaded["data_source"] == "synthetic"

    cfg = AppConfig(only_validated_edges=True, data_source="synthetic")
    allowed = _gated_edges(cfg, lambda _m: None)
    ids = {getattr(E, "edge_id") for E in allowed}
    assert ids == {"edge01_filing", "edge08_momo"}


def test_gating_off_uses_all_edges():
    from app.config import AppConfig
    from app.runner import _gated_edges
    from harness.signals import ALL_FREE_EDGES

    allowed = _gated_edges(AppConfig(only_validated_edges=False), lambda _m: None)
    assert len(allowed) == len(ALL_FREE_EDGES)


def test_gating_blocks_when_nothing_validated(tmp_path, monkeypatch):
    import app.gating as gating
    from app.config import AppConfig
    from app.runner import _gated_edges

    monkeypatch.setattr(gating, "VALIDATED_PATH", tmp_path / "none.json")
    allowed = _gated_edges(AppConfig(only_validated_edges=True), lambda _m: None)
    assert allowed == []
