"""Confluence engine: percentile, within-family collapse, >=2-family rule, top-K."""

from system.config import SystemConfig
from system.confluence import run_confluence
from system.schemas import SpecialistRead

CFG = SystemConfig()


def R(edge, fam, score, conf=0.6):
    return SpecialistRead(edge, fam, score, conf, "long", [f"{edge}"])


def test_requires_two_families_agreeing():
    reads = {
        # AAA: strong in family A AND family C -> 2 families agree.
        "AAA": [R("edge01", "A", 10.0), R("edge06", "C", 10.0)],
        # BBB: strong only in family A -> 1 family, excluded.
        "BBB": [R("edge01", "A", 1.0), R("edge06", "C", 1.0)],
    }
    cands = run_confluence(reads, CFG)
    syms = {c.symbol for c in cands}
    assert "AAA" in syms and "BBB" not in syms
    aaa = next(c for c in cands if c.symbol == "AAA")
    assert aaa.n_families == 2


def test_within_family_collapse_does_not_double_count():
    # AAA has TWO family-A edges plus one family-C edge. Collapse means family A
    # counts once: agreeing families = {A, C} = 2, not 3.
    reads = {
        "AAA": [R("edge01", "A", 10.0), R("edge02", "A", 10.0), R("edge06", "C", 10.0)],
        "BBB": [R("edge01", "A", 5.0), R("edge02", "A", 5.0), R("edge06", "C", 5.0)],
        "CCC": [R("edge01", "A", 1.0), R("edge02", "A", 1.0), R("edge06", "C", 1.0)],
    }
    cands = run_confluence(reads, CFG)
    aaa = next(c for c in cands if c.symbol == "AAA")
    assert aaa.n_families == 2
    assert set(aaa.families) == {"A", "C"}


def test_strong_single_family_qualifies():
    reads = {
        "AAA": [R("edge08", "E", 100.0)],
        "BBB": [R("edge08", "E", 1.0)],
    }
    cands = run_confluence(reads, CFG)
    aaa = [c for c in cands if c.symbol == "AAA"]
    assert aaa and aaa[0].strong_single


def test_top_k_limit():
    cfg = SystemConfig()
    object.__setattr__(cfg, "top_k", 2)
    reads = {}
    for i in range(5):
        s = f"S{i}"
        reads[s] = [R("edge01", "A", float(i)), R("edge06", "C", float(i))]
    cands = run_confluence(reads, cfg)
    assert len(cands) <= 2
    # Ranked by combined score descending.
    assert cands == sorted(cands, key=lambda c: c.combined_score, reverse=True)
