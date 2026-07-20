"""Deployment-readiness math and gating (PSR / MinTRL / calibration / verdict)."""

import math

from system.reflection import readiness as R


def _record(n, win_rate, win_ret=8.0, loss_ret=-4.0, conv=0.6,
            months=("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"),
            sectors=("Tech", "Health", "Energy", "Financials")):
    """A synthetic scored-call record with a target win rate, spread across
    months and sectors."""
    rows = []
    n_win = round(n * win_rate)
    for i in range(n):
        win = i < n_win
        rows.append({"return_pct": win_ret if win else loss_ret,
                     "conviction": conv,
                     "evaluated_on": months[i % len(months)] + "-15",
                     "sector": sectors[i % len(sectors)]})
    return rows


def test_psr_rises_with_a_stronger_record():
    weak = [1.0, -1.0, 1.0, -1.0, 0.5, -0.4] * 3
    strong = [6.0, 7.0, -2.0, 8.0, 5.0, -1.5] * 5
    assert R.probabilistic_sharpe(weak) < R.probabilistic_sharpe(strong)
    assert 0.0 <= R.probabilistic_sharpe(strong) <= 1.0


def test_min_track_record_infinite_when_edge_not_positive():
    flat = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0]
    assert R.min_track_record_length(flat) == math.inf
    winning = [5.0, 6.0, -2.0, 7.0, 4.0, -1.0, 6.0, 5.0]
    mintrl = R.min_track_record_length(winning)
    assert math.isfinite(mintrl) and mintrl > 1


def test_brier_and_reliability():
    # Perfectly confident and correct -> Brier 0; bands ascending -> monotonic.
    assert R.brier_score([1.0, 1.0], [1, 1]) == 0.0
    bands = [{"lo": 0.0, "hi": 0.5, "n": 5, "win_rate_pct": 40.0},
             {"lo": 0.5, "hi": 1.0, "n": 5, "win_rate_pct": 70.0}]
    assert R.reliability(bands)["monotonic"] is True
    inv = [{"lo": 0.0, "hi": 0.5, "n": 5, "win_rate_pct": 70.0},
           {"lo": 0.5, "hi": 1.0, "n": 5, "win_rate_pct": 40.0}]
    assert R.reliability(inv)["monotonic"] is False


def test_max_drawdown():
    assert abs(R.max_drawdown([1.0, 1.2, 0.9, 1.1]) - (-0.25)) < 1e-6
    assert R.max_drawdown([1.0]) == 0.0


def test_thin_record_is_not_ready():
    rd = R.assess(_record(5, 0.8), equity_curve=[1, 1.1, 1.2, 1.0, 1.1, 1.2],
                  shadow_days=10)
    assert rd["stage"] == "shadow"
    assert "NOT READY" in rd["verdict"]
    assert any(g["name"] == "Sample size" and g["status"] == "fail"
               for g in rd["gates"])


def test_strong_diverse_record_clears_critical_gates():
    rows = _record(40, 0.62)
    eq, v = [1.0], 1.0
    for r in sorted(rows, key=lambda x: x["evaluated_on"]):
        v *= 1 + r["return_pct"] / 100.0
        eq.append(round(v, 4))
    bands = [{"lo": 0.0, "hi": 0.5, "n": 0, "win_rate_pct": None},
             {"lo": 0.5, "hi": 0.65, "n": 40, "win_rate_pct": 62.0},
             {"lo": 0.65, "hi": 1.01, "n": 0, "win_rate_pct": None}]
    rd = R.assess(rows, equity_curve=eq, calibration_bands=bands, shadow_days=150)
    by = {g["name"]: g["status"] for g in rd["gates"]}
    assert by["Sample size"] == "pass"
    assert by["Edge is real (PSR)"] == "pass"          # PSR clears 95%
    assert rd["psr"] >= R.PSR_TARGET
    assert rd["score"] >= 70
    # All-pass is required for the tiny-capital recommendation.
    if all(g["status"] == "pass" for g in rd["gates"]):
        assert rd["stage"] == "challenger"


def test_human_gate_is_always_present():
    rd = R.assess(_record(40, 0.62), shadow_days=150)
    assert "human action" in rd["human_gate"]


def test_every_gate_carries_a_next_step_field():
    # Explainability: each gate exposes next_step; passing gates carry None,
    # every non-passing gate a concrete, non-empty action to clear it.
    rd = R.assess(_record(5, 0.8), equity_curve=[1, 1.1, 1.2, 1.0, 1.1, 1.2],
                  shadow_days=10)
    for g in rd["gates"]:
        assert "next_step" in g
        if g["status"] == "pass":
            assert g["next_step"] is None
        else:
            assert isinstance(g["next_step"], str) and g["next_step"].strip()


def test_sample_size_next_step_quantifies_the_gap():
    # A thin record (5 calls) must say exactly how many more to reach the bar.
    rd = R.assess(_record(5, 0.8), shadow_days=10)
    g = next(g for g in rd["gates"] if g["name"] == "Sample size")
    assert g["status"] == "fail"
    assert "more call" in g["next_step"]
    assert str(R.MIN_SCORED_PROGRESS - 5) in g["next_step"]


def test_diversity_next_step_names_the_missing_axis():
    # Enough months but only one sector -> the step should ask for more sectors.
    rows = _record(20, 0.6,
                   months=("2026-01", "2026-02", "2026-03", "2026-04",
                           "2026-05", "2026-06"),
                   sectors=("Tech",))
    rd = R.assess(rows, shadow_days=150)
    g = next(g for g in rd["gates"] if g["name"] == "Diversity")
    assert g["status"] != "pass"
    assert "sector" in g["next_step"]


def test_passed_gate_has_no_next_step():
    # A strong, diverse record clears Sample size -> its next_step is None.
    g = next(g for g in R.assess(_record(40, 0.62), shadow_days=150)["gates"]
             if g["name"] == "Sample size")
    assert g["status"] == "pass" and g["next_step"] is None
