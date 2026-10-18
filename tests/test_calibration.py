"""Selection-hurdle calibration: null false positives and edge power."""

from __future__ import annotations

from purged_cv.calibration import METHODS, calibration_table, selection_tests_one_seed


def test_one_seed_reports_every_method_and_effective_n():
    out = selection_tests_one_seed(0.0, seed=0, cost_bps=2.0, n=900, n_sims=500)
    for m in METHODS:
        assert m in out
    assert out["n_eff_raw"] == 40.0
    # the 40-rule grid is heavily correlated (mirrors, shared lookbacks)
    assert out["n_eff_participation"] < out["n_eff_simulated"] < 40.0
    assert 0.0 < out["max_p"] <= 1.0


def test_null_false_positives_stay_low_and_edge_power_is_higher():
    rows = calibration_table(phis=(0.0, 0.15), seeds=range(12), cost_bps=(2.0,), n=1500, n_sims=1000)
    null, edge = rows
    for m in METHODS:
        assert null[m] <= 0.25
        assert edge[m] >= null[m]
    # correlation-aware hurdles are never stricter than counting 40 independent trials
    assert edge["dsr_simulated"] >= edge["dsr_raw"]
    assert edge["dsr_participation"] >= edge["dsr_simulated"]
