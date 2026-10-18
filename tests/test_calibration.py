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


def test_calibration_script_builds_small_report_offline():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_dsr_calibration.py"
    spec = importlib.util.spec_from_file_location("run_dsr_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(
        ["--n-seeds", "2", "--n-bars", "700", "--phis", "0.0", "--cost-bps", "2", "--n-sims", "300", "--no-write"]
    )
    rep = mod.build(args)
    assert len(rep["rows"]) == 1
    assert "Selection-hurdle calibration" in mod.summarise(rep)


def test_slice4_helpers_are_exported_from_package_root():
    import purged_cv

    for name in (
        "deflated_sharpe_for_selection",
        "effective_trials_participation",
        "implied_effective_trials",
        "max_sharpe_pvalue",
        "null_max_sharpe_draws",
        "calibration_table",
        "empirical_bayes_sharpes",
    ):
        assert name in purged_cv.__all__ and callable(getattr(purged_cv, name))
