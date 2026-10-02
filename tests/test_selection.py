"""Selection diagnostics with costs: null vs planted edge, causality, costing."""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv.selection import (
    default_rule_grid,
    make_ar_returns,
    net_returns_from_positions,
    rule_positions,
    run_selection_diagnostics,
    walk_forward_select,
)
from purged_cv.sharpe import deflated_sharpe_for_selection


def test_rule_positions_are_causal():
    r = make_ar_returns(400, phi=0.1, seed=0)
    grid = default_rule_grid(lookbacks=(1, 5, 21))
    P = rule_positions(r, grid)
    r2 = r.copy()
    t0 = 250
    r2[t0:] = np.random.default_rng(9).normal(0, 0.01, size=len(r) - t0)
    P2 = rule_positions(r2, grid)
    # positions held up to and including bar t0 were decided by t0-1;
    # (the full-sample vol used for z-scaling is the one global input, so
    # compare only the zero-threshold rules, which do not use it)
    zero_thr = [j for j, (_, _, z) in enumerate(grid) if z == 0.0]
    np.testing.assert_array_equal(P[: t0 + 1, zero_thr], P2[: t0 + 1, zero_thr])
    assert (P[0] == 0).all()


def test_costs_are_charged_on_position_changes():
    r = np.array([0.0, 0.01, -0.01, 0.02])
    pos = np.array([0.0, 1.0, -1.0, -1.0])
    g, c, n = net_returns_from_positions(r, pos, cost_bps=10.0)
    np.testing.assert_allclose(c, [0.0, 0.001, 0.002, 0.0])
    np.testing.assert_allclose(n, g - c)
    g0, c0, n0 = net_returns_from_positions(r, pos, cost_bps=0.0)
    np.testing.assert_allclose(n0, g0)


def test_walk_forward_charges_switching_trades():
    r = make_ar_returns(900, phi=0.0, seed=4)
    P = rule_positions(r, default_rule_grid(lookbacks=(1, 3, 8)))
    wf = walk_forward_select(r, P, cost_bps=5.0, train_window=300, step=50)
    assert len(wf["oos_net"]) == 600
    assert len(wf["chosen"]) == 12
    np.testing.assert_allclose(wf["oos_net"], wf["oos_gross"] - wf["oos_costs"])
    with pytest.raises(ValueError):
        walk_forward_select(r, P, 5.0, train_window=900)


def test_null_selection_looks_good_in_sample_but_not_out_of_sample():
    runs = [run_selection_diagnostics(phi=0.0, cost_bps=2.0, seed=s, n_partitions=10) for s in range(4)]
    is_sr = np.mean([d.is_best_net_sharpe_ann for d in runs])
    wf_sr = np.mean([d.wf_oos_net_sharpe_ann for d in runs])
    assert is_sr > 0.25  # the naive backtest of the best of 40 trials
    assert wf_sr < is_sr - 0.4  # walk-forward OOS collapses
    assert np.mean([d.pbo for d in runs]) > 0.4
    assert max(d.dsr_null for d in runs) < 0.95


def test_planted_edge_survives_pbo_dsr_and_walk_forward_net_of_costs():
    for s in (0, 1):
        d = run_selection_diagnostics(phi=0.15, cost_bps=2.0, seed=s, n_partitions=10)
        assert d.best_trial[0] == 1 and d.best_trial[1] == 1  # lag-1 momentum
        assert d.pbo < 0.05
        assert d.dsr_null > 0.95
        assert d.wf_oos_net_sharpe_ann > 1.0
        assert d.wf_oos_net_sharpe_ann < d.wf_oos_gross_sharpe_ann


def test_higher_costs_lower_net_performance():
    lo = run_selection_diagnostics(phi=0.1, cost_bps=2.0, seed=1, n_partitions=10)
    hi = run_selection_diagnostics(phi=0.1, cost_bps=5.0, seed=1, n_partitions=10)
    assert hi.is_best_net_sharpe_ann < lo.is_best_net_sharpe_ann
    assert hi.wf_oos_net_sharpe_ann < lo.wf_oos_net_sharpe_ann
    assert hi.wf_oos_cost_drag_bps_per_day > lo.wf_oos_cost_drag_bps_per_day


def test_cross_sectional_dsr_over_deflates_mirrored_grid():
    # Documented weakness: long/short mirror trials of a real edge inflate the
    # cross-sectional Sharpe variance, so that DSR rejects a genuine strategy.
    d = run_selection_diagnostics(phi=0.15, cost_bps=2.0, seed=0, n_partitions=10)
    assert d.dsr_cross_sectional < 0.5 < d.dsr_null
    with pytest.raises(ValueError):
        deflated_sharpe_for_selection(np.zeros((10, 3)), sr_variance="bogus")


def test_diagnostics_script_runs_offline_small():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_overfitting_diagnostics.py"
    spec = importlib.util.spec_from_file_location("diag_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--seeds", "1", "--n-bars", "900", "--cost-bps", "2", "--n-partitions", "6"])
    rep = mod.build_report(args)
    assert set(rep) == {"leakage", "leakage_controls", "selection"}
    assert len(rep["selection"]) == 2
