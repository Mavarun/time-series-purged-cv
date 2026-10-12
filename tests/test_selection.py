"""Selection diagnostics with costs: null vs planted edge, causality, costing."""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv.selection import (
    default_rule_grid,
    expanding_vol,
    make_ar_returns,
    net_returns_from_positions,
    net_returns_with_impact,
    rule_positions,
    run_selection_diagnostics,
    trailing_scores,
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


def test_expanding_vol_is_causal_and_positive():
    r = make_ar_returns(300, phi=0.0, seed=3)
    v = expanding_vol(r, min_periods=20)
    assert len(v) == len(r) and (v > 0).all()
    r2 = r.copy()
    r2[200:] = np.random.default_rng(1).normal(0, 0.05, size=len(r) - 200)
    v2 = expanding_vol(r2, min_periods=20)
    np.testing.assert_allclose(v[:200], v2[:200])


def test_z_threshold_positions_differ_under_full_sample_vol():
    r = make_ar_returns(800, phi=0.05, seed=5)
    grid = [(5, 1, 0.5), (21, -1, 0.5)]
    P_exp = rule_positions(r, grid, vol_mode="expanding")
    P_full = rule_positions(r, grid, vol_mode="full_sample")
    assert not np.allclose(P_exp, P_full)
    with pytest.raises(ValueError):
        rule_positions(r, grid, vol_mode="bogus")


def test_impact_costs_exceed_linear_bps():
    r = np.array([0.0, 0.01, -0.01, 0.02, 0.0])
    pos = np.array([0.0, 1.0, -1.0, 1.0, 0.0])
    g0, c0, n0 = net_returns_with_impact(r, pos, cost_bps=10.0, impact_coef=0.0)
    g1, c1, n1 = net_returns_with_impact(r, pos, cost_bps=10.0, impact_coef=0.01)
    np.testing.assert_allclose(g0, g1)
    assert (c1 >= c0).all() and c1.sum() > c0.sum()
    np.testing.assert_allclose(n1, g1 - c1)


def test_trailing_scores_shrink_and_turnover():
    rng = np.random.default_rng(0)
    net = rng.normal(0, 0.01, size=(200, 4))
    net[:, 0] += 0.002
    P = np.zeros((200, 4))
    P[::2, 0] = 1.0
    P[1::2, 0] = -1.0
    P[:, 1] = 1.0
    plain = trailing_scores(net, P, mode="sharpe")
    shrunk = trailing_scores(net, P, mode="shrunk_sharpe", shrink_intensity=1.0)
    assert np.allclose(shrunk, shrunk.mean())
    with_to = trailing_scores(net, P, mode="sharpe_minus_turnover", turnover_penalty=50.0)
    assert with_to[0] < plain[0]
    with pytest.raises(ValueError):
        trailing_scores(net, mode="nope")


def test_turnover_penalty_reduces_switches_on_null():
    r = make_ar_returns(1500, phi=0.0, seed=2)
    P = rule_positions(r, default_rule_grid())
    plain = walk_forward_select(r, P, cost_bps=5.0, train_window=504, step=63, mode="sharpe")
    penal = walk_forward_select(
        r, P, cost_bps=5.0, train_window=504, step=63, mode="sharpe_minus_turnover", turnover_penalty=10.0
    )
    assert penal["n_switches"] <= plain["n_switches"]
    assert penal["oos_costs"].mean() <= plain["oos_costs"].mean() + 1e-12


def test_selector_ablation_reports_all_modes():
    from purged_cv.selection import run_selector_ablation

    rep = run_selector_ablation(n=900, phi=0.1, seed=0, cost_bps=2.0, train_window=300, step=50)
    assert set(rep["by_impact"]) == {"0.0", "0.0005", "0.001"}
    modes = set(rep["by_impact"]["0.0"])
    assert modes == {
        "sharpe",
        "shrunk_sharpe",
        "sharpe_minus_turnover",
        "shrunk_sharpe_minus_turnover",
    }
    # Honest: turnover penalty cuts switches; a lag-1 edge is high turnover so
    # penalised OOS Sharpe can fall. Only require fewer/equal switches.
    edge = rep["by_impact"]["0.0"]
    assert edge["sharpe_minus_turnover"]["n_switches"] <= edge["sharpe"]["n_switches"]


def test_selector_ablation_script_builds_offline_report():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_selector_ablation.py"
    spec = importlib.util.spec_from_file_location("run_selector_ablation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(
        ["--seeds", "0", "--n-bars", "900", "--phis", "0.0", "--cost-bps", "2", "--impact-coefs", "0.0"]
    )
    rep = mod.build(args)
    assert "runs" in rep and len(rep["runs"]) == 1
    md = mod.summarise(rep)
    assert "Walk-forward selector ablation" in md
