"""The frozen slice 7 protocol: grid, arms, seeds and the recommendation rule."""

import numpy as np

from purged_cv import embargo_calibration as P


def test_frozen_grid():
    assert P.SHOCK_PHIS == (0.0, 0.5, 0.8, 0.9) and P.HORIZONS == (5, 20)
    assert P.SEEDS == tuple(range(7000, 7100)) and P.N_SAMPLES == 1500
    assert list(P.ARMS) == ["purge_only", "pct1_after", "pct5_after", "acf_after", "acf_both"]
    assert max(P.SEEDS) < 10_000 and min(P.SEEDS) > 5299  # disjoint from slices 4-6


def test_one_seed_runs_every_arm_and_both_sides_keep_fewer_rows():
    r = P.one_seed(0.9, 5, 7000, n_samples=500)
    assert set(r.accuracy) == set(P.ARMS)
    assert r.train_rows["acf_both"] <= r.train_rows["acf_after"] <= r.train_rows["purge_only"]
    assert r.embargo["purge_only"] == 0 and r.embargo["pct1_after"] == 5


def _cell(cand_opt, cand_se, def_opt, keep=1.0):
    return {P.CANDIDATE: {"leak_free": cand_opt <= 2 * cand_se, "keep_vs_default": keep},
            P.DEFAULT_ARM: {"leak_free": def_opt <= 2 * 0.005}}


def test_call_rule():
    cells = {(phi, h): _cell(0.0, 0.005, 0.03 if phi >= 0.8 else 0.0) for phi in P.SHOCK_PHIS for h in P.HORIZONS}
    assert P.call(cells)["recommend"] is True
    cells[(0.0, 5)] = _cell(0.0, 0.005, 0.0, keep=0.9)
    assert P.call(cells)["recommend"] is False
    cells[(0.0, 5)] = _cell(0.0, 0.005, 0.0)
    cells[(0.9, 20)] = _cell(0.05, 0.005, 0.03)
    assert P.call(cells)["recommend"] is False


def test_summarize_shapes():
    rs = [P.SeedResult({a: 0.5 + 0.01 * i for a in P.ARMS}, {a: 1000 for a in P.ARMS}, {a: 0 for a in P.ARMS})
          for i in range(4)]
    s = P.summarize(rs)
    assert np.isclose(s["acf_both"]["optimism"], 0.015) and s["acf_both"]["keep_vs_default"] == 1.0
