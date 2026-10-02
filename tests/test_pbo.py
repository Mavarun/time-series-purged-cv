"""PBO via CSCV: noise vs genuine edge, and input validation."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from purged_cv.overfitting import _sharpe_from_stats, probability_of_backtest_overfitting


def test_block_sharpe_matches_direct_computation():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, size=(100, 4))
    direct = x.mean(0) / x.std(0, ddof=1)
    fast = _sharpe_from_stats(100.0, x.sum(0), (x**2).sum(0))
    np.testing.assert_allclose(fast, direct, rtol=1e-10)


def test_pure_noise_strategies_have_pbo_near_one_half():
    # Ground truth: no configuration has an edge, so the IS winner is a coin
    # flip OOS. Averaged over seeds PBO should sit near 0.5.
    pbos = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        R = rng.normal(0.0, 0.01, size=(1000, 40))
        res = probability_of_backtest_overfitting(R, n_partitions=10)
        assert res.n_splits == comb(10, 5)
        pbos.append(res.pbo)
    assert 0.35 < np.mean(pbos) < 0.65
    assert abs(res.summary()["mean_oos_sharpe_of_best"]) < 0.05


def test_one_genuine_strategy_gives_low_pbo():
    # 2000 rows so the planted per-period Sharpe of 0.2 (~3.2 annualised) is
    # well resolved; with 1000 rows sampling noise alone can halve it.
    rng = np.random.default_rng(1)
    R = rng.normal(0.0, 0.01, size=(2000, 40))
    R[:, 7] += 0.002
    res = probability_of_backtest_overfitting(R, n_partitions=10)
    assert res.pbo < 0.05
    assert np.mean(res.is_best_index == 7) > 0.9
    assert res.prob_oos_loss < 0.05


def test_degradation_slope_negative_for_noise():
    # For noise, IS and OOS are complementary halves of the same series, so a
    # luckier IS winner tends to do *worse* OOS (negative slope).
    rng = np.random.default_rng(3)
    R = rng.normal(0.0, 0.01, size=(1200, 30))
    res = probability_of_backtest_overfitting(R, n_partitions=12)
    assert res.degradation_slope < 0.0


def test_validation():
    R = np.zeros((40, 3))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(R, n_partitions=5)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((40, 1)), n_partitions=4)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((10, 3)), n_partitions=8)
