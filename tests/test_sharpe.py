"""PSR / DSR: published example, selection bias on noise, genuine edge."""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv.sharpe import (
    deflated_sharpe_for_selection,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_std_error,
)


def test_reproduces_bailey_lopez_de_prado_2014_numerical_example():
    # JPM 2014 example: annualised SR 2.5, 1250 daily obs, 100 trials,
    # annualised trial-SR variance 0.5, skew -3, kurtosis 10 -> DSR ~ 0.90.
    sr = 2.5 / np.sqrt(250)
    var = 0.5 / 250
    assert expected_max_sharpe(100, var) == pytest.approx(0.1132, abs=5e-4)
    assert deflated_sharpe_ratio(sr, 1250, 100, var, skew=-3, kurtosis=10) == pytest.approx(
        0.9004, abs=1e-3
    )


def test_psr_basic_properties():
    assert probabilistic_sharpe_ratio(0.0, 500) == pytest.approx(0.5)
    assert probabilistic_sharpe_ratio(0.1, 1000) > probabilistic_sharpe_ratio(0.1, 100)
    # negative skew / fat tails widen the SE and lower PSR for SR > 0
    assert probabilistic_sharpe_ratio(0.1, 500, skew=-2, kurtosis=8) < probabilistic_sharpe_ratio(
        0.1, 500
    )
    assert sharpe_std_error(0.0, 101) == pytest.approx(0.1)


def test_min_track_record_length_hits_target_confidence():
    n = min_track_record_length(0.08, confidence=0.95)
    assert probabilistic_sharpe_ratio(0.08, int(np.ceil(n))) >= 0.95
    assert probabilistic_sharpe_ratio(0.08, int(np.floor(n)) - 5) < 0.95
    assert min_track_record_length(0.0) == float("inf")


def test_expected_max_sharpe_grows_with_trials():
    vals = [expected_max_sharpe(n, 0.001) for n in (1, 2, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert deflated_sharpe_ratio(0.1, 1000, 1000, 0.001) < deflated_sharpe_ratio(0.1, 1000, 10, 0.001)


def test_best_of_100_noise_strategies_looks_significant_until_deflated():
    # Ground truth: zero skill everywhere. Naive PSR(0) of the max-Sharpe
    # trial is "significant"; deflating by 100 trials removes that.
    psr, dsr = [], []
    for seed in range(5):
        R = np.random.default_rng(seed).normal(0.0, 0.01, size=(1000, 100))
        rep = deflated_sharpe_for_selection(R)
        psr.append(rep.psr_vs_zero)
        dsr.append(rep.dsr)
    assert min(psr) > 0.95
    assert max(dsr) < 0.95
    assert np.mean(dsr) < 0.7


def test_genuine_edge_survives_deflation():
    for seed in range(3):
        R = np.random.default_rng(seed).normal(0.0, 0.01, size=(2000, 100))
        R[:, 3] += 0.0025  # per-period SR 0.25 (~4 annualised)
        rep = deflated_sharpe_for_selection(R)
        assert rep.best_index == 3
        assert rep.dsr > 0.95
