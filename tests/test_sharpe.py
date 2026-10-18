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


def test_effective_trials_participation_limits():
    from purged_cv.sharpe import effective_trials_participation

    rng = np.random.default_rng(0)
    indep = rng.normal(size=(5000, 8))
    assert effective_trials_participation(indep) == pytest.approx(8.0, rel=0.05)
    base = rng.normal(size=(5000, 1))
    clones = np.hstack([base, 2 * base, -base, base + 1.0])
    assert effective_trials_participation(clones) == pytest.approx(1.0, abs=1e-6)
    # a never-trading (constant) column counts as its own direction, not NaN
    with_flat = np.hstack([indep[:, :2], np.zeros((5000, 1))])
    assert effective_trials_participation(with_flat) == pytest.approx(3.0, rel=0.05)


def test_null_max_draws_match_closed_form_for_independent_trials():
    from purged_cv.sharpe import null_max_sharpe_draws

    rng = np.random.default_rng(1)
    M = rng.normal(size=(4000, 20))
    var = 1.0 / 3999
    draws = null_max_sharpe_draws(M, n_sims=20000, seed=0)
    # closed form is an approximation; agree within a few percent at N=20
    assert draws.mean() == pytest.approx(expected_max_sharpe(20, var), rel=0.06)
    # perfectly correlated trials -> max of one draw, mean ~ 0
    one = np.repeat(M[:, :1], 20, axis=1)
    assert abs(null_max_sharpe_draws(one, n_sims=20000, seed=0).mean()) < 0.1 * np.sqrt(var)
    # mirror pair -> E|z| = sqrt(2/pi), larger than the max of 2 iid draws
    mirror = np.hstack([M[:, :1], -M[:, :1]])
    m = null_max_sharpe_draws(mirror, n_sims=40000, seed=0).mean() / np.sqrt(var)
    assert m == pytest.approx(np.sqrt(2 / np.pi), abs=0.02)
    assert m > expected_max_sharpe(2, 1.0)


def test_implied_effective_trials_inverts_closed_form():
    from purged_cv.sharpe import implied_effective_trials

    var = 1.0 / 2000
    for n in (2, 10, 40, 500):
        assert implied_effective_trials(expected_max_sharpe(n, var), var) == pytest.approx(n, rel=1e-6)
    assert implied_effective_trials(-0.01, var) == 1.0
    with pytest.raises(ValueError):
        implied_effective_trials(0.1, 0.0)


def test_selection_modes_raw_unchanged_and_effective_n_lowers_hurdle():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, size=(1500, 5))
    # 40 trials that are 8 noisy copies of 5 underlying series -> N_eff << 40
    M = np.hstack([base + rng.normal(0, 0.002, size=base.shape) for _ in range(8)])
    raw = deflated_sharpe_for_selection(M, sr_variance="null")
    raw2 = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="raw")
    assert raw.dsr == raw2.dsr and raw.n_effective_trials == 40.0
    pr = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="participation")
    sim = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="simulated")
    assert pr.n_effective_trials < 10 and sim.n_effective_trials < 15
    assert sim.expected_max_sharpe < raw.expected_max_sharpe
    assert sim.dsr >= raw.dsr and pr.dsr >= raw.dsr
    again = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="simulated")
    assert again.dsr == sim.dsr  # seeded
    with pytest.raises(ValueError):
        deflated_sharpe_for_selection(M, n_trials_mode="bogus")


def test_max_sharpe_pvalue_null_vs_edge():
    from purged_cv.sharpe import max_sharpe_pvalue

    # calibration on iid nulls: ~5% of p-values below 0.05, mean ~0.5
    ps = np.array(
        [
            max_sharpe_pvalue(np.random.default_rng(s).normal(0, 0.01, size=(2000, 10)), n_sims=2000)
            for s in range(100)
        ]
    )
    assert 0.01 <= (ps < 0.05).mean() <= 0.10
    assert ps.mean() == pytest.approx(0.5, abs=0.08)
    null = np.random.default_rng(5).normal(0, 0.01, size=(2000, 10))
    edge = null.copy()
    edge[:, 3] += 0.002  # per-period SR ~0.2 -> t ~ 9
    assert max_sharpe_pvalue(edge, n_sims=5000) < 0.001
