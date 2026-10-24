import numpy as np
import pytest

from purged_cv.bootstrap import reality_check_draws, reality_check_pvalue, stationary_bootstrap_indices


def test_indices_are_in_range_reproducible_and_blocky():
    a = stationary_bootstrap_indices(500, 200, mean_block=10.0, seed=3)
    b = stationary_bootstrap_indices(500, 200, mean_block=10.0, seed=3)
    assert a.shape == (200, 500) and np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 500
    # a block continues with index + 1 (mod n); breaks happen with prob 1/mean_block
    cont = (np.diff(a, axis=1) % 500) == 1
    assert abs(cont.mean() - 0.9) < 0.01


def test_mean_block_one_is_the_iid_bootstrap():
    a = stationary_bootstrap_indices(1000, 50, mean_block=1.0, seed=0)
    cont = (np.diff(a, axis=1) % 1000) == 1
    assert cont.mean() < 0.01


@pytest.mark.parametrize("bad", [0.5, 0.0])
def test_bad_block_length_is_rejected(bad):
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(10, 2, mean_block=bad)


def test_null_draws_are_recentred_and_the_pvalue_is_roughly_uniform():
    rng = np.random.default_rng(0)
    ps = []
    for s in range(60):
        M = rng.standard_normal((400, 5))
        ps.append(reality_check_pvalue(M, n_boot=199, mean_block=5.0, seed=s))
    ps = np.array(ps)
    assert 0.3 < ps.mean() < 0.7
    assert np.mean(ps < 0.05) <= 0.15


def test_a_strong_edge_is_detected_and_the_statistic_is_the_scaled_best_sharpe():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((1000, 4))
    M[:, 2] += 0.2
    stat, draws = reality_check_draws(M, n_boot=199, mean_block=5.0, seed=0)
    sr = M.mean(0) / M.std(0, ddof=1)
    assert np.isclose(stat, np.sqrt(1000) * sr.max())
    assert abs(np.median(draws)) < stat
    assert reality_check_pvalue(M, n_boot=199, mean_block=5.0, seed=0) <= 0.01
