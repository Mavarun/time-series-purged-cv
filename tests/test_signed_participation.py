import numpy as np
import pytest

from purged_cv.selection import default_rule_grid, make_ar_returns, net_returns_from_positions, rule_positions
from purged_cv.sharpe import (
    deflated_sharpe_for_selection,
    effective_trials_participation,
    effective_trials_signed,
)


def test_no_mirrors_equals_participation():
    M = np.random.default_rng(0).standard_normal((500, 8))
    assert effective_trials_signed(M) == pytest.approx(effective_trials_participation(M))


def test_exact_mirror_pairs_double_the_count():
    X = np.random.default_rng(1).standard_normal((800, 5))
    M = np.hstack([X, -X])
    pr = effective_trials_participation(M)
    assert pr == pytest.approx(5.0, rel=0.05)
    assert effective_trials_signed(M) == pytest.approx(2 * pr)


def test_capped_at_n_and_partial_mirrors():
    X = np.random.default_rng(2).standard_normal((600, 4))
    M = np.hstack([X, -X[:, :1]])  # one mirror pair among 5 columns
    n = effective_trials_signed(M)
    assert effective_trials_participation(M) < n <= 5.0
    assert n == pytest.approx(min(effective_trials_participation(M) * (1 + 2 / 5), 5.0))


def test_default_grid_mode_is_stricter_than_participation():
    grid = default_rule_grid()
    r = make_ar_returns(504, 0.0, 0.01, 3)
    _, _, net = net_returns_from_positions(r, rule_positions(r, grid), 2.0)
    M = net[90:]
    a = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="participation")
    b = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode="participation_signed")
    assert b.n_effective_trials > a.n_effective_trials and b.expected_max_sharpe > a.expected_max_sharpe
    assert b.dsr <= a.dsr
    with pytest.raises(ValueError):
        deflated_sharpe_for_selection(M, n_trials_mode="nope")
