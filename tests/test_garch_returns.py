import numpy as np
import pytest
from scipy import stats

from purged_cv.selection import make_ar_returns, make_garch_ar_returns


def test_unconditional_std_matches_vol_and_tails_are_fat():
    r = make_garch_ar_returns(n=200_000, phi=0.0, vol=0.01, seed=0)
    assert abs(r.std() / 0.01 - 1.0) < 0.05
    assert stats.kurtosis(r, fisher=False) > 4.0
    g = make_ar_returns(n=200_000, phi=0.0, vol=0.01, seed=0)
    assert abs(stats.kurtosis(g, fisher=False) - 3.0) < 0.1


def test_volatility_clusters_but_returns_are_not_autocorrelated_at_phi_zero():
    r = make_garch_ar_returns(n=100_000, phi=0.0, seed=1)
    ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]  # noqa: E731
    assert abs(ac(r)) < 0.02
    assert ac(r**2) > 0.1


def test_ar_coefficient_is_kept():
    r = make_garch_ar_returns(n=100_000, phi=0.1, seed=2)
    assert abs(np.corrcoef(r[1:], r[:-1])[0, 1] - 0.1) < 0.02


def test_seeded_and_validated():
    assert np.array_equal(make_garch_ar_returns(500, seed=4), make_garch_ar_returns(500, seed=4))
    with pytest.raises(ValueError):
        make_garch_ar_returns(alpha=0.2, beta=0.85)
    with pytest.raises(ValueError):
        make_garch_ar_returns(nu=2.0)
