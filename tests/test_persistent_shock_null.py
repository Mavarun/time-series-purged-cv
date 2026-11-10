"""Persistent-shock null (slice 7): labels stay correlated past the horizon."""

import numpy as np
import pytest

from purged_cv.leakage import make_autocorrelated_null_dataset as make


def _acf(y, lag):
    z = np.asarray(y, float) - np.mean(y)
    return float(np.dot(z[:-lag], z[lag:]) / np.dot(z, z))


def test_default_draws_are_unchanged():
    a = make(n_samples=400, horizon=10, seed=3)
    b = make(n_samples=400, horizon=10, seed=3, shock_phi=0.0)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.X, b.X)
    assert b.meta["shock_phi"] == 0.0


def test_features_do_not_depend_on_shock_phi():
    a = make(n_samples=300, horizon=5, seed=1)
    b = make(n_samples=300, horizon=5, seed=1, shock_phi=0.9)
    np.testing.assert_array_equal(a.X, b.X)
    assert not np.array_equal(a.y, b.y)


def test_labels_correlate_beyond_the_horizon_only_with_persistent_shocks():
    h, lag = 5, 15  # 10 bars past the last shared shock
    iid = np.mean([_acf(make(n_samples=4000, horizon=h, seed=s).y, lag) for s in range(6)])
    ar = np.mean([_acf(make(n_samples=4000, horizon=h, seed=s, shock_phi=0.9).y, lag) for s in range(6)])
    assert abs(iid) < 0.03
    assert ar > 0.15


def test_shock_variance_is_unit():
    d = make(n_samples=20000, horizon=1, seed=0, shock_phi=0.8)
    # horizon-1 forward return is one shock
    assert np.std(d.forward_returns) == pytest.approx(1.0, abs=0.05)


def test_shock_phi_validated():
    with pytest.raises(ValueError):
        make(shock_phi=1.0)
    with pytest.raises(ValueError):
        make(shock_phi=-0.1)
