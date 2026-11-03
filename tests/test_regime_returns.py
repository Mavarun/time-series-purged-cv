import numpy as np
import pytest

from purged_cv.selection import make_ar_returns, make_regime_ar_returns


def test_unconditional_std_matches_vol_and_tails_are_fat_from_mixing():
    # many independent paths: a single 100-bar-regime path is too short to average the chain out
    r = np.concatenate([make_regime_ar_returns(5000, phi=0.0, vol=0.01, seed=s) for s in range(40)])
    assert abs(r.std() / 0.01 - 1.0) < 0.04
    kurt = ((r - r.mean()) ** 4).mean() / r.var() ** 2
    assert kurt > 3.3  # a variance mixture is leptokurtic even with Gaussian shocks


def test_volatility_clusters_but_returns_are_not_autocorrelated_at_phi_zero():
    r = make_regime_ar_returns(200_000, phi=0.0, seed=1)
    ac = lambda x: np.corrcoef(x[1:], x[:-1])[0, 1]  # noqa: E731
    assert abs(ac(r)) < 0.02
    assert ac(r**2) > 0.1


def test_ar_coefficient_is_kept():
    r = make_regime_ar_returns(200_000, phi=0.1, seed=2)
    assert abs(np.corrcoef(r[1:], r[:-1])[0, 1] - 0.1) < 0.02


def test_equal_multipliers_reduce_to_the_constant_vol_ar1_moments():
    r = make_regime_ar_returns(100_000, phi=0.0, vol=0.01, seed=3, vol_mult=(1.0, 1.0))
    g = make_ar_returns(100_000, phi=0.0, vol=0.01, seed=3)
    assert abs(r.std() / g.std() - 1.0) < 0.02


def test_regime_lengths_follow_p_stay():
    # recover the state from the shock scale: switches per bar should be about 1 - p_stay
    r = make_regime_ar_returns(60_000, phi=0.0, seed=5, vol_mult=(0.2, 3.0), p_stay=0.98)
    big = np.abs(r) > np.median(np.abs(r))
    blocks = np.flatnonzero(np.diff(big.astype(int)) != 0)
    # heavy vol separation => clusters: far fewer switches than for iid scale draws (~0.5 per bar)
    assert len(blocks) / len(r) < 0.2


def test_seeded_and_validated():
    assert np.array_equal(make_regime_ar_returns(500, seed=4), make_regime_ar_returns(500, seed=4))
    with pytest.raises(ValueError):
        make_regime_ar_returns(p_stay=1.0)
    with pytest.raises(ValueError):
        make_regime_ar_returns(vol_mult=(0.0, 1.0))
    with pytest.raises(ValueError):
        make_regime_ar_returns(phi=1.0)
