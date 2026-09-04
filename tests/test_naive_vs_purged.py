"""Prove engineered leakage inflates naive CV relative to purged CV."""

from __future__ import annotations

import numpy as np

from purged_cv.data import make_engineered_leakage_dataset, make_synthetic_ar_dataset
from purged_cv.metrics import compare_naive_vs_purged_cv
from purged_cv.weights import average_uniqueness, sample_weights_from_returns


def test_engineered_leakage_naive_scores_higher_than_purged():
    ds = make_engineered_leakage_dataset(n_samples=1000, horizon=25, seed=0)
    result = compare_naive_vs_purged_cv(
        ds.X,
        ds.y,
        ds.label_start_times,
        ds.label_end_times,
        n_splits=5,
        embargo_pct=0.05,
        model_kind="logistic",
        shuffle_naive=True,
        random_state=0,
    )
    assert result.naive_mean > result.purged_mean + 0.04, (
        f"expected naive >> purged; got naive={result.naive_mean:.3f} "
        f"purged={result.purged_mean:.3f}"
    )
    # Purged should look closer to chance than the inflated naive score.
    assert result.purged_mean < result.naive_mean


def test_synthetic_ar_reports_both_scores_honestly():
    ds = make_synthetic_ar_dataset(
        n_bars=700, horizon=10, n_lags=5, ar_coef=0.2, seed=1
    )
    result = compare_naive_vs_purged_cv(
        ds.X,
        ds.y,
        ds.label_start_times,
        ds.label_end_times,
        n_splits=5,
        embargo_pct=0.01,
        model_kind="ridge",
        shuffle_naive=True,
        random_state=1,
    )
    assert 0.0 <= result.naive_mean <= 1.0
    assert 0.0 <= result.purged_mean <= 1.0
    assert len(result.naive_scores) == 5
    assert len(result.purged_scores) == 5
    # On lag-only features, neither should claim near-perfect accuracy.
    assert result.naive_mean < 0.85
    assert result.purged_mean < 0.85


def test_average_uniqueness_positive_and_bounded():
    ds = make_synthetic_ar_dataset(n_bars=120, horizon=5, seed=3)
    u = average_uniqueness(ds.label_start_times, ds.label_end_times)
    assert len(u) == len(ds.y)
    assert (u > 0).all()
    assert (u <= 1.0 + 1e-9).all()


def test_sample_weights_normalize():
    ds = make_synthetic_ar_dataset(n_bars=120, horizon=5, seed=4)
    w = sample_weights_from_returns(
        ds.forward_returns, ds.label_start_times, ds.label_end_times, normalize=True
    )
    assert len(w) == len(ds.y)
    np.testing.assert_allclose(w.sum(), len(w), rtol=1e-6)
