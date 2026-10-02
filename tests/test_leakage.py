"""Naive K-fold inflates scores on autocorrelated null labels; purged/CPCV do not.

Ground truth accuracy is 0.5 by construction (features independent of labels).
Results are averaged over 3 seeds because overlapping labels leave only
~n/h effectively independent observations per dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from purged_cv.leakage import leakage_sweep, make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier


def test_null_dataset_has_autocorrelated_labels_and_uninformative_features():
    ds = make_autocorrelated_null_dataset(n_samples=1500, horizon=20, seed=0)
    y = ds.y.astype(float)
    lag1 = np.corrcoef(y[:-1], y[1:])[0, 1]
    assert lag1 > 0.8  # adjacent labels share 19 of 20 shocks
    assert ds.X.shape == (1500, 3)
    assert (ds.label_end_times - ds.label_start_times == 20).all()
    # feature persistence matches the requested AR(1) coefficient
    x = ds.X[:, 0]
    assert np.corrcoef(x[:-1], x[1:])[0, 1] > 0.9
    with pytest.raises(ValueError):
        make_autocorrelated_null_dataset(horizon=0)


def test_naive_kfold_inflates_while_purged_and_cpcv_stay_at_chance():
    r = leakage_sweep(seeds=(0, 1, 2), horizon=20, feature_phi=0.98, model_kind="knn")
    assert r["naive_kfold"] > 0.62
    assert r["naive_minus_purged"] > 0.12
    assert abs(r["purged_kfold"] - 0.5) < 0.06
    assert abs(r["cpcv_path_mean"] - 0.5) < 0.06
    # independent fresh sample confirms the realised OOS accuracy is ~chance
    assert abs(r["fresh_holdout"] - 0.5) < 0.05


def test_controls_no_overlap_or_no_feature_persistence_remove_the_leak():
    no_overlap = leakage_sweep(seeds=(0, 1, 2), horizon=1, feature_phi=0.98)
    iid_features = leakage_sweep(seeds=(0, 1, 2), horizon=20, feature_phi=0.0)
    assert abs(no_overlap["naive_minus_purged"]) < 0.04
    assert abs(iid_features["naive_minus_purged"]) < 0.05


def test_inflation_grows_with_label_overlap():
    gaps = [
        leakage_sweep(seeds=(0, 1, 2), horizon=h, feature_phi=0.98)["naive_minus_purged"]
        for h in (1, 5, 20)
    ]
    assert gaps[0] < gaps[1] < gaps[2]


def test_knn_kind_available():
    clf = make_sign_classifier("knn", n_neighbors=3)
    assert clf.named_steps["clf"].n_neighbors == 3
    with pytest.raises(ValueError):
        make_sign_classifier("svm")
