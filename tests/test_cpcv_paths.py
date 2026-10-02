"""CPCV backtest-path assembly: coverage, OOS guarantee, and skill recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from purged_cv.cpcv import (
    CombinatorialPurgedKFold,
    cpcv_oos_predictions,
    cpcv_path_map,
    path_accuracies,
)
from purged_cv.model import make_sign_classifier


class _Memorizer(ClassifierMixin, BaseEstimator):
    """Predicts 1 iff the row id (column 0) was in its training set."""

    def fit(self, X, y):
        self.seen_ = set(np.asarray(X)[:, 0].astype(int).tolist())
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        ids = np.asarray(X)[:, 0].astype(int)
        return np.array([1 if i in self.seen_ else 0 for i in ids])


def test_path_map_uses_each_split_exactly_k_times():
    cv = CombinatorialPurgedKFold(6, 2)
    pm = cpcv_path_map(cv)
    assert pm.shape == (cv.n_paths, 6) == (5, 6)
    counts = np.bincount(pm.ravel(), minlength=cv.get_n_splits())
    assert (counts == 2).all()  # every split contributes its k=2 groups once
    combos = cv.test_group_combinations()
    for p in range(cv.n_paths):
        for g in range(6):
            assert g in combos[pm[p, g]]


def test_no_path_value_comes_from_a_model_trained_on_that_sample():
    n = 150
    X = np.column_stack([np.arange(n), np.zeros(n)])
    y = np.zeros(n, dtype=int)
    cv = CombinatorialPurgedKFold(
        6, 2, pd.Series(np.arange(n)), pd.Series(np.arange(n) + 4), embargo_pct=0.02
    )
    paths = cpcv_oos_predictions(_Memorizer(), X, y, cv)
    assert paths.predictions.shape == (cv.n_paths, n)
    assert not np.isnan(paths.predictions).any()
    assert (paths.predictions == 0).all()


def test_paths_recover_real_skill_and_chance_on_noise():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 3))
    y_signal = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    y_noise = rng.integers(0, 2, size=n)
    t = pd.Series(np.arange(n))
    cv = CombinatorialPurgedKFold(6, 2, t, t, embargo_pct=0.0)
    model = make_sign_classifier("logistic")
    acc_sig = path_accuracies(cpcv_oos_predictions(model, X, y_signal, cv), y_signal)
    acc_noise = path_accuracies(cpcv_oos_predictions(model, X, y_noise, cv), y_noise)
    assert len(acc_sig) == cv.n_paths == 5
    assert acc_sig.min() > 0.75
    assert abs(acc_noise.mean() - 0.5) < 0.06


def test_predict_proba_paths_are_probabilities():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    cv = CombinatorialPurgedKFold(5, 2)
    paths = cpcv_oos_predictions(make_sign_classifier("logistic"), X, y, cv, method="predict_proba")
    assert paths.n_paths == 4
    assert ((paths.predictions >= 0) & (paths.predictions <= 1)).all()
