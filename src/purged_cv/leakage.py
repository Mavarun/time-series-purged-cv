"""Leakage experiment: naive K-fold vs purged CV vs CPCV on a *null* problem.

Design (ground truth known by construction)
-------------------------------------------
* Bar shocks ``e_t`` are iid N(0, 1). Labels are overlapping horizon signs,
  ``y_i = 1{ e_{i+1} + ... + e_{i+h} > 0 }`` so neighbouring labels share
  ``h - |i - j|`` shocks and are strongly autocorrelated for large ``h``.
* Features are ``n_features`` independent AR(1) series with coefficient
  ``feature_phi`` (slow "regime" variables), generated **independently** of
  ``e``. They carry *zero* information about the labels: the true achievable
  accuracy is 0.5.
* A local learner (k-NN) under shuffled K-fold finds test rows' temporal
  neighbours in the training set, and those neighbours share most of the
  test label's shocks, so the CV score is inflated above 0.5. Purged K-fold
  and CPCV drop train rows whose label windows overlap the test rows (plus an
  embargo), removing the shared-shock channel.
* A *fresh holdout* (independent draw from the same process, model fitted on
  the full original sample) measures the realised out-of-sample accuracy.

Controls: with ``horizon=1`` (no label overlap) or ``feature_phi=0`` (no
feature persistence) naive K-fold should not be inflated - the leak needs
both overlapping labels and serially correlated features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.cpcv import CombinatorialPurgedKFold, cpcv_oos_predictions, path_accuracies
from purged_cv.data import LabeledDataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold


def make_autocorrelated_null_dataset(
    n_samples: int = 1500,
    horizon: int = 20,
    feature_phi: float = 0.98,
    n_features: int = 3,
    seed: int = 0,
) -> LabeledDataset:
    """Overlapping-horizon sign labels with features independent of labels."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not 0.0 <= feature_phi < 1.0:
        raise ValueError("feature_phi must be in [0, 1)")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n_samples + horizon + 1)
    csum = np.concatenate([[0.0], np.cumsum(shocks)])
    idx = np.arange(n_samples)
    fwd = csum[idx + 1 + horizon] - csum[idx + 1]  # e_{i+1} + ... + e_{i+h}
    y = (fwd > 0).astype(int)

    z = rng.normal(size=(n_samples, n_features))
    X = np.empty_like(z)
    X[0] = z[0]
    scale = np.sqrt(1.0 - feature_phi**2)
    for t in range(1, n_samples):
        X[t] = feature_phi * X[t - 1] + scale * z[t]

    starts = pd.Series(idx, name="label_start")
    ends = pd.Series(idx + horizon, name="label_end")
    return LabeledDataset(
        X=X,
        y=y,
        forward_returns=fwd,
        label_start_times=starts,
        label_end_times=ends,
        feature_names=[f"ar_feature_{k}" for k in range(n_features)],
        meta={
            "source": "autocorrelated_null",
            "horizon": horizon,
            "feature_phi": feature_phi,
            "seed": seed,
            "true_accuracy": 0.5,
            "citation": "Synthetic null: features independent of labels by construction.",
        },
    )


@dataclass
class LeakageReport:
    naive_kfold: float
    purged_kfold: float
    cpcv_path_mean: float
    cpcv_path_min: float
    cpcv_path_max: float
    fresh_holdout: float
    true_accuracy: float
    naive_inflation_vs_fresh: float
    n_samples: int
    horizon: int
    feature_phi: float
    model_kind: str

    def to_dict(self) -> dict:
        return asdict(self)


def _cv_accuracy(model, X, y, cv) -> float:
    scores = []
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        scores.append(float((m.predict(X[te]) == y[te]).mean()))
    return float(np.mean(scores))


def leakage_experiment(
    n_samples: int = 1500,
    horizon: int = 20,
    feature_phi: float = 0.98,
    n_features: int = 3,
    seed: int = 0,
    model_kind: str = "knn",
    n_splits: int = 5,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
) -> LeakageReport:
    """Score one null dataset four ways; see module docstring for the design."""
    ds = make_autocorrelated_null_dataset(n_samples, horizon, feature_phi, n_features, seed)
    fresh = make_autocorrelated_null_dataset(
        n_samples, horizon, feature_phi, n_features, seed + 10_000
    )
    model = make_sign_classifier(model_kind)
    X, y = ds.X, ds.y
    s, e = ds.label_start_times, ds.label_end_times

    naive = _cv_accuracy(model, X, y, KFold(n_splits, shuffle=True, random_state=seed))
    purged = _cv_accuracy(model, X, y, PurgedKFold(n_splits, s, e, embargo_pct))
    cpcv = CombinatorialPurgedKFold(n_groups, n_test_groups, s, e, embargo_pct)
    paths = path_accuracies(cpcv_oos_predictions(model, X, y, cpcv), y)
    full = clone(model).fit(X, y)
    fresh_acc = float((full.predict(fresh.X) == fresh.y).mean())

    return LeakageReport(
        naive_kfold=naive,
        purged_kfold=purged,
        cpcv_path_mean=float(paths.mean()),
        cpcv_path_min=float(paths.min()),
        cpcv_path_max=float(paths.max()),
        fresh_holdout=fresh_acc,
        true_accuracy=0.5,
        naive_inflation_vs_fresh=naive - fresh_acc,
        n_samples=n_samples,
        horizon=horizon,
        feature_phi=feature_phi,
        model_kind=model_kind,
    )


def leakage_sweep(seeds=(0, 1, 2), **kwargs) -> dict:
    """Average :func:`leakage_experiment` over seeds (reduces fold noise)."""
    reports = [leakage_experiment(seed=s, **kwargs) for s in seeds]
    keys = ["naive_kfold", "purged_kfold", "cpcv_path_mean", "fresh_holdout"]
    out = {k: float(np.mean([getattr(r, k) for r in reports])) for k in keys}
    out["naive_minus_purged"] = out["naive_kfold"] - out["purged_kfold"]
    out["n_seeds"] = len(reports)
    out["per_seed"] = [r.to_dict() for r in reports]
    return out
