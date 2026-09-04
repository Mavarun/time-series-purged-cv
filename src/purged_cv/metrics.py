"""Compare naive KFold scores vs purged/embargoed CV on the same data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold
from purged_cv.weights import sample_weights_from_returns


@dataclass
class CVComparisonResult:
    naive_scores: list[float]
    purged_scores: list[float]
    naive_mean: float
    purged_mean: float
    naive_std: float
    purged_std: float
    gap_naive_minus_purged: float
    n_splits: int
    model_kind: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fold_scores(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    sample_weight: np.ndarray | None = None,
) -> list[float]:
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        m = clone(model)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight[train_idx]
        m.fit(X[train_idx], y[train_idx], **fit_kwargs)
        pred = m.predict(X[test_idx])
        scores.append(float((pred == y[test_idx]).mean()))
    return scores


def compare_naive_vs_purged_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_start_times: pd.Series,
    label_end_times: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    model_kind: str = "logistic",
    shuffle_naive: bool = True,
    random_state: int = 42,
    forward_returns: np.ndarray | None = None,
    use_sample_weights: bool = False,
) -> CVComparisonResult:
    """Run shuffled (or ordered) KFold vs PurgedKFold; report accuracy means.

    Hypothesis check: naive_mean is expected to be >= purged_mean when labels
    overlap and/or leakage is present. This is a research diagnostic - not a
    claim of production edge or live PnL.
    """
    model = make_sign_classifier(kind=model_kind)
    naive_cv = KFold(
        n_splits=n_splits,
        shuffle=shuffle_naive,
        random_state=random_state if shuffle_naive else None,
    )
    purged_cv = PurgedKFold(
        n_splits=n_splits,
        label_start_times=label_start_times,
        label_end_times=label_end_times,
        embargo_pct=embargo_pct,
    )

    sw = None
    notes: list[str] = []
    if use_sample_weights:
        if forward_returns is None:
            raise ValueError("forward_returns required when use_sample_weights=True")
        sw = sample_weights_from_returns(
            forward_returns, label_start_times, label_end_times
        )
        notes.append("sample weights = |fwd_return| * average_uniqueness")

    naive_scores = _fold_scores(model, X, y, naive_cv, sample_weight=sw)
    purged_scores = _fold_scores(model, X, y, purged_cv, sample_weight=sw)

    naive_mean = float(np.mean(naive_scores)) if naive_scores else float("nan")
    purged_mean = float(np.mean(purged_scores)) if purged_scores else float("nan")
    naive_std = float(np.std(naive_scores, ddof=0)) if naive_scores else float("nan")
    purged_std = float(np.std(purged_scores, ddof=0)) if purged_scores else float("nan")

    notes.append(
        "Naive CV uses sklearn KFold"
        + (" (shuffled)" if shuffle_naive else " (ordered)")
        + "; purged CV uses contiguous folds + purge/embargo."
    )
    notes.append(
        "Scores are fold accuracies for a scaled linear return-sign classifier. "
        "Not production accuracy; not live PnL."
    )

    return CVComparisonResult(
        naive_scores=naive_scores,
        purged_scores=purged_scores,
        naive_mean=naive_mean,
        purged_mean=purged_mean,
        naive_std=naive_std,
        purged_std=purged_std,
        gap_naive_minus_purged=naive_mean - purged_mean,
        n_splits=n_splits,
        model_kind=model_kind,
        notes=notes,
    )


def quick_naive_cv_score(X, y, n_splits=5, model_kind="logistic", random_state=42) -> float:
    """Convenience wrapper around cross_val_score for smoke checks."""
    model = make_sign_classifier(kind=model_kind)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean())
