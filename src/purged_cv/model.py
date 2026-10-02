"""Return-sign models: logistic, ridge, and a k-NN for leakage diagnostics.

The k-NN is deliberately included as a *flexible, local* learner: with
serially correlated features its nearest neighbours in feature space are
mostly its neighbours in time, which is exactly the setting where shuffled
K-fold leaks overlapping labels from train into test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedScores:
    accuracy: float
    predictions: np.ndarray
    probabilities: np.ndarray | None


def make_sign_classifier(
    kind: str = "logistic", C: float = 1.0, alpha: float = 1.0, n_neighbors: int = 5
) -> Pipeline:
    """Return a scaled classifier for binary return-sign prediction.

    ``kind`` is ``"logistic"``, ``"ridge"`` or ``"knn"``. k-NN does not accept
    sample weights.
    """
    kind = kind.lower()
    if kind == "logistic":
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs",
        )
    elif kind in {"ridge", "ridgeclassifier"}:
        clf = RidgeClassifier(alpha=alpha)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError("kind must be 'logistic', 'ridge' or 'knn'")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def fit_predict_fold(
    model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> FittedScores:
    """Fit on one fold and score accuracy on the test fold."""
    fit_params = {}
    if sample_weight is not None:
        fit_params["clf__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_params)
    pred = model.predict(X_test)
    acc = float((pred == y_test).mean()) if len(y_test) else float("nan")
    proba = None
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    return FittedScores(accuracy=acc, predictions=pred, probabilities=proba)
