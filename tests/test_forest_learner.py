"""The random-forest learner added for slice 9."""

import numpy as np
import pytest

from purged_cv.model import make_sign_classifier


def test_forest_is_seeded_and_fits_a_nonlinear_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like: no linear model can fit it
    a = make_sign_classifier("forest").fit(X[:400], y[:400]).predict(X[400:])
    b = make_sign_classifier("rf").fit(X[:400], y[:400]).predict(X[400:])
    assert np.array_equal(a, b)
    assert (a == y[400:]).mean() > 0.85
    lin = make_sign_classifier("logistic").fit(X[:400], y[:400]).predict(X[400:])
    assert (lin == y[400:]).mean() < 0.65


def test_unknown_kind_lists_forest():
    with pytest.raises(ValueError, match="forest"):
        make_sign_classifier("svm")
