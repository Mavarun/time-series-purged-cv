"""Post-hoc pool of the purged k-NN default cell across slices 10, 12, 13."""

import math

import pytest

from purged_cv.residual_pool import cell_estimates, pool


def test_pool_matches_hand_computation():
    e = {"slice10": {"optimism": 0.0, "se": 1.0, "n_seeds": 1}, "slice12": {"optimism": 2.0, "se": 1.0, "n_seeds": 1},
         "slice13": {"optimism": 1.0, "se": 1.0, "n_seeds": 1}}
    p = pool(e)
    assert p["mean"] == pytest.approx(1.0) and p["se"] == pytest.approx(1 / math.sqrt(3))
    assert p["q"] == pytest.approx(2.0) and p["df"] == 2 and p["q_p"] == pytest.approx(math.exp(-1.0))
    assert p["z12_vs13"] == pytest.approx(1 / math.sqrt(2))


def test_committed_artifacts_pool():
    e = cell_estimates()
    assert [v["n_seeds"] for v in e.values()] == [240, 480, 480]
    p = pool(e)
    assert p["n_seeds"] == 1200
    assert 2.0 < p["z"] < 2.1          # borderline: just over the 2-s.e. line
    assert p["q_p"] > 0.10             # no significant heterogeneity across the three runs
    assert 1.9 < p["z12_vs13"] < 2.0   # slices 12 and 13 disagree by just under 2 s.e.
