"""Slice 9 frozen protocol: grid, seeds and the pre-stated per-learner calls."""

from purged_cv import embargo_calibration as S7
from purged_cv import fold_count_calibration as S8
from purged_cv import learner_calibration as P


def test_protocol_constants_are_frozen_and_seeds_fresh():
    assert P.LEARNERS == ("logistic", "knn", "forest") and P.N_SPLITS == (5, 20) and P.SHOCK_PHIS == (0.0, 0.9)
    assert P.N_SAMPLES == 1500 and P.HORIZON == 20 and len(P.SEEDS) == 60
    assert not set(P.SEEDS) & (set(S7.SEEDS) | set(S8.SEEDS))
    assert P.ARMS["pct1_after"] == S8.ARMS["pct1_after"] and P.ARMS["purge_only"] == S8.ARMS["purge_only"]
    assert P.MIN_CONTROL_LEAKS == 2 and P.MIN_TESTABLE_LEARNERS == 2


def test_one_seed_runs_every_learner():
    for learner in P.LEARNERS:
        r = P.one_seed(learner, 5, 0.0, 9000)
        assert set(r) == {"purge_only", "pct1_after", "contiguous_no_purge"}
        assert all(0.0 <= v <= 1.0 for v in r.values())


def _cell(default_leak, control_leak):
    mk = lambda leak: {"optimism": 0.0, "se": 0.0, "leak": leak}
    return {"purge_only": mk(False), "pct1_after": mk(default_leak), "contiguous_no_purge": mk(control_leak)}


def _cells(spec):
    """spec: learner -> (default leaks, control leaks) as counts over its 4 cells."""
    out = {}
    for learner in P.LEARNERS:
        d, c = spec[learner]
        keys = [(learner, k, p) for k in P.N_SPLITS for p in P.SHOCK_PHIS]
        for i, key in enumerate(keys):
            out[key] = _cell(i < d, i < c)
    return out


def test_claim_widens_only_with_two_clean_testable_learners():
    assert P.call(_cells({"logistic": (0, 0), "knn": (0, 3), "forest": (0, 2)}))["widen_claim"]
    c = P.call(_cells({"logistic": (0, 0), "knn": (0, 3), "forest": (0, 1)}))
    assert c["n_testable"] == 1 and not c["widen_claim"]
    c = P.call(_cells({"logistic": (0, 4), "knn": (0, 3), "forest": (1, 4)}))
    assert not c["per_learner"]["forest"]["default_holds"] and not c["widen_claim"]
    assert not P.call(_cells({"logistic": (0, 1), "knn": (0, 1), "forest": (0, 0)}))["per_learner"]["knn"]["default_holds"]
