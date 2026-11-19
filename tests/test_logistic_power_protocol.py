"""Slice 10 protocol (frozen 2026-11-20): grid, seeds and the pre-stated calls."""

from purged_cv import learner_calibration as S9
from purged_cv import logistic_power as P


def _cell(ctrl, default, se=0.005):
    arm = lambda o: {"optimism": o, "se": se, "leak": o > 2 * se}  # noqa: E731
    return {P.CONTROL: arm(ctrl), "purge_only": arm(0.0), P.DEFAULT_ARM: arm(default)}


def test_seeds_are_fresh_and_grid_is_as_written():
    assert len(P.SEEDS) == 240 and not set(P.SEEDS) & set(S9.SEEDS)
    assert (P.N_SAMPLES, P.N_SPLITS, P.HORIZON) == (500, 20, 20) and P.LEARNERS == ("logistic", "knn")


def test_calls():
    good_knn = {("knn", 0.0): _cell(0.06, 0.0), ("knn", 0.9): _cell(0.10, 0.0)}
    c = P.call({**good_knn, ("logistic", 0.0): _cell(0.0, 0.0), ("logistic", 0.9): _cell(0.02, 0.0)})
    assert c["reproduced"] and c["logistic_testable"] and c["logistic_default_holds"]
    c = P.call({**good_knn, ("logistic", 0.0): _cell(0.0, 0.0), ("logistic", 0.9): _cell(0.005, 0.0)})
    assert not c["logistic_testable"] and abs(c["logistic_control_upper_bound"] - 0.015) < 1e-12
    c = P.call({**good_knn, ("logistic", 0.0): _cell(0.03, 0.02), ("logistic", 0.9): _cell(0.02, 0.0)})
    assert c["logistic_testable"] and not c["logistic_default_holds"]
    bad = {("knn", 0.0): _cell(0.0, 0.0), ("knn", 0.9): _cell(0.10, 0.0)}
    c = P.call({**bad, ("logistic", 0.0): _cell(0.03, 0.0), ("logistic", 0.9): _cell(0.02, 0.0)})
    assert not c["reproduced"] and not c["logistic_testable"]


def test_one_seed_runs_on_the_null():
    acc = P.one_seed("logistic", 0.0, 9100)
    assert set(acc) == {"purge_only", "pct1_after", P.CONTROL}
    assert all(0.3 < v < 0.7 for v in acc.values())
