"""Slice 12 protocol: frozen grid, fresh seeds, calls on hand-made cells."""

from purged_cv import knn_residual as P, learner_calibration, logistic_n1500, logistic_power


def test_grid_and_fresh_seeds():
    assert P.SEEDS == tuple(range(9900, 10380)) and P.SHOCK_PHIS == (0.0, 0.9)
    used = set(logistic_power.SEEDS) | set(learner_calibration.SEEDS) | set(logistic_n1500.SEEDS)
    assert not used & set(P.SEEDS)


def _cell(ctrl, dflt, se=0.003):
    arm = lambda o: {"optimism": o, "se": se, "leak": o > 2 * se}  # noqa: E731
    return {P.CONTROL: arm(ctrl), P.DEFAULT_ARM: arm(dflt), "purge_only": arm(dflt)}


def test_calls():
    c = P.call({0.0: _cell(0.06, 0.002), 0.9: _cell(0.1, 0.008)})
    assert c["reproduced"] and c["residual_real"] and c["default_leaks"] == 1
    c = P.call({0.0: _cell(0.06, 0.002), 0.9: _cell(0.1, 0.004)})
    assert not c["residual_real"] and abs(c["default_upper_bound"] - 0.010) < 1e-12
    assert not P.call({0.0: _cell(0.001, 0.009), 0.9: _cell(0.1, 0.009)})["residual_real"]
