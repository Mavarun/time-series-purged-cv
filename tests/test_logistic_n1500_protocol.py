"""Slice 11 protocol: frozen grid, fresh seeds, and the calls on hand-made cells."""

from purged_cv import learner_calibration, logistic_n1500 as P, logistic_power


def test_grid_is_frozen_and_seeds_are_fresh():
    assert P.N_SAMPLES == 1500 and P.SHOCK_PHIS == (0.0, 0.9) and P.SEEDS == tuple(range(9400, 9880))
    assert not set(P.SEEDS) & (set(logistic_power.SEEDS) | set(learner_calibration.SEEDS))
    assert P.SLICE10_CONTROL == {0.0: 0.0279, 0.9: 0.0541}


def _cell(ctrl, ctrl_se, dflt, dflt_se=0.004):
    def arm(o, se):
        return {"optimism": o, "se": se, "leak": o > 2 * se}
    return {P.CONTROL: arm(ctrl, ctrl_se), P.DEFAULT_ARM: arm(dflt, dflt_se), "purge_only": arm(dflt, dflt_se)}


def test_calls():
    c = P.call({0.0: _cell(0.005, 0.004, 0.0), 0.9: _cell(0.015, 0.004, 0.001)})
    assert c["measured"] and c["default_holds"] and c["shrinks_with_n"] and c["control_leaks"] == 1
    c = P.call({0.0: _cell(0.005, 0.004, 0.0), 0.9: _cell(0.006, 0.004, 0.0)})
    assert not c["measured"] and not c["default_holds"] and abs(c["control_upper_bound"] - 0.014) < 1e-12
    c = P.call({0.0: _cell(0.03, 0.004, 0.0), 0.9: _cell(0.015, 0.004, 0.009)})
    assert c["measured"] and not c["default_holds"] and not c["shrinks_with_n"]


def test_one_seed_runs_on_a_dev_seed():
    r = P.one_seed(0.0, 1, n=300)
    assert set(r) == {"purge_only", "pct1_after", P.CONTROL}
