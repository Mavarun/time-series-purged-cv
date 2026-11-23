from purged_cv import signed_dsr_calibration as P


def test_frozen_grid_and_fresh_seeds():
    assert P.SEEDS == tuple(range(11000, 11300)) and P.NS == (252, 504, 1260)
    assert P.RETURNS == ("normal", "garch", "regime") and P.PHIS == (0.0, 0.1)
    assert not set(P.SEEDS) & set(P.DEV_SEEDS)
    assert (P.NULL_CAP, P.POWER_SLACK, P.POWER_CELLS_NEEDED) == (0.063, 0.03, 6)


def _cells(null_signed, power_signed, max_p=0.2):
    cells = {}
    for n in P.NS:
        for r in P.RETURNS:
            cells[(n, r, 0.0)] = {"dsr_signed": null_signed, "max_p": 0.04}
            cells[(n, r, 0.1)] = {"dsr_signed": power_signed, "max_p": max_p}
    return cells


def test_calls_on_hand_made_cells():
    assert P.call(_cells(0.05, 0.19)) == {"H1": True, "H2": True, "power_cells_ok": 9, "recommend_signed": True}
    v = P.call(_cells(0.07, 0.19))
    assert not v["H1"] and v["H2"] and not v["recommend_signed"]
    v = P.call(_cells(0.01, 0.10))
    assert v["H1"] and not v["H2"] and v["power_cells_ok"] == 0


def test_one_seed_is_deterministic_and_signed_is_stricter():
    a, b = P.one_seed(252, "normal", 0.0, 1), P.one_seed(252, "normal", 0.0, 1)
    assert a == b
    assert a["n_eff_signed"] > a["n_eff_participation"] and a["dsr_signed"] <= a["dsr_participation"]
