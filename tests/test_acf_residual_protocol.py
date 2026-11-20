"""Slice 13 protocol is frozen: grid, seeds, arms and call logic."""

import pytest

from purged_cv import acf_residual as P


def test_frozen_grid_and_seeds():
    assert (P.N_SAMPLES, P.N_SPLITS, P.HORIZON, P.FEATURE_PHI, P.SHOCK_PHI) == (500, 20, 20, 0.98, 0.9)
    assert P.SEEDS[0] == 10400 and len(P.SEEDS) == 480
    assert not set(P.SMOKE_SEEDS) & set(P.SEEDS)
    assert P.SEEDS[0] > 10379  # disjoint from slice 12's seeds 9900-10379
    assert P.ARMS["acf_both"] == {"embargo_bars": "acf", "embargo_side": "both"}
    assert (P.DEFAULT_ARM, P.CANDIDATE, P.SE_MULT, P.MIN_ROW_SHARE) == ("pct1_after", "acf_both", 2.0, 0.90)


def _summary(ctrl, dflt, cand, diff, rows=(9000.0, 8500.0)):
    arm = lambda m, s: {"optimism": m, "se": s, "leak": m > 2 * s, "mean_train_rows": rows[0]}
    a = {"purge_only": arm(*dflt), "pct1_after": arm(*dflt), "acf_both": arm(*cand), "contiguous_no_purge": arm(*ctrl)}
    a["acf_both"]["mean_train_rows"] = rows[1]
    return {"arms": a, "paired_diff": {"mean": diff[0], "se": diff[1]}}


def test_call_removes_when_leak_free_and_paired_drop():
    c = P.call(_summary((0.1, 0.004), (0.011, 0.004), (0.001, 0.004), (0.010, 0.003)))
    assert c["reproduced"] and c["removes_residual"] and c["within_budget"]
    assert c["row_share"] == pytest.approx(8500 / 9000)


def test_call_fails_on_power():
    c = P.call(_summary((0.1, 0.004), (0.011, 0.004), (0.005, 0.004), (0.006, 0.004)))
    assert c["candidate_leak_free"] and not c["paired_reduction_significant"] and not c["removes_residual"]


def test_no_call_without_reproduction():
    c = P.call(_summary((0.1, 0.004), (0.004, 0.004), (0.0, 0.004), (0.004, 0.001)))
    assert not c["reproduced"] and not c["removes_residual"]


def test_over_budget_is_flagged():
    c = P.call(_summary((0.1, 0.004), (0.011, 0.004), (0.0, 0.004), (0.011, 0.003), rows=(9000.0, 7000.0)))
    assert c["removes_residual"] and not c["within_budget"]


def test_one_smoke_seed_runs():
    r = P.one_seed(P.SMOKE_SEEDS[0])
    assert set(r["accuracy"]) == {*P.ARMS, P.CONTROL}
    assert r["train_rows"]["acf_both"] <= r["train_rows"]["pct1_after"] <= r["train_rows"]["purge_only"]
