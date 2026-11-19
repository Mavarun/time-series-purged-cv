"""Slice 8 frozen protocol: grid, seeds and the pre-stated calls."""

from purged_cv import embargo_calibration as S7
from purged_cv import fold_count_calibration as P


def test_protocol_constants_are_frozen_and_seeds_disjoint_from_slice7():
    assert P.N_SAMPLES == (500, 1500) and P.N_SPLITS == (5, 10, 20) and P.SHOCK_PHIS == (0.0, 0.9)
    assert P.HORIZON == 20 and P.MANY_FOLDS == 10 and P.MIN_KEEP_SHARE == 0.90
    assert not set(P.SEEDS) & set(S7.SEEDS) and len(P.SEEDS) == 100
    assert P.ARMS["pct1_after"] == S7.ARMS["pct1_after"] and P.ARMS["acf_both"] == S7.ARMS["acf_both"]


def test_one_seed_returns_every_arm_and_control_with_fewer_rows_under_purge():
    r = P.one_seed(500, 10, 0.9, 8000)
    assert set(r["accuracy"]) == {"purge_only", "pct1_after", "acf_both", "contiguous_no_purge"}
    assert r["train_rows"]["contiguous_no_purge"] > r["train_rows"]["purge_only"] >= r["train_rows"]["pct1_after"]


def _cell(def_leak, cand_leak, ctrl_leak, keep=1.0):
    mk = lambda leak, k=1.0: {"optimism": 0.0, "se": 0.0, "leak": leak, "keep_vs_default": k}
    return {"purge_only": mk(False), "pct1_after": mk(def_leak), "acf_both": mk(cand_leak, keep),
            "contiguous_no_purge": mk(ctrl_leak)}


def _cells(**over):
    keys = [(n, k, p) for n in P.N_SAMPLES for k in P.N_SPLITS for p in P.SHOCK_PHIS]
    return {key: over.get(str(key), _cell(False, False, True)) for key in keys}


def test_call_default_holds_when_nothing_leaks_and_control_sees_leaks():
    c = P.call(_cells())
    assert c["testable"] and c["default_holds"] and not c["recommend_candidate"]


def test_call_recommends_candidate_only_with_two_default_failures_and_clean_candidate():
    over = {str((500, 20, 0.9)): _cell(True, False, True), str((500, 10, 0.9)): _cell(True, False, True)}
    assert P.call(_cells(**over))["recommend_candidate"]
    over[str((500, 10, 0.9))] = _cell(True, True, True)
    assert not P.call(_cells(**over))["recommend_candidate"]


def test_call_is_not_testable_without_a_visible_control_leak():
    keys = [(n, k, p) for n in P.N_SAMPLES for k in P.N_SPLITS for p in P.SHOCK_PHIS]
    c = P.call({key: _cell(False, False, False) for key in keys})
    assert not c["testable"] and not c["default_holds"]
