"""The committed slice 9 artifact agrees with the frozen protocol, and the script runs."""

import importlib.util
import json
from pathlib import Path

from purged_cv import learner_calibration as P

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "learner_calibration.json"


def _rec():
    return json.loads(ART.read_text(encoding="utf-8"))


def _cells():
    return {(c["learner"], c["n_splits"], c["shock_phi"]): c["arms"] for c in _rec()["cells"]}


def test_full_grid_and_seeds():
    r = _rec()
    assert r["n_seeds"] == 60 and r["seed_start"] == 9000 and r["n"] == 1500
    assert set(_cells()) == {(m, k, p) for m in P.LEARNERS for k in P.N_SPLITS for p in P.SHOCK_PHIS}


def test_call_reproduces_and_widens_the_claim_on_knn_and_forest():
    rec = _rec()
    assert P.call(_cells()) == rec["call"]
    per = rec["call"]["per_learner"]
    assert not per["logistic"]["testable"] and per["logistic"]["control_leaks"] == 0
    for m in ("knn", "forest"):
        assert per[m]["testable"] and per[m]["default_holds"] and per[m]["default_leaks"] == 0
    assert rec["call"]["widen_claim"]


def test_both_flexible_learners_leak_only_at_20_folds_and_purge_removes_it():
    c = _cells()
    for m in ("knn", "forest"):
        for p in P.SHOCK_PHIS:
            assert c[(m, 20, p)]["contiguous_no_purge"]["leak"] and not c[(m, 5, p)]["contiguous_no_purge"]["leak"]
            assert not c[(m, 20, p)]["purge_only"]["leak"] and not c[(m, 20, p)]["pct1_after"]["leak"]


def test_script_smoke_run(tmp_path):
    spec = importlib.util.spec_from_file_location("lc", ROOT / "scripts" / "run_learner_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--n-seeds", "2", "--out", str(tmp_path), "--workers", "2"]) == 0
    assert (tmp_path / "learner_calibration.md").read_text(encoding="utf-8").startswith("# Learner calibration")
