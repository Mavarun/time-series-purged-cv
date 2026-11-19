"""The committed slice 8 artifact agrees with the frozen protocol, and the script runs."""

import importlib.util
import json
from pathlib import Path

from purged_cv import fold_count_calibration as P

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "fold_count_calibration.json"


def _rec():
    return json.loads(ART.read_text(encoding="utf-8"))


def _cells():
    return {(c["n"], c["n_splits"], c["shock_phi"]): c["arms"] for c in _rec()["cells"]}


def test_full_grid_and_seeds():
    r = _rec()
    assert r["n_seeds"] == 100 and r["seed_start"] == 8000 and r["horizon"] == 20
    assert set(_cells()) == {(n, k, p) for n in P.N_SAMPLES for k in P.N_SPLITS for p in P.SHOCK_PHIS}


def test_call_reproduces_default_holds_and_candidate_not_recommended():
    rec = _rec()
    assert P.call(_cells()) == rec["call"]
    c = rec["call"]
    assert c["testable"] and c["control_leaks"] == 7
    assert c["default_holds"] and c["default_fails_many_folds"] == 0
    assert c["recommend_candidate"] is False


def test_control_leak_grows_with_fold_count_and_purge_removes_it_everywhere():
    c = _cells()
    for n in P.N_SAMPLES:
        for p in P.SHOCK_PHIS:
            ctrl = [c[(n, k, p)]["contiguous_no_purge"]["optimism"] for k in P.N_SPLITS]
            assert ctrl == sorted(ctrl), (n, p)
    assert c[(500, 20, 0.9)]["contiguous_no_purge"]["optimism"] > 0.09
    for key, arms in c.items():
        for name in P.ARMS:
            assert not arms[name]["leak"], (key, name)


def test_script_smoke_one_seed(tmp_path):
    spec = importlib.util.spec_from_file_location("fc", ROOT / "scripts" / "run_fold_count_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--n-seeds", "1", "--out", str(tmp_path), "--workers", "2"]) == 0
    out = json.loads((tmp_path / "fold_count_calibration.json").read_text())
    assert out["n_seeds"] == 1 and len(out["cells"]) == 12
