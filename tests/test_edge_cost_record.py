"""The committed slice 8b artifact agrees with the frozen protocol, and the script runs."""

import importlib.util
import json
from pathlib import Path

from purged_cv import edge_cost as P

ROOT = Path(__file__).resolve().parents[1]


def _rec():
    return json.loads((ROOT / "artifacts" / "edge_cost.json").read_text(encoding="utf-8"))


def _cells():
    return {(c["n_splits"], c["shock_phi"]): c for c in _rec()["cells"]}


def test_grid_and_call_reproduce():
    r = _rec()
    assert r["n_seeds"] == 100 and r["seed_start"] == 8100 and set(_cells()) == {(k, p) for k in P.FOLDS for p in P.SHOCK_PHIS}
    assert P.call(_cells()) == r["call"]
    assert r["call"]["cheap"] is False and abs(r["call"]["max_cost"] - 0.0111) < 5e-5


def test_default_is_cheap_everywhere_and_the_two_sided_acf_embargo_is_not():
    c = _cells()
    assert max(x["pct1_after"]["cost_vs_purge_only"] for x in c.values()) < 0.003
    worst = c[(20, 0.9)]["acf_both"]
    assert worst["cost_vs_purge_only"] > 3 * worst["cost_se"]


def test_persistent_shocks_swamp_the_edge():
    c = _cells()
    assert c[(5, 0.0)]["truth"] > 0.55 and c[(5, 0.9)]["truth"] < 0.51


def test_script_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location("ec", ROOT / "scripts" / "run_edge_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--n-seeds", "1", "--out", str(tmp_path), "--workers", "2"]) == 0
    assert len(json.loads((tmp_path / "edge_cost.json").read_text())["cells"]) == 4
