"""The committed slice 7 artifact agrees with the frozen protocol's rule."""

import json
from pathlib import Path

from purged_cv import embargo_calibration as P

ART = Path(__file__).resolve().parents[1] / "artifacts" / "embargo_calibration.json"


def _rec():
    return json.loads(ART.read_text(encoding="utf-8"))


def _cells():
    return {(c["shock_phi"], c["horizon"]): c["arms"] for c in _rec()["cells"]}


def test_full_grid_and_seeds():
    r = _rec()
    assert r["n_seeds"] == 100 and r["seed_start"] == 7000 and r["n_samples"] == 1500
    assert set(_cells()) == {(p, h) for p in P.SHOCK_PHIS for h in P.HORIZONS}


def test_no_arm_leaks_anywhere_not_even_purge_only():
    for key, arms in _cells().items():
        for name, d in arms.items():
            assert d["leak_free"], (key, name)
            assert abs(d["optimism"]) < 0.006, (key, name)


def test_call_reproduces_and_is_no_because_the_default_never_leaks():
    rec = _rec()
    assert P.call(_cells()) == rec["call"]
    assert rec["call"]["recommend"] is False
    assert rec["call"]["default_fails_in_persistent_cells"] == 0
    assert rec["call"]["candidate_leak_free_everywhere"] is True


def test_acf_embargo_grows_with_persistence_at_short_horizon_and_costs_little():
    c = _cells()
    med = [c[(p, 5)]["acf_after"]["median_embargo"] for p in P.SHOCK_PHIS]
    assert med == sorted(med) and med[0] == 0 and med[-1] >= 15
    assert min(a["acf_both"]["keep_vs_default"] for a in c.values()) > 0.98
    assert all(abs(a["pct5_after"]["keep_vs_default"] - 0.958) < 0.002 for a in c.values())
