"""Slice 7 post-hoc positive control: the measurement sees the shuffled-K-fold leak."""

import json
from pathlib import Path

from purged_cv import embargo_calibration as P

ART = Path(__file__).resolve().parents[1] / "artifacts" / "embargo_positive_control.json"


def _cells():
    rec = json.loads(ART.read_text(encoding="utf-8"))
    assert rec["n_seeds"] == 100 and rec["seed_start"] == 7000
    return {(c["shock_phi"], c["horizon"]): c["arms"] for c in rec["cells"]}


def test_shuffled_kfold_leaks_in_every_cell_and_more_with_persistence():
    c = _cells()
    for key, arms in c.items():
        assert arms["shuffled_kfold"]["leak"] and arms["shuffled_kfold"]["optimism"] > 0.10, key
    for h in P.HORIZONS:
        opt = [c[(p, h)]["shuffled_kfold"]["optimism"] for p in P.SHOCK_PHIS]
        assert opt == sorted(opt)


def test_contiguous_folds_without_purge_do_not_leak_measurably():
    for key, arms in _cells().items():
        assert not arms["contiguous_no_purge"]["leak"], key
        assert arms["contiguous_no_purge"]["optimism"] < 0.01, key


def test_control_seed_matches_one_seed_data():
    r = P.control_seed(0.9, 5, 7000, n_samples=400)
    assert set(r) == set(P.CONTROL_ARMS) and all(0 <= v <= 1 for v in r.values())
