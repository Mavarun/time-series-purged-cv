"""The committed slice 10 artifact matches the frozen protocol's calls, and a small run reproduces its direction."""

import json
from pathlib import Path

from purged_cv import logistic_power as P

ART = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "logistic_power.json").read_text())


def _cells():
    return {(c["learner"], c["shock_phi"]): c["arms"] for c in ART["cells"]}


def test_artifact_uses_the_frozen_grid():
    assert ART["n_seeds"] == 240 and ART["seed_start"] == 9100
    assert (ART["n"], ART["n_splits"], ART["horizon"]) == (500, 20, 20)
    assert set(_cells()) == {(m, phi) for m in P.LEARNERS for phi in P.SHOCK_PHIS}


def test_calls_recompute_from_the_cells():
    assert P.call(_cells()) == ART["call"]
    c = ART["call"]
    assert c["reproduced"] and c["logistic_testable"] and c["logistic_default_holds"]
    assert c["logistic_control_leaks"] == 2 and c["logistic_default_leaks"] == 0


def test_logistic_leaks_less_than_knn_without_purge():
    cells = _cells()
    for phi in P.SHOCK_PHIS:
        lg, kn = cells[("logistic", phi)][P.CONTROL], cells[("knn", phi)][P.CONTROL]
        assert 0.02 < lg["optimism"] < kn["optimism"]


def test_small_run_direction():
    # 6 seeds: the logistic control is already above both purged arms on average
    rs = [P.one_seed("logistic", 0.9, s) for s in P.SEEDS[:6]]
    ctrl = sum(r[P.CONTROL] for r in rs) / 6
    assert ctrl > sum(r["purge_only"] for r in rs) / 6 and ctrl > sum(r["pct1_after"] for r in rs) / 6
