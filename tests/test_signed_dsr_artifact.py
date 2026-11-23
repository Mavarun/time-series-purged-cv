import json
from pathlib import Path

from purged_cv import signed_dsr_calibration as P

A = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "signed_dsr_calibration.json").read_text())
CELLS = {(c["n"], c["returns"], c["phi"]): c for c in A["cells"]}


def test_artifact_shape_and_seeds():
    assert A["n_seeds"] == 300 and A["seed_start"] == 11000 and len(CELLS) == 18
    assert all(c["n_seeds"] == 300 for c in CELLS.values())


def test_calls_recompute():
    assert P.call(CELLS) == A["call"]
    assert A["call"] == {"H1": True, "H2": False, "power_cells_ok": 1, "recommend_signed": False}


def test_reading():
    nulls = [c for k, c in CELLS.items() if k[2] == 0.0]
    power = [c for k, c in CELLS.items() if k[2] == 0.1]
    # the signed count roughly doubles N and brings every null cell to <= 2.7%
    assert all(c["dsr_signed"] <= 0.027 for c in nulls)
    assert all(1.9 < c["mean_n_eff_signed"] / c["mean_n_eff_participation"] < 2.01 for c in CELLS.values())
    # the plain participation DSR reproduces slice 6's liberal short-sample nulls
    assert CELLS[(252, "garch", 0.0)]["dsr_participation"] > 0.09 and CELLS[(252, "normal", 0.0)]["dsr_participation"] > 0.07
    # the signed DSR sits between the simulated DSR and max-stat on power in every cell
    assert all(c["dsr_simulated"] < c["dsr_signed"] < c["max_p"] for c in power)


def test_one_cell_reproduces():
    rows = [P.one_seed(252, "normal", 0.0, s) for s in P.SEEDS[:5]]
    assert all(r["seed"] == s for r, s in zip(rows, P.SEEDS[:5]))
