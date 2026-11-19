"""The committed slice 12 artifact: frozen grid, calls recompute from per-seed scores, seeds reproduce."""

import json
from pathlib import Path

import pytest

from purged_cv import knn_residual as P

ART = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "knn_residual.json").read_text())


def test_artifact_uses_the_frozen_grid():
    assert ART["n_seeds"] == 480 and ART["seed_start"] == 9900 and ART["n"] == 500
    assert [c["shock_phi"] for c in ART["cells"]] == list(P.SHOCK_PHIS)
    assert all(len(ART["per_seed"][str(phi)]) == 480 for phi in P.SHOCK_PHIS)


def test_cells_and_calls_recompute_from_per_seed_scores():
    cells = {phi: P.summarize(ART["per_seed"][str(phi)]) for phi in P.SHOCK_PHIS}
    assert cells == {c["shock_phi"]: c["arms"] for c in ART["cells"]}
    c = P.call(cells)
    assert c == ART["call"]
    assert c["reproduced"] and c["residual_real"]
    assert (c["default_leaks"], c["purge_only_leaks"]) == (1, 1)


def test_the_leak_is_the_phi_09_cell_only():
    by_phi = {c["shock_phi"]: c["arms"] for c in ART["cells"]}
    assert by_phi[0.9][P.DEFAULT_ARM]["leak"] and not by_phi[0.0][P.DEFAULT_ARM]["leak"]


@pytest.mark.parametrize("k", [0, 241, 479])
def test_stored_seeds_reproduce(k):
    phi = 0.9
    assert P.run_job((phi, P.SEEDS[k]))[1] == pytest.approx(ART["per_seed"][str(phi)][k], abs=1e-12)
