"""The committed slice 11 artifact: frozen grid, calls recompute from per-seed scores, seeds reproduce."""

import json
from pathlib import Path

import pytest

from purged_cv import logistic_n1500 as P

ART = json.loads((Path(__file__).resolve().parents[1] / "artifacts" / "logistic_n1500.json").read_text())


def test_artifact_uses_the_frozen_grid():
    assert ART["n_seeds"] == 480 and ART["seed_start"] == 9400 and ART["n"] == 1500
    assert [c["shock_phi"] for c in ART["cells"]] == list(P.SHOCK_PHIS)
    assert all(len(ART["per_seed"][str(phi)]) == 480 for phi in P.SHOCK_PHIS)


def test_cells_and_calls_recompute_from_per_seed_scores():
    cells = {phi: P.summarize(ART["per_seed"][str(phi)]) for phi in P.SHOCK_PHIS}
    assert cells == {c["shock_phi"]: c["arms"] for c in ART["cells"]}
    c = P.call(cells)
    assert c == ART["call"]
    assert c["measured"] and c["default_holds"] and c["shrinks_with_n"]
    assert (c["control_leaks"], c["default_leaks"]) == (2, 0)


@pytest.mark.parametrize("k", [0, 241, 479])
def test_stored_seeds_reproduce(k):
    phi = 0.9
    assert P.one_seed(phi, P.SEEDS[k]) == pytest.approx(ART["per_seed"][str(phi)][k], abs=1e-12)
