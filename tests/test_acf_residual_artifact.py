"""The committed slice 13 artifact: frozen grid, calls recompute, seeds reproduce, same pipeline as slice 12."""

import json
from pathlib import Path

import pytest

from purged_cv import acf_residual as P
from purged_cv import knn_residual

ROOT = Path(__file__).resolve().parents[1]
ART = json.loads((ROOT / "artifacts" / "acf_residual.json").read_text())


def test_artifact_uses_the_frozen_grid():
    assert ART["n_seeds"] == 480 and ART["seed_start"] == 10400 and len(ART["per_seed"]) == 480


def test_summary_and_calls_recompute_from_per_seed():
    s = P.summarize(ART["per_seed"])
    assert s == ART["summary"]
    c = P.call(s)
    assert c == ART["call"]
    # slice 12's residual does not reappear, so no candidate call is made
    assert not c["reproduced"] and not c["removes_residual"]
    assert c["candidate_leak_free"] and c["paired_reduction_significant"] and c["within_budget"]


def test_acf_both_is_pessimistic_not_neutral():
    x = ART["summary"]["arms"]["acf_both"]
    assert x["optimism"] < -2 * x["se"]


@pytest.mark.parametrize("k", [0, 479])
def test_stored_seeds_reproduce(k):
    r = P.one_seed(P.SEEDS[k])
    for name, v in r["accuracy"].items():
        assert v == pytest.approx(ART["per_seed"][k]["accuracy"][name], abs=1e-12)
    assert r["train_rows"] == ART["per_seed"][k]["train_rows"]


def test_same_pipeline_as_slice_12():
    """Non-reproduction is not a code difference: on a slice 12 seed both modules give the same scores."""
    seed = knn_residual.SEEDS[3]
    a = P.one_seed(seed)["accuracy"]
    b = knn_residual.run_job((0.9, seed))[1]
    for name in ("purge_only", "pct1_after", "contiguous_no_purge"):
        assert a[name] == pytest.approx(b[name], abs=1e-12)
