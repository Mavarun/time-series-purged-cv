"""Short-sample calibration script: runs offline and serially; committed artifact is consistent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("run_short", ROOT / "scripts" / "run_short_sample_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_builds_a_small_report_offline_and_serially():
    mod = _mod()
    args = mod.parse_args(["--n-seeds", "2", "--n-bars", "300", "--phis", "0.0", "--innovations", "regime",
                           "--n-sims", "100", "--n-boot", "29", "--workers", "1", "--no-write"])
    rep = mod.build(args)
    (row,) = rep["rows"]
    assert (row["innovations"], row["n_bars"], row["n_seeds"]) == ("regime", 300, 2)
    assert "reality check p<0.05" in mod.summarise(rep)


@pytest.fixture(scope="module")
def artifact():
    return json.loads((ROOT / "artifacts" / "short_sample_calibration.json").read_text())


def test_committed_artifact_covers_every_cell(artifact):
    assert (artifact["n_seeds"], artifact["seed_start"], artifact["n_boot"]) == (300, 5000, 499)
    cells = {(r["innovations"], r["n_bars"], r["phi"]) for r in artifact["rows"]}
    assert len(cells) == 24 == len(artifact["rows"])


def test_artifact_seeds_are_disjoint_from_slices_4_and_5(artifact):
    assert artifact["seed_start"] >= 200


def test_max_stat_and_reality_check_do_not_inflate_the_null_at_any_n(artifact):
    # nominal 5%, 300 seeds: a rate above 0.09 would be ~3 s.e. over
    for r in artifact["rows"]:
        if r["phi"] == 0.0:
            assert r["max_p"] < 0.09 and r["rc_p"] < 0.09, r


def test_power_rises_with_n_and_is_near_the_null_at_one_year(artifact):
    for innov in ("normal", "garch", "regime"):
        p = {r["n_bars"]: r["max_p"] for r in artifact["rows"] if r["innovations"] == innov and r["phi"] == 0.1}
        assert p[252] < p[504] < p[1260] < p[2520]
        assert p[252] < 0.15  # one year of daily bars cannot find a phi = 0.1 edge in a 40-rule grid


def test_participation_dsr_is_the_liberal_hurdle_at_one_year(artifact):
    rows = [r for r in artifact["rows"] if r["phi"] == 0.0 and r["n_bars"] == 252]
    for r in rows:
        assert r["dsr_participation"] > r["max_p"]
        assert r["dsr_raw"] < 0.02  # raw N=40 is very conservative
