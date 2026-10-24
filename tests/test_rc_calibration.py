"""Reality-check calibration: the script runs offline and the committed artifact is consistent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("run_rc_calibration", ROOT / "scripts" / "run_rc_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_builds_a_small_report_offline_and_serially():
    mod = _mod()
    args = mod.parse_args(["--n-seeds", "2", "--n-bars", "600", "--phis", "0.0", "--cost-bps", "2",
                           "--innovations", "garch", "--n-sims", "200", "--n-boot", "49", "--workers", "1",
                           "--no-write"])
    rep = mod.build(args)
    (row,) = rep["rows"]
    assert row["innovations"] == "garch" and row["n_seeds"] == 2 and "rc_p" in row
    assert "reality check p<0.05" in mod.summarise(rep)


def test_committed_artifact_covers_every_cell_and_reproduces_slice_4_on_normal_returns():
    rep = json.loads((ROOT / "artifacts" / "rc_calibration.json").read_text())
    assert (rep["n_seeds"], rep["n_bars"], rep["n_boot"], rep["mean_block"]) == (200, 2520, 499, 10.0)
    cells = {(r["innovations"], r["phi"], r["cost_bps"]): r for r in rep["rows"]}
    assert len(cells) == 8
    slice4 = json.loads((ROOT / "artifacts" / "dsr_calibration.json").read_text())
    for r in slice4["rows"]:
        key = ("normal", r["phi"], r["cost_bps"])
        if key in cells:
            for m in ("dsr_raw", "dsr_participation", "dsr_simulated", "max_p"):
                assert cells[key][m] == r[m], (key, m)
    # the bootstrap null is as well calibrated as the Gaussian one, with no power gain
    for innov in ("normal", "garch"):
        null = cells[(innov, 0.0, 0.0)]
        assert null["rc_p"] <= 0.05 + 2 * null["rc_p_se"]
        edge = cells[(innov, 0.1, 2.0)]
        assert edge["rc_p"] <= edge["max_p"]
