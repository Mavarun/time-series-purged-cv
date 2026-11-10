"""Smoke test of scripts/run_embargo_calibration.py on one seed per cell."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_script_writes_both_artifacts(tmp_path):
    spec = importlib.util.spec_from_file_location("emb", ROOT / "scripts" / "run_embargo_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--n-seeds", "1", "--out", str(tmp_path), "--workers", "2"]) == 0
    out = json.loads((tmp_path / "embargo_calibration.json").read_text())
    assert out["n_seeds"] == 1 and len(out["cells"]) == 8
    assert "Recommend `acf_both`" in (tmp_path / "embargo_calibration.md").read_text()
