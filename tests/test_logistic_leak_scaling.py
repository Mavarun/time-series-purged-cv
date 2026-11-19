"""leak_scaling: the post-hoc power law of the logistic control leak between n 500 and n 1500."""

import json
from pathlib import Path

import numpy as np

from purged_cv.logistic_n1500 import CONTROL, leak_scaling

ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def _cells():
    a = json.loads((ROOT / "logistic_n1500.json").read_text())
    b = json.loads((ROOT / "logistic_power.json").read_text())
    return ({c["shock_phi"]: c["arms"] for c in a["cells"]},
            {c["shock_phi"]: c["arms"] for c in b["cells"] if c["learner"] == "logistic"})


def test_exact_one_over_n_gives_exponent_one():
    arm = lambda o: {CONTROL: {"optimism": o, "se": 0.1 * o}}  # noqa: E731
    out = leak_scaling({0.0: arm(0.01), 0.9: arm(0.02)}, {0.0: arm(0.03), 0.9: arm(0.06)})
    assert all(abs(v["exponent"] - 1.0) < 1e-12 for v in out.values())
    assert abs(out[0.0]["se"] - np.sqrt(0.02) / np.log(3)) < 1e-12


def test_committed_artifacts_are_consistent_with_one_over_n_at_two_se():
    out = leak_scaling(*_cells())
    for v in out.values():
        assert v["exponent"] > 1.0 and abs(v["exponent"] - 1.0) < 2.0 * v["se"]
