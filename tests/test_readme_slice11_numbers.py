"""Every slice-11 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

from purged_cv.logistic_n1500 import leak_scaling

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SECTION = README[README.index("## Slice 11 (2026-11-20)"): README.index("## Slice 12 (2026-11-20)")]
A = json.loads((ROOT / "artifacts" / "logistic_n1500.json").read_text())
B = json.loads((ROOT / "artifacts" / "logistic_power.json").read_text())


def test_table_is_the_artifact():
    for line in (ROOT / "artifacts" / "logistic_n1500.md").read_text().splitlines():
        if line.startswith("|"):
            assert line in SECTION, line


def test_calls_and_prose():
    v = A["call"]
    assert v["measured"] and v["default_holds"] and f"leaks in {v['control_leaks']} of 2 cells" in SECTION
    cells = {c["shock_phi"]: c["arms"] for c in A["cells"]}
    for phi in (0.0, 0.9):
        x = cells[phi]["contiguous_no_purge"]
        assert f"{x['optimism']:+.4f} (s.e. {x['se']:.4f})" in SECTION
        for arm in ("purge_only", "pct1_after"):
            assert round(abs(cells[phi][arm]["optimism"]), 4) <= 0.0010  # README quotes 4 dp
    x = cells[0.0]["contiguous_no_purge"]
    assert f"at\n{x['optimism'] / x['se']:.1f} s.e." in SECTION


def test_scaling_numbers():
    n15 = {c["shock_phi"]: c["arms"] for c in A["cells"]}
    n5 = {c["shock_phi"]: c["arms"] for c in B["cells"] if c["learner"] == "logistic"}
    s = leak_scaling(n15, n5)
    assert f"falls {s[0.0]['ratio']:.1f}x (phi 0) and {s[0.9]['ratio']:.1f}x (phi 0.9)" in SECTION
    assert (f"{s[0.0]['exponent']:.2f} ± {s[0.0]['se']:.2f} and {s[0.9]['exponent']:.2f} ± {s[0.9]['se']:.2f}"
            in SECTION)
