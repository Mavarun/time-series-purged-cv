"""Every slice-13 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

from purged_cv.residual_pool import cell_estimates, pool

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SECTION = README[README.index("## Slice 13 (2026-11-20)"): README.index("## How to run")]
A = json.loads((ROOT / "artifacts" / "acf_residual.json").read_text())
ARMS = A["summary"]["arms"]


def test_table_is_the_artifact():
    for line in (ROOT / "artifacts" / "acf_residual.md").read_text().splitlines():
        if line.startswith("|"):
            assert line in SECTION, line


def test_prose_numbers():
    d, c = ARMS["pct1_after"], ARMS["acf_both"]
    assert f"{d['optimism']:+.4f} (s.e. {d['se']:.4f})" in SECTION
    assert f"{c['optimism']:+.4f} (s.e. {c['se']:.4f}), {abs(c['optimism']) / c['se']:.1f} s.e." in SECTION
    p = A["summary"]["paired_diff"]
    assert f"{p['mean']:+.4f} (s.e. {p['se']:.4f}) at {A['call']['row_share']:.0%}" in SECTION
    assert f"(+{ARMS['contiguous_no_purge']['optimism']:.3f})" in SECTION
    assert not A["call"]["reproduced"] and "**Slice 12's\nresidual is not reproduced**" in SECTION


def test_pool_numbers():
    p = pool(cell_estimates())
    assert f"{p['n_seeds']}\n  disjoint seeds) is {p['mean']:+.4f} (s.e. {p['se']:.4f}), z {p['z']:.2f}" in SECTION
    assert f"p {p['q_p']:.2f}" in SECTION and f"differ by\n  {p['z12_vs13']:.2f} s.e." in SECTION
    assert f"+{p['upper_2se']:.3f} (mean + 2 s.e.)" in SECTION
    assert f"+{p['mean']:.4f} (s.e. {p['se']:.4f})" in README.split("## Limits")[1]
