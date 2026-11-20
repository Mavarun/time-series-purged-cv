"""Every slice-12 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SECTION = README[README.index("## Slice 12 (2026-11-20)"): README.index("## Slice 13 (2026-11-20)")]
A = json.loads((ROOT / "artifacts" / "knn_residual.json").read_text())
B = json.loads((ROOT / "artifacts" / "logistic_power.json").read_text())
CELLS = {c["shock_phi"]: c["arms"] for c in A["cells"]}


def test_table_is_the_artifact():
    for line in (ROOT / "artifacts" / "knn_residual.md").read_text().splitlines():
        if line.startswith("|"):
            assert line in SECTION, line


def test_calls_and_prose():
    v = A["call"]
    assert v["reproduced"] and v["residual_real"] and f"leaks in {v['default_leaks']} of 2 cells" in SECTION
    for phi in (0.0, 0.9):
        x = CELLS[phi]["pct1_after"]
        assert f"{x['optimism']:+.4f}\n(s.e. {x['se']:.4f})" in SECTION or f"{x['optimism']:+.4f} (s.e. {x['se']:.4f})" in SECTION
    x = CELLS[0.9]["pct1_after"]
    assert f"{x['optimism'] / x['se']:.1f} s.e. above zero" in SECTION
    assert f"(+{x['optimism']:.3f} against +{CELLS[0.9]['contiguous_no_purge']['optimism']:.3f})" in SECTION


def test_bonferroni_claim():
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - 0.0228 / 2)  # 2-s.e. one-sided level split over 2 cells
    assert f"({z:.2f} s.e.)" in SECTION
    x = CELLS[0.9]["pct1_after"]
    assert x["optimism"] / x["se"] > z


def test_posthoc_pool():
    for phi in (0.0, 0.9):
        a = CELLS[phi]["pct1_after"]
        b = next(c for c in B["cells"] if c["shock_phi"] == phi and c["learner"] == "knn")["arms"]["pct1_after"]
        w1, w2 = a["se"] ** -2, b["se"] ** -2
        m, s = (a["optimism"] * w1 + b["optimism"] * w2) / (w1 + w2), (w1 + w2) ** -0.5
        assert f"+{m:.4f} (s.e. {s:.4f}) at phi {'0.9' if phi else '0'}" in SECTION


def test_run_command_listed():
    assert "python scripts/run_knn_residual.py" in README
