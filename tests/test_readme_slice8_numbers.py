"""Every slice 8 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FC = json.loads((ROOT / "artifacts" / "fold_count_calibration.json").read_text())
EC = json.loads((ROOT / "artifacts" / "edge_cost.json").read_text())


def _f(d):
    return f"{d['optimism']:+.4f}{'*' if d['leak'] else ''}"


def test_null_table_rows():
    for c in FC["cells"]:
        a = c["arms"]
        row = (f"| {c['n']} | {c['n_splits']} | {c['shock_phi']} | {_f(a['purge_only'])} | {_f(a['pct1_after'])} | "
               f"{_f(a['acf_both'])} | {_f(a['contiguous_no_purge'])} |")
        assert row in README, row


def test_edge_cost_rows_and_call():
    for c in EC["cells"]:
        row = (f"| {c['n_splits']} | {c['shock_phi']} | {c['truth']:.3f} | {c['purge_only']['bias']:+.4f} | "
               f"{c['pct1_after']['cost_vs_purge_only']:+.4f} ({c['pct1_after']['cost_se']:.4f}) | "
               f"{c['acf_both']['cost_vs_purge_only']:+.4f} ({c['acf_both']['cost_se']:.4f}) |")
        assert row in README, row
    assert f"costs up to {EC['call']['max_cost']:.4f}" in README and EC["call"]["cheap"] is False


def test_prose_counts():
    v = FC["call"]
    assert f"it leaks in {v['control_leaks']} of 12 cells" in README
    assert f"The default leaks in {v['default_fails_many_folds']} of {v['n_many_fold_cells']} many-fold cells" in README
    assert "**not** recommended (pre-stated call)" in README and not v["recommend_candidate"]
