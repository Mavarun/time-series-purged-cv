"""Every slice-9 number the README quotes comes from the committed artifact."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 9 (2026-11-19)"): README.index("## Slice 10 (2026-11-20)")]
REC = json.loads((ROOT / "artifacts" / "learner_calibration.json").read_text(encoding="utf-8"))


def _fmt(a):
    return f"{a['optimism']:+.4f} ({a['se']:.4f}){'*' if a['leak'] else ''}"


def test_every_table_row_matches_the_artifact():
    for c in REC["cells"]:
        a = c["arms"]
        row = (f"| {c['learner']} | {c['n_splits']} | {c['shock_phi']} | {_fmt(a['purge_only'])} | "
               f"{_fmt(a['pct1_after'])} | {_fmt(a['contiguous_no_purge'])} |")
        assert row in SECTION, row


def test_prose_numbers_match_the_artifact():
    cells = {(c["learner"], c["n_splits"], c["shock_phi"]): c["arms"] for c in REC["cells"]}
    ctrl = lambda m, k, p: cells[(m, k, p)]["contiguous_no_purge"]
    assert f"+{ctrl('forest', 20, 0.9)['optimism']:.3f} vs +{ctrl('knn', 20, 0.9)['optimism']:.3f}" in SECTION
    assert f"({ctrl('logistic', 20, 0.9)['se']:.4f} vs {ctrl('knn', 20, 0.9)['se']:.4f}" in SECTION
    assert f"+{ctrl('logistic', 20, 0.9)['optimism']:.3f} point estimate" in SECTION
    low = min(cells[(m, 5, p)][a]["optimism"] for m in ("knn", "forest") for p in (0.0, 0.9) for a in ("purge_only", "pct1_after"))
    assert f"down to {low:.3f}" in SECTION
    assert REC["call"]["widen_claim"] and "caveat is dropped" in SECTION
