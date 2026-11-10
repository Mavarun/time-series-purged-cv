"""Numbers in the README's slice 7 section match the committed artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
HEAD = "## Slice 7 (2026-11-10)"
SECTION = README[README.index(HEAD): README.index("\n## ", README.index(HEAD) + 5)]


def _j(name):
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def test_table_rows_match_both_artifacts():
    g = {(c["shock_phi"], c["horizon"]): c["arms"] for c in _j("embargo_calibration.json")["cells"]}
    pc = {(c["shock_phi"], c["horizon"]): c["arms"] for c in _j("embargo_positive_control.json")["cells"]}
    for k, a in g.items():
        f = lambda d: f"{d['optimism']:+.4f}"  # noqa: E731
        row = (f"| {k[0]} | {k[1]} | {f(a['purge_only'])} | {f(a['pct1_after'])} | {f(a['acf_both'])} | "
               f"{a['acf_after']['median_embargo']:.0f} | {a['acf_both']['keep_vs_default']:.3f} | "
               f"{f(pc[k]['contiguous_no_purge'])} | **{f(pc[k]['shuffled_kfold'])}** |")
        assert row in SECTION, k


def test_call_and_default_stated():
    call = _j("embargo_calibration.json")["call"]
    assert call["recommend"] is False and "does\n  **not** recommend" in SECTION
    assert "the default is unchanged" in SECTION


def test_section_sits_after_slice_6_and_before_how_to_run():
    assert README.index("## Slice 6 (2026-11-03)") < README.index(HEAD) < README.index("## How to run")
    assert "run_embargo_calibration.py" in README[README.index("## How to run"):]
