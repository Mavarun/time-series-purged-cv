"""Every slice 10 number in the README is read from artifacts/logistic_power.json."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SECTION = README[README.index("## Slice 10 (2026-11-20)"): README.index("## How to run")]
ART = json.loads((ROOT / "artifacts" / "logistic_power.json").read_text())
NAMES = ("purge_only", "pct1_after", "contiguous_no_purge")


def _cells():
    return {(c["learner"], c["shock_phi"]): c["arms"] for c in ART["cells"]}


def test_table_rows():
    for (m, phi), arms in _cells().items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''}" for x in NAMES]
        assert f"| {m} | {phi} | " + " | ".join(cols) + " |" in SECTION


def test_quoted_numbers():
    c = _cells()
    lg0, lg9 = c[("logistic", 0.0)]["contiguous_no_purge"], c[("logistic", 0.9)]["contiguous_no_purge"]
    k0, k9 = c[("knn", 0.0)]["contiguous_no_purge"], c[("knn", 0.9)]["contiguous_no_purge"]
    assert f"(+{k0['optimism']:.3f} / +{k9['optimism']:.3f} against slice 8's" in SECTION
    assert f"+{lg9['optimism']:.3f} at shock phi 0.9, about half the k-NN's +{k9['optimism']:.3f}" in SECTION
    assert f"its control leaks (+{lg0['optimism']:.3f} / +{lg9['optimism']:.3f})" in README
    p0, p9 = c[("knn", 0.0)]["purge_only"], c[("knn", 0.9)]["purge_only"]
    assert f"+{p0['optimism']:.3f} / +{p9['optimism']:.3f}, {p0['optimism'] / p0['se']:.1f} and {p9['optimism'] / p9['se']:.1f} s.e." in SECTION


def test_calls_quoted():
    v = ART["call"]
    assert v["reproduced"] and "The k-NN reproduction passes" in SECTION
    assert v["logistic_testable"] and v["logistic_default_holds"] and "**The default holds for a linear learner**" in SECTION
    assert f"{ART['n_seeds']} fresh seeds ({ART['seed_start']}-{ART['seed_start'] + ART['n_seeds'] - 1})" in SECTION
