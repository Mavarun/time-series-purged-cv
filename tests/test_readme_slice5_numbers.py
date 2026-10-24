"""Numbers quoted in the README's slice-5 section match artifacts/rc_calibration.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Slice 5 (2026-10-24)")
    return text[start: text.index("\n## ", start + 5)]


def _cells() -> dict:
    rep = json.loads((ROOT / "artifacts" / "rc_calibration.json").read_text())
    return {(r["innovations"], r["phi"], r["cost_bps"]): r for r in rep["rows"]}


def test_table_rows_match_the_artifact():
    s, cells = _section(), _cells()
    shown = [("normal", 0.0, 0.0), ("normal", 0.0, 2.0), ("normal", 0.1, 2.0),
             ("garch", 0.0, 0.0), ("garch", 0.0, 2.0), ("garch", 0.1, 2.0)]
    for innov, phi, cost in shown:
        r = cells[(innov, phi, cost)]
        label = "garch-t" if innov == "garch" else "normal"
        mx = f"**{r['max_p']:.3f}**" if phi > 0 else f"{r['max_p']:.3f}"
        row = (f"| {label} | {phi:.2f} | {cost:g} | {r['dsr_raw']:.3f} | {r['dsr_simulated']:.3f} | "
               f"{mx} | {r['rc_p']:.3f} |")
        assert row in s, row


def test_reading_numbers_match_and_the_rc_never_beats_max_stat_on_power():
    s, c = _section(), _cells()
    assert f"0.085 false positives" in s and c[("garch", 0.0, 0.0)]["dsr_participation"] == 0.085
    assert c[("normal", 0.0, 0.0)]["dsr_participation"] == 0.06
    for innov in ("normal", "garch"):
        for cost in (0.0, 2.0):
            e = c[(innov, 0.1, cost)]
            assert e["rc_p"] < e["max_p"]
