"""Every number in the README's slice 14 section comes from the committed artifact."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
A = json.loads((ROOT / "artifacts" / "signed_dsr_calibration.json").read_text())
C = {(c["n"], c["returns"], c["phi"]): c for c in A["cells"]}
SEC = README[README.index("## Slice 14 (2026-11-24)"): README.index("## How to run")]
L = {"normal": "normal", "garch": "GARCH-t", "regime": "vol regime"}


def test_order():
    assert README.index("## Slice 13 (2026-11-20)") < README.index("## Slice 14 (2026-11-24)") < README.index("## How to run")


def test_tables():
    for (n, r, phi), c in C.items():
        row = (f"| {L[r]} | {n} | {c['dsr_participation']:.3f} | {c['dsr_signed']:.3f} | "
               f"{c['dsr_simulated']:.3f} | {c['max_p']:.3f} |")
        assert row in SEC


def test_prose():
    short_null = [c for k, c in C.items() if k[2] == 0.0 and k[0] <= 504]
    assert round(min(c["dsr_participation"] for c in short_null), 3) == 0.057
    assert round(max(c["dsr_participation"] for c in short_null), 3) == 0.097
    assert round(min(c["dsr_signed"] for c in short_null), 3) == 0.010
    assert round(max(c["dsr_signed"] for k, c in C.items() if k[2] == 0.0), 3) == 0.027
    assert "5.7-9.7%" in SEC and "1.0-2.7%" in SEC and "1.0-2.7% in every cell" in SEC
    assert "1 of 9 power cells" in SEC and A["call"]["power_cells_ok"] == 1
    assert "0.067 vs 0.123" in SEC and round(C[(252, "normal", 0.1)]["dsr_signed"], 3) == 0.067
    assert round(C[(252, "normal", 0.1)]["max_p"], 3) == 0.123
    assert "(4.4 -> 8.8 on normal returns)" in SEC
    assert round(C[(252, "normal", 0.0)]["mean_n_eff_participation"], 1) == 4.4
    assert round(C[(252, "normal", 0.0)]["mean_n_eff_signed"], 1) == 8.8
    max_null = [c["max_p"] for k, c in C.items() if k[2] == 0.0]
    assert "(1.7-4.7%)" in SEC and round(min(max_null), 3) == 0.017 and round(max(max_null), 3) == 0.047
    assert "**not recommended**" in SEC and not A["call"]["recommend_signed"]
