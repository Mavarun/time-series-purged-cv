#!/usr/bin/env python3
"""DSR / max-statistic calibration across many seeds (offline, seeded).

Writes artifacts/dsr_calibration.{json,md}. False-positive rate on the null
(phi=0) and power on planted lag-1 momentum edges, gross and net of costs,
for raw-N DSR, participation-N DSR, simulated-N DSR and the max-statistic
p-value. Research diagnostics only - not live PnL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv.calibration import METHODS, calibration_table  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--n-bars", type=int, default=2520)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.03, 0.05, 0.1])
    ap.add_argument("--cost-bps", type=float, nargs="+", default=[0.0, 2.0, 5.0])
    ap.add_argument("--n-sims", type=int, default=4000)
    ap.add_argument("--no-write", action="store_true", help="do not overwrite artifacts/")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def build(args) -> dict:
    rows = calibration_table(
        phis=tuple(args.phis),
        seeds=range(args.n_seeds),
        cost_bps=tuple(args.cost_bps),
        n=args.n_bars,
        n_sims=args.n_sims,
    )
    return {"n_seeds": args.n_seeds, "n_bars": args.n_bars, "n_sims": args.n_sims, "rows": rows}


def summarise(rep: dict) -> str:
    lines = [
        "# Selection-hurdle calibration (DSR effective N, max-statistic test)",
        "",
        f"{rep['n_seeds']} seeds per cell, n={rep['n_bars']} daily bars, 40-rule grid, "
        f"{rep['n_sims']} null draws. Pass = DSR > 0.95 (null Sharpe variance) or max p < 0.05. "
        "phi=0 rows are false-positive rates; phi>0 rows are power. +- is one binomial s.e.",
        "",
        "| phi | cost bps | DSR raw N=40 | DSR participation N | DSR simulated N | max-stat p<0.05 "
        "| mean N (particip.) | mean N (simulated) | best = lag-1 long |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rep["rows"]:
        cells = " | ".join(f"{r[m]:.3f} +- {r[m + '_se']:.3f}" for m in METHODS)
        lines.append(
            f"| {r['phi']:.2f} | {r['cost_bps']:g} | {cells} | {r['mean_n_eff_participation']:.1f} "
            f"| {r['mean_n_eff_simulated']:.1f} | {r['frac_best_is_lag1_long']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = build(args)
    md = summarise(rep)
    if not args.no_write:
        art = ROOT / "artifacts"
        art.mkdir(exist_ok=True)
        (art / "dsr_calibration.json").write_text(json.dumps(rep, indent=2) + "\n")
        (art / "dsr_calibration.md").write_text(md)
    print(json.dumps(rep, indent=2) if args.json else md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
