#!/usr/bin/env python3
"""Reality check vs Gaussian max-statistic under normal and GARCH-t returns (offline, seeded).

Writes artifacts/rc_calibration.{json,md}. For each innovation type
(iid-normal AR(1), AR(1) with GARCH(1,1) Student-t(5) shocks, same
unconditional vol), phi in {0, 0.1} and cost in {0, 2} bps, the best of the
40-rule grid is tested by every slice-4 hurdle plus White's reality check
(stationary bootstrap, mean block 10 bars, fixed a priori). phi=0 rows are
false-positive rates, phi=0.1 rows are power. Cells run in parallel
processes; every run is seeded by its seed index, so results do not depend
on scheduling. Research diagnostics only - not live PnL.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv.calibration import METHODS, calibration_table  # noqa: E402

COLS = METHODS + ("rc_p",)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--n-bars", type=int, default=2520)
    ap.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--cost-bps", type=float, nargs="+", default=[0.0, 2.0])
    ap.add_argument("--innovations", nargs="+", default=["normal", "garch"])
    ap.add_argument("--n-sims", type=int, default=4000)
    ap.add_argument("--n-boot", type=int, default=499)
    ap.add_argument("--mean-block", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-write", action="store_true")
    return ap.parse_args(argv)


def _cell(job):
    innov, phi, cost, a = job
    (row,) = calibration_table(
        phis=(phi,), seeds=range(a["n_seeds"]), cost_bps=(cost,), n=a["n_bars"], n_sims=a["n_sims"],
        innovations=innov, with_rc=True, n_boot=a["n_boot"], mean_block=a["mean_block"],
    )
    return {"innovations": innov, **row}


def build(args) -> dict:
    a = {k: getattr(args, k) for k in ("n_seeds", "n_bars", "n_sims", "n_boot", "mean_block")}
    jobs = [(i, p, c, a) for i in args.innovations for p in args.phis for c in args.cost_bps]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_cell, jobs))
    else:
        rows = [_cell(j) for j in jobs]
    return {**a, "rows": rows}


def summarise(rep: dict) -> str:
    lines = [
        "# Reality check (stationary bootstrap) vs Gaussian max-statistic: normal vs GARCH-t returns",
        "",
        f"{rep['n_seeds']} seeds per cell, n={rep['n_bars']} daily bars, 40-rule grid, {rep['n_sims']} Gaussian "
        f"null draws, {rep['n_boot']} bootstrap draws, mean block {rep['mean_block']:g} bars. "
        "Pass = DSR > 0.95 or p < 0.05. phi=0 rows are false-positive rates; phi>0 rows are power. "
        "+- is one binomial s.e.",
        "",
        "| returns | phi | cost bps | DSR raw N=40 | DSR participation N | DSR simulated N | max-stat p<0.05 "
        "| reality check p<0.05 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rep["rows"]:
        cells = " | ".join(f"{r[m]:.3f} +- {r[m + '_se']:.3f}" for m in COLS)
        lines.append(f"| {r['innovations']} | {r['phi']:.2f} | {r['cost_bps']:g} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = build(args)
    md = summarise(rep)
    if not args.no_write:
        art = ROOT / "artifacts"
        art.mkdir(exist_ok=True)
        (art / "rc_calibration.json").write_text(json.dumps(rep, indent=2) + "\n")
        (art / "rc_calibration.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
