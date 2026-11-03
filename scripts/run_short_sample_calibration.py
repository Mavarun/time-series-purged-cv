#!/usr/bin/env python3
"""Selection hurdles at short samples and under regime switching (offline, seeded).

Writes artifacts/short_sample_calibration.{json,md}. Slice 4/5 calibrated the hurdles
at n = 2520 daily bars (10 years) only. Here the same 40-rule grid, 2 bps cost and
hurdles (DSR raw / participation / simulated, Gaussian max-statistic p, White's reality
check with the stationary bootstrap, mean block 10, fixed a priori) are run at
n in {252, 504, 1260, 2520}, on iid-normal, GARCH(1,1)-t and Markov-vol-regime AR(1)
returns, phi in {0, 0.1}. phi = 0 rows are false-positive rates (nominal 5%); phi = 0.1
rows are power. Seeds start at 5000, disjoint from the 0-199 used by slices 4 and 5.
Nothing is fitted; every run is seeded by its index. Research diagnostics only.
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
    ap.add_argument("--n-seeds", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--n-bars", type=int, nargs="+", default=[252, 504, 1260, 2520])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--innovations", nargs="+", default=["normal", "garch", "regime"])
    ap.add_argument("--n-sims", type=int, default=4000)
    ap.add_argument("--n-boot", type=int, default=499)
    ap.add_argument("--mean-block", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-write", action="store_true")
    return ap.parse_args(argv)


def _cell(job):
    innov, n, phi, a = job
    seeds = range(a["seed_start"], a["seed_start"] + a["n_seeds"])
    (row,) = calibration_table(
        phis=(phi,), seeds=seeds, cost_bps=(a["cost_bps"],), n=n, n_sims=a["n_sims"],
        innovations=innov, with_rc=True, n_boot=a["n_boot"], mean_block=a["mean_block"],
    )
    return {"innovations": innov, "n_bars": n, **row}


def build(args) -> dict:
    a = {k: getattr(args, k) for k in ("n_seeds", "seed_start", "cost_bps", "n_sims", "n_boot", "mean_block")}
    jobs = [(i, n, p, a) for i in args.innovations for n in args.n_bars for p in args.phis]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_cell, jobs))
    else:
        rows = [_cell(j) for j in jobs]
    return {**a, "rows": rows}


def summarise(rep: dict) -> str:
    lines = [
        "# Selection hurdles at short samples and under regime switching",
        "",
        f"{rep['n_seeds']} seeds per cell (seeds {rep['seed_start']}+), {rep['cost_bps']:g} bps cost, 40-rule grid, "
        f"{rep['n_sims']} Gaussian null draws, {rep['n_boot']} bootstrap draws, mean block {rep['mean_block']:g}. "
        "Pass = DSR > 0.95 or p < 0.05. phi=0 rows are false-positive rates (nominal 0.05); phi=0.1 rows are "
        "power. +- is one binomial s.e.",
        "",
        "| returns | bars | phi | DSR raw N=40 | DSR participation N | DSR simulated N | max-stat p<0.05 "
        "| reality check p<0.05 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rep["rows"]:
        cells = " | ".join(f"{r[m]:.3f} +- {r[m + '_se']:.3f}" for m in COLS)
        lines.append(f"| {r['innovations']} | {r['n_bars']} | {r['phi']:.2f} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = build(args)
    md = summarise(rep)
    if not args.no_write:
        art = ROOT / "artifacts"
        art.mkdir(exist_ok=True)
        (art / "short_sample_calibration.json").write_text(json.dumps(rep, indent=2) + "\n")
        (art / "short_sample_calibration.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
