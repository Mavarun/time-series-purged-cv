#!/usr/bin/env python3
"""Slice 14: participation vs mirror-aware (signed) DSR on short nulls and a phi 0.1 edge.

Writes artifacts/signed_dsr_calibration.{json,md}. The protocol is frozen in
``purged_cv.signed_dsr_calibration``. ``--n-seeds`` shrinks the run for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import signed_dsr_calibration as P  # noqa: E402

LABELS = {"normal": "normal", "garch": "GARCH-t", "regime": "vol regime"}


def markdown(cells: dict, v: dict, n_seeds: int) -> str:
    lines = [f"# Slice 14: signed participation DSR ({n_seeds} seeds per cell)", ""]
    for phi, title in ((0.0, "False positives (phi = 0)"), (0.1, "Power (phi = 0.1)")):
        lines += [f"## {title}", "", "| returns | bars | DSR participation | DSR signed | DSR simulated | max-stat p<0.05 | mean N participation / signed |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |"]
        for ret in P.RETURNS:
            for n in P.NS:
                c = cells[(n, ret, phi)]
                lines.append(f"| {LABELS[ret]} | {n} | {c['dsr_participation']:.3f} | {c['dsr_signed']:.3f} | "
                             f"{c['dsr_simulated']:.3f} | {c['max_p']:.3f} | {c['mean_n_eff_participation']:.1f} / {c['mean_n_eff_signed']:.1f} |")
        lines.append("")
    lines += [f"Calls: H1 {v['H1']}, H2 {v['H2']} ({v['power_cells_ok']}/9 power cells within 0.03 of max-stat), "
              f"recommend signed: {v['recommend_signed']}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    jobs = [(n, ret, phi, s) for n in P.NS for ret in P.RETURNS for phi in P.PHIS for s in seeds]
    per: dict = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for key, r in ex.map(P.run_job, jobs, chunksize=16):
            per.setdefault(key, []).append(r)
    cells = {k: P.summarize(v) for k, v in per.items()}
    v = P.call(cells)
    a.out.mkdir(exist_ok=True)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "call": v,
           "cells": [{"n": k[0], "returns": k[1], "phi": k[2], **c} for k, c in cells.items()]}
    (a.out / "signed_dsr_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    (a.out / "signed_dsr_calibration.md").write_text(markdown(cells, v, len(seeds)))
    print(markdown(cells, v, len(seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
