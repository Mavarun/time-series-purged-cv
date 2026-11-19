#!/usr/bin/env python3
"""Slice 12: is the purged k-NN residual at n 500 / 20 folds real? (480 seeds)

Writes artifacts/knn_residual.{json,md}. The protocol is frozen in
``purged_cv.knn_residual``. ``--n-seeds`` shrinks the run for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import knn_residual as P  # noqa: E402
from purged_cv.logistic_power import ARMS  # noqa: E402

SLICE10_DEFAULT = {0.0: 0.0073, 0.9: 0.0063}  # slice 10's k-NN pct1_after cells (README table)
LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "contiguous_no_purge": "no purge (control)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    jobs = [(phi, s) for phi in P.SHOCK_PHIS for s in seeds]
    per: dict[float, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for phi, r in ex.map(P.run_job, jobs, chunksize=8):  # map keeps the seed order
            per.setdefault(phi, []).append(r)
    cells = {phi: P.summarize(per[phi]) for phi in P.SHOCK_PHIS}
    v = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n": 500,
           "cells": [{"shock_phi": phi, "arms": arms} for phi, arms in cells.items()], "call": v,
           "per_seed": {str(phi): per[phi] for phi in P.SHOCK_PHIS}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "knn_residual.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = (*ARMS, P.CONTROL)
    md = ["# Purged k-NN residual at n 500 (slice 12)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n 500, 20 folds, horizon 20; "
          "cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).", "",
          "| shock phi | " + " | ".join(LABELS[x] for x in names) + " | slice 10 default |",
          "| ---: |" + " :--- |" * len(names) + " ---: |"]
    for phi, arms in cells.items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''}" for x in names]
        md.append(f"| {phi} | " + " | ".join(cols) + f" | {SLICE10_DEFAULT[phi]:+.4f} |")
    md += ["", f"Control reproduces (leaks in both cells): **{'yes' if v['reproduced'] else 'NO'}**. "
           f"Default leaks in {v['default_leaks']} of 2 cells, purge only in {v['purge_only_leaks']}: residual real "
           f"**{'yes' if v['residual_real'] else 'no'}**. Largest default optimism + 2 s.e.: {v['default_upper_bound']:.4f}."]
    (a.out / "knn_residual.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
