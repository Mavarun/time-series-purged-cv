#!/usr/bin/env python3
"""Slice 11: the logistic boundary leak at n 1500 / 20 folds / 480 seeds.

Writes artifacts/logistic_n1500.{json,md}. The protocol is frozen in
``purged_cv.logistic_n1500``. ``--n-seeds`` shrinks the run for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import logistic_n1500 as P  # noqa: E402

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
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n": P.N_SAMPLES,
           "cells": [{"shock_phi": phi, "arms": arms} for phi, arms in cells.items()], "call": v,
           "per_seed": {str(phi): per[phi] for phi in P.SHOCK_PHIS}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logistic_n1500.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = (*P.ARMS, P.CONTROL)
    md = ["# Logistic leak at n 1500 (slice 11)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n {P.N_SAMPLES}, 20 folds, horizon 20; "
          "cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).", "",
          "| shock phi | " + " | ".join(LABELS[x] for x in names) + " | slice 10 control (n 500) |",
          "| ---: |" + " :--- |" * len(names) + " ---: |"]
    for phi, arms in cells.items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''}" for x in names]
        md.append(f"| {phi} | " + " | ".join(cols) + f" | {P.SLICE10_CONTROL[phi]:+.4f} |")
    md += ["", f"Control leaks in {v['control_leaks']} of 2 cells: measured **{'yes' if v['measured'] else 'no'}**; "
           f"default leaks in {v['default_leaks']}: default holds **{'yes' if v['default_holds'] else 'no'}**; "
           f"shrinks with n **{'yes' if v['shrinks_with_n'] else 'no'}**. "
           f"Largest control optimism + 2 s.e.: {v['control_upper_bound']:.4f} (slice 9 bound {P.SLICE9_BOUND:.3f})."]
    (a.out / "logistic_n1500.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
