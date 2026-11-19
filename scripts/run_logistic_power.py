#!/usr/bin/env python3
"""Slice 10: is logistic regression's boundary leak visible at n 500 / 20 folds / 240 seeds?

Writes artifacts/logistic_power.{json,md}. The protocol (grid, arms, seeds and the pre-stated
calls) is frozen in ``purged_cv.logistic_power``. ``--n-seeds`` shrinks the run for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import logistic_power as P  # noqa: E402

LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "contiguous_no_purge": "no purge (control)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    keys = [(m, phi) for m in P.LEARNERS for phi in P.SHOCK_PHIS]
    jobs = [(*key, s) for key in keys for s in seeds]
    per: dict[tuple, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, phi, r in ex.map(P.run_job, jobs, chunksize=4):
            per.setdefault((m, phi), []).append(r)
    cells = {key: P.summarize(per[key]) for key in keys}
    verdict = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n": P.N_SAMPLES, "n_splits": P.N_SPLITS,
           "horizon": P.HORIZON,
           "cells": [{"learner": m, "shock_phi": phi, "arms": arms} for (m, phi), arms in cells.items()],
           "call": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "logistic_power.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = (*P.ARMS, P.CONTROL)
    md = ["# Logistic power (slice 10)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n {P.N_SAMPLES}, {P.N_SPLITS} folds, horizon {P.HORIZON}; "
          "cells show CV accuracy - 0.5 (s.e.). `*` = more than 2 s.e. above zero (leak).", "",
          "| learner | shock phi | " + " | ".join(LABELS[x] for x in names) + " |",
          "| :--- | ---: |" + " :--- |" * len(names)]
    for (m, phi), arms in cells.items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''}" for x in names]
        md.append(f"| {m} | {phi} | " + " | ".join(cols) + " |")
    v = verdict
    md += ["", f"k-NN reproduction of slice 8: **{'pass' if v['reproduced'] else 'FAIL'}**. "
           f"Logistic control leaks in {v['logistic_control_leaks']} of 2 cells: testable **{'yes' if v['logistic_testable'] else 'no'}**; "
           f"default leaks in {v['logistic_default_leaks']}: default holds **{'yes' if v['logistic_default_holds'] else 'no'}**. "
           f"Largest logistic control optimism + 2 s.e.: {v['logistic_control_upper_bound']:.4f}."]
    (a.out / "logistic_power.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
