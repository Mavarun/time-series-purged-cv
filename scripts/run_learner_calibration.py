#!/usr/bin/env python3
"""Slice 9: do the purge results hold for logistic and random-forest learners, not just k-NN?

Writes artifacts/learner_calibration.{json,md}. The protocol (grid, arms, seeds and the
pre-stated calls) is frozen in ``purged_cv.learner_calibration``. ``--n-seeds`` shrinks the
run for a smoke test (the committed artifact uses all 60 seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import learner_calibration as P  # noqa: E402

LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "contiguous_no_purge": "no purge (control)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    keys = [(m, k, phi) for m in P.LEARNERS for k in P.N_SPLITS for phi in P.SHOCK_PHIS]
    jobs = [(*key, s) for key in keys for s in seeds]
    per: dict[tuple, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for m, k, phi, r in ex.map(P.run_job, jobs, chunksize=2):
            per.setdefault((m, k, phi), []).append(r)
    cells = {key: P.summarize(per[key]) for key in keys}
    verdict = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n": P.N_SAMPLES, "horizon": P.HORIZON,
           "cells": [{"learner": m, "n_splits": k, "shock_phi": phi, "arms": arms} for (m, k, phi), arms in cells.items()],
           "call": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "learner_calibration.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = (*P.ARMS, P.CONTROL)
    md = ["# Learner calibration (slice 9)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n {P.N_SAMPLES}, horizon {P.HORIZON}; cells show CV accuracy - 0.5 "
          "(s.e.). `*` = more than 2 s.e. above zero (leak).", "",
          "| learner | folds | shock phi | " + " | ".join(LABELS[x] for x in names) + " |",
          "| :--- | ---: | ---: |" + " :--- |" * len(names)]
    for (m, k, phi), arms in cells.items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''}" for x in names]
        md.append(f"| {m} | {k} | {phi} | " + " | ".join(cols) + " |")
    md += ["", "| learner | control leaks | testable | default leaks | default holds |", "| :--- | ---: | :--- | ---: | :--- |"]
    for m, v in verdict["per_learner"].items():
        md.append(f"| {m} | {v['control_leaks']} of {v['n_cells']} | {'yes' if v['testable'] else 'no'} | "
                  f"{v['default_leaks']} | {'yes' if v['default_holds'] else 'no'} |")
    md += ["", f"Testable learners: {verdict['n_testable']} (need >= {P.MIN_TESTABLE_LEARNERS}); "
           f"drop the k-NN-only caveat: **{'yes' if verdict['widen_claim'] else 'no'}**."]
    (a.out / "learner_calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
