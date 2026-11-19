#!/usr/bin/env python3
"""Slice 8: does purging stop being enough at many folds or short samples?

Writes artifacts/fold_count_calibration.{json,md}. The protocol (grid, arms, seeds and the
pre-stated calls) is frozen in ``purged_cv.fold_count_calibration``. Offline, seeded, k-NN on
a null whose true accuracy is 0.5. ``--n-seeds`` shrinks the run for a smoke test (the
committed artifact uses the full 100 seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import fold_count_calibration as P  # noqa: E402

LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "acf_both": "ACF both sides",
          "contiguous_no_purge": "no purge (control)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    keys = [(n, k, phi) for n in P.N_SAMPLES for k in P.N_SPLITS for phi in P.SHOCK_PHIS]
    jobs = [(*key, s) for key in keys for s in seeds]
    per: dict[tuple, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for n, k, phi, r in ex.map(P.run_job, jobs, chunksize=4):
            per.setdefault((n, k, phi), []).append(r)
    cells = {key: P.summarize(per[key]) for key in keys}
    verdict = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "horizon": P.HORIZON,
           "cells": [{"n": n, "n_splits": k, "shock_phi": phi, "arms": arms} for (n, k, phi), arms in cells.items()],
           "call": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fold_count_calibration.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = (*P.ARMS, P.CONTROL)
    md = ["# Fold-count calibration (slice 8)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), horizon {P.HORIZON}, k-NN; cells show CV accuracy - 0.5 "
          "(s.e.) and, in brackets, the share of the default arm's train rows kept. `*` = more than 2 s.e. above zero (leak).", "",
          "| n | folds | shock phi | " + " | ".join(LABELS[x] for x in names) + " |",
          "| ---: | ---: | ---: |" + " :--- |" * len(names)]
    for (n, k, phi), arms in cells.items():
        cols = [f"{arms[x]['optimism']:+.4f} ({arms[x]['se']:.4f}){'*' if arms[x]['leak'] else ''} "
                f"[{arms[x]['keep_vs_default']:.3f}]" for x in names]
        md.append(f"| {n} | {k} | {phi} | " + " | ".join(cols) + " |")
    v = verdict
    md += ["", f"Sensitivity: the no-purge control leaks in {v['control_leaks']} of {v['n_cells']} cells "
           f"(need >= {int(P.MIN_CONTROL_SHARE * v['n_cells'])}); testable: **{'yes' if v['testable'] else 'no'}**.",
           f"Default (1% after) leaks in {v['default_fails_many_folds']} of {v['n_many_fold_cells']} cells with >= "
           f"{P.MANY_FOLDS} folds; default documented as sufficient: **{'yes' if v['default_holds'] else 'no'}**.",
           f"Candidate (ACF both sides) leak-free at many folds: {v['candidate_leak_free_many_folds']}; min rows kept "
           f"{v['candidate_min_keep']:.3f} (need >= {P.MIN_KEEP_SHARE}); recommend: **{'yes' if v['recommend_candidate'] else 'no'}**."]
    (a.out / "fold_count_calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
