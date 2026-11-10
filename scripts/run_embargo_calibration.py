#!/usr/bin/env python3
"""Slice 7: does the embargo remove the leak purging leaves under persistent shocks?

Writes artifacts/embargo_calibration.{json,md}. The protocol (grid, arms, seeds and the
pre-stated recommendation rule) is frozen in ``purged_cv.embargo_calibration``. Offline,
seeded, k-NN on a null whose true accuracy is 0.5. ``--n-seeds`` shrinks the run for a
smoke test (the committed artifact uses the full 100 seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import embargo_calibration as P  # noqa: E402

LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "pct5_after": "5% after",
          "acf_after": "ACF after", "acf_both": "ACF both sides"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    jobs = [(phi, h, s) for phi in P.SHOCK_PHIS for h in P.HORIZONS for s in seeds]
    per: dict[tuple, list] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for phi, h, r in ex.map(P.run_job, jobs, chunksize=4):
            per.setdefault((phi, h), []).append(r)
    cells = {k: P.summarize(v) for k, v in per.items()}
    verdict = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n_samples": P.N_SAMPLES,
           "cells": [{"shock_phi": phi, "horizon": h, "arms": arms} for (phi, h), arms in cells.items()],
           "call": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "embargo_calibration.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = ["# Embargo calibration (slice 7)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n = {P.N_SAMPLES}, k-NN, 5 purged folds; "
          "cells show CV accuracy - 0.5 (s.e.) and, in brackets, the share of the default arm's train rows kept. "
          "`*` = more than 2 s.e. above zero (leak).", "",
          "| shock phi | horizon | " + " | ".join(LABELS[n] for n in P.ARMS) + " | median ACF embargo |",
          "| ---: | ---: |" + " :--- |" * len(P.ARMS) + " ---: |"]
    for (phi, h), arms in cells.items():
        cols = [f"{d['optimism']:+.4f} ({d['se']:.4f}){'' if d['leak_free'] else '*'} [{d['keep_vs_default']:.3f}]"
                for d in arms.values()]
        md.append(f"| {phi} | {h} | " + " | ".join(cols) + f" | {arms['acf_after']['median_embargo']:.0f} |")
    md += ["", f"Pre-stated call: candidate leak-free in every cell: {verdict['candidate_leak_free_everywhere']}; "
           f"default leaks in {verdict['default_fails_in_persistent_cells']} of 6 persistent cells (need >= {P.MIN_DEFAULT_FAILS}); "
           f"candidate keeps {verdict['candidate_min_keep_at_phi0']:.3f} of the default's rows at shock phi 0 (need >= {P.MIN_KEEP_SHARE}). "
           f"Recommend `acf_both`: **{'yes' if verdict['recommend'] else 'no'}**."]
    (a.out / "embargo_calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
