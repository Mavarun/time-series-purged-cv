#!/usr/bin/env python3
"""Slice 7 post-hoc positive control -> artifacts/embargo_positive_control.{json,md}.

Selects nothing: the slice 7 call stands. Scores the grid's seeds with two splitters known
to leak (contiguous K-fold without purge, shuffled K-fold) to show the measurement can see
a leak on this null at all.
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
        for phi, h, r in ex.map(P.run_control_job, jobs, chunksize=4):
            per.setdefault((phi, h), []).append(r)
    grid = json.loads((ROOT / "artifacts" / "embargo_calibration.json").read_text(encoding="utf-8"))
    purge = {(c["shock_phi"], c["horizon"]): c["arms"]["purge_only"] for c in grid["cells"]}
    cells = [{"shock_phi": phi, "horizon": h, "arms": P.summarize_control(v)} for (phi, h), v in per.items()]
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "cells": cells}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "embargo_positive_control.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = ["# Slice 7 positive control (post-hoc, selects nothing)", "",
          f"Same {len(seeds)} seeds per cell; CV accuracy - 0.5 (s.e.); `*` = leak (> 2 s.e.). "
          "Purge-only is copied from the grid for comparison.", "",
          "| shock phi | horizon | contiguous, no purge | shuffled K-fold | purge only (grid) |",
          "| ---: | ---: | :--- | :--- | :--- |"]
    for c in cells:
        cols = [f"{d['optimism']:+.4f} ({d['se']:.4f}){'*' if d['leak'] else ''}" for d in c["arms"].values()]
        p = purge.get((c["shock_phi"], c["horizon"]))
        pc = f"{p['optimism']:+.4f} ({p['se']:.4f})" if p else "n/a"
        md.append(f"| {c['shock_phi']} | {c['horizon']} | " + " | ".join(cols) + f" | {pc} |")
    (a.out / "embargo_positive_control.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
