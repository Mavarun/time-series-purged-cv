#!/usr/bin/env python3
"""Slice 8b: what does the embargo cost when there is a real edge?

Writes artifacts/edge_cost.{json,md}. Protocol frozen in ``purged_cv.edge_cost``.
``--n-seeds`` shrinks the run for a smoke test (the committed artifact uses 100 seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv import edge_cost as P  # noqa: E402

LABELS = {"purge_only": "purge only", "pct1_after": "1% after (default)", "acf_both": "ACF both sides"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=len(P.SEEDS))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    seeds = P.SEEDS[: a.n_seeds]
    keys = [(k, phi) for k in P.FOLDS for phi in P.SHOCK_PHIS]
    per: dict = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, phi, r in ex.map(P.run_job, [(*key, s) for key in keys for s in seeds], chunksize=4):
            per.setdefault((k, phi), []).append(r)
    cells = {key: P.summarize(per[key]) for key in keys}
    verdict = P.call(cells)
    out = {"n_seeds": len(seeds), "seed_start": seeds[0], "n": P.N, "beta": P.BETA,
           "cells": [{"n_splits": k, "shock_phi": phi, **c} for (k, phi), c in cells.items()], "call": verdict}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "edge_cost.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = ["# Embargo cost on a real edge (slice 8b)", "",
          f"{len(seeds)} seeds per cell (from {seeds[0]}), n = {P.N}, beta = {P.BETA}, horizon {P.HORIZON}, k-NN. "
          "Truth = full-sample model on the next 5000 bars. Cells: CV accuracy, bias vs truth (s.e.), and cost vs purge only (s.e.).", "",
          "| folds | shock phi | truth | " + " | ".join(LABELS.values()) + " |", "| ---: | ---: | ---: |" + " :--- |" * 3]
    for (k, phi), c in cells.items():
        cols = [f"{c[x]['cv']:.4f}, bias {c[x]['bias']:+.4f} ({c[x]['bias_se']:.4f}), cost {c[x]['cost_vs_purge_only']:+.4f} ({c[x]['cost_se']:.4f})"
                for x in LABELS]
        md.append(f"| {k} | {phi} | {c['truth']:.4f} | " + " | ".join(cols) + " |")
    md += ["", f"Pre-stated call: worst embargo cost {verdict['max_cost']:+.4f} (limit {P.MAX_COST}); "
           f"embargo cheap on a real edge: **{'yes' if verdict['cheap'] else 'no'}**."]
    (a.out / "edge_cost.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
