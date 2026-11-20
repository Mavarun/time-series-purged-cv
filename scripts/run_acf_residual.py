"""Slice 13: does the two-sided ACF embargo remove the purged k-NN residual? (480 seeds, one cell)."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from purged_cv import acf_residual as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the two smoke seeds only (outside the evaluation range)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    a = ap.parse_args()
    seeds = P.SMOKE_SEEDS if a.smoke else P.SEEDS
    with ProcessPoolExecutor(a.workers) as ex:
        per_seed = list(ex.map(P.one_seed, seeds, chunksize=8))
    s = P.summarize(per_seed)
    v = P.call(s)
    out = {"seeds": list(seeds[:1]) + [len(seeds)], "seed_start": seeds[0], "n_seeds": len(seeds),
           "summary": s, "call": v, "per_seed": per_seed}
    md = ["# Two-sided ACF embargo vs the purged k-NN residual (slice 13)", "",
          f"{len(seeds)} seeds (from {seeds[0]}), k-NN, n 500, 20 folds, horizon 20, shock phi 0.9. "
          "Optimism is CV accuracy - 0.5 (s.e.); `*` = more than 2 s.e. above zero.", "",
          "| arm | optimism | mean train rows |", "| :--- | :--- | ---: |"]
    for name, x in s["arms"].items():
        md.append(f"| {name} | {x['optimism']:+.4f} ({x['se']:.4f}){'*' if x['leak'] else ''} | {x['mean_train_rows']:.0f} |")
    p = s["paired_diff"]
    md += ["", f"Paired default - acf_both: {p['mean']:+.4f} (s.e. {p['se']:.4f}).", "",
           f"Reproduced: **{'yes' if v['reproduced'] else 'NO'}**. acf_both leak-free: "
           f"**{'yes' if v['candidate_leak_free'] else 'no'}**. Paired drop > 2 s.e.: "
           f"**{'yes' if v['paired_reduction_significant'] else 'no'}**. Removes residual: "
           f"**{'yes' if v['removes_residual'] else 'no'}**. Row share {v['row_share']:.3f} "
           f"({'within' if v['within_budget'] else 'over'} the 0.90 budget)."]
    if not a.smoke:
        a.out.mkdir(parents=True, exist_ok=True)
        del out["seeds"]
        (a.out / "acf_residual.json").write_text(json.dumps(out, indent=1) + "\n")
        (a.out / "acf_residual.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
