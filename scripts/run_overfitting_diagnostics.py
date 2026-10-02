#!/usr/bin/env python3
"""Leakage + backtest-overfitting diagnostics (synthetic, seeded, offline).

Part 1 - leakage: naive shuffled K-fold vs purged K-fold vs CPCV paths vs a
fresh holdout on autocorrelated null labels (true accuracy 0.5).
Part 2 - selection: best-of-40 causal sign rules on AR(1) returns, net of
per-trade costs; full-sample best Sharpe vs DSR vs PBO (CSCV) vs walk-forward
OOS net Sharpe, on a null (phi=0) and a planted-edge (phi>0) series.

Research diagnostics only - not live PnL.
"""

from __future__ import annotations

import argparse
import json
import sys

from purged_cv.leakage import leakage_sweep
from purged_cv.selection import run_selection_diagnostics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3, help="seeds for the leakage sweep")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--n-bars", type=int, default=2520, help="bars per selection run")
    p.add_argument("--signal-phi", type=float, default=0.1)
    p.add_argument("--cost-bps", type=float, nargs="+", default=[2.0, 5.0])
    p.add_argument("--selection-seed", type=int, default=1)
    p.add_argument("--n-partitions", type=int, default=16)
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def build_report(args) -> dict:
    leak = leakage_sweep(seeds=tuple(range(args.seeds)), horizon=args.horizon)
    leak.pop("per_seed")
    controls = {
        "horizon_1": leakage_sweep(seeds=tuple(range(args.seeds)), horizon=1),
        "iid_features": leakage_sweep(
            seeds=tuple(range(args.seeds)), horizon=args.horizon, feature_phi=0.0
        ),
    }
    for c in controls.values():
        c.pop("per_seed")
    selection = []
    for phi in (0.0, args.signal_phi):
        for cost in args.cost_bps:
            d = run_selection_diagnostics(
                n=args.n_bars,
                phi=phi,
                cost_bps=cost,
                seed=args.selection_seed,
                n_partitions=args.n_partitions,
            )
            selection.append(d.to_dict())
    return {"leakage": leak, "leakage_controls": controls, "selection": selection}


def main(argv=None) -> int:
    args = parse_args(argv)
    rep = build_report(args)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    lk = rep["leakage"]
    print(f"## Leakage on autocorrelated null labels (k-NN, h={args.horizon}, {lk['n_seeds']} seeds, truth=0.5)")
    print("| naive KFold | purged KFold | CPCV path mean | fresh holdout |")
    print("|---|---|---|---|")
    print(f"| {lk['naive_kfold']:.3f} | {lk['purged_kfold']:.3f} | {lk['cpcv_path_mean']:.3f} | {lk['fresh_holdout']:.3f} |")
    for name, c in rep["leakage_controls"].items():
        print(f"control {name}: naive-purged gap = {c['naive_minus_purged']:+.3f}")
    print()
    print("## Best-of-40 rule selection, net of costs (annualised Sharpe)")
    print("| phi | cost bps | best trial | IS best net SR | PSR(0) | DSR null | DSR x-sec | PBO | WF OOS net SR | WF OOS gross SR |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in rep["selection"]:
        print(
            f"| {d['phi']:.2f} | {d['cost_bps']:.1f} | {tuple(d['best_trial'])} | "
            f"{d['is_best_net_sharpe_ann']:.2f} | {d['psr_vs_zero']:.3f} | {d['dsr_null']:.3f} | "
            f"{d['dsr_cross_sectional']:.3f} | {d['pbo']:.3f} | {d['wf_oos_net_sharpe_ann']:.2f} | "
            f"{d['wf_oos_gross_sharpe_ann']:.2f} |"
        )
    print("\nResearch diagnostics on synthetic data only; not live PnL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
