#!/usr/bin/env python3
"""Walk-forward selector ablation: shrinkage, turnover penalty, sqrt impact.

Offline, seeded. Reports OOS net Sharpe (annualised) for each score mode
under linear bps costs and optional square-root impact. Null (phi=0) and
planted short-horizon momentum (phi>0) are both shown so a reader can see
whether the enhanced selectors merely suppress noise or also help a real edge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_cv.selection import run_selector_ablation  # noqa: E402
from purged_cv.sharpe import sharpe_ratio  # noqa: E402


def build(args: argparse.Namespace) -> dict:
    rows = []
    for phi in args.phis:
        for seed in args.seeds:
            for cost in args.cost_bps:
                rep = run_selector_ablation(
                    n=args.n_bars,
                    phi=phi,
                    cost_bps=cost,
                    seed=seed,
                    train_window=args.train_window,
                    step=args.step,
                    impact_coefs=tuple(args.impact_coefs),
                    shrink_intensity=args.shrink_intensity,
                    turnover_penalty=args.turnover_penalty,
                )
                rows.append(rep)
    return {
        "n_bars": args.n_bars,
        "train_window": args.train_window,
        "step": args.step,
        "shrink_intensity": args.shrink_intensity,
        "turnover_penalty": args.turnover_penalty,
        "impact_coefs": list(args.impact_coefs),
        "runs": rows,
    }


def summarise(report: dict) -> str:
    lines = [
        "# Walk-forward selector ablation (shrinkage, turnover, sqrt impact)",
        "",
        f"Settings: n={report['n_bars']}, train={report['train_window']}, "
        f"step={report['step']}, shrink_intensity={report['shrink_intensity']}, "
        f"turnover_penalty={report['turnover_penalty']}.",
        "",
        "Each cell is OOS net Sharpe (annualised, sqrt(252)), mean over seeds. "
        "Costs are charged on the stitched OOS path (switches included).",
        "",
    ]
    # group by (phi, cost, impact)
    from collections import defaultdict

    bucket: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    switch_bucket: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in report["runs"]:
        for ic, modes in run["by_impact"].items():
            key = (run["phi"], run["cost_bps"], float(ic))
            for mode, m in modes.items():
                bucket[key][mode].append(m["oos_net_sharpe_ann"])
                switch_bucket[key][mode].append(m["n_switches"])

    lines.append("| phi | cost bps | impact | sharpe | shrunk | sharpe-turnover | shrunk-turnover |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(bucket):
        phi, cost, ic = key
        m = bucket[key]
        lines.append(
            f"| {phi:.2f} | {cost:g} | {ic:g} | "
            f"{np.mean(m['sharpe']):.2f} | {np.mean(m['shrunk_sharpe']):.2f} | "
            f"{np.mean(m['sharpe_minus_turnover']):.2f} | "
            f"{np.mean(m['shrunk_sharpe_minus_turnover']):.2f} |"
        )
    lines += ["", "Mean walk-forward switches (lower is less churn):", ""]
    lines.append("| phi | cost bps | impact | sharpe | shrunk | sharpe-turnover | shrunk-turnover |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(switch_bucket):
        phi, cost, ic = key
        m = switch_bucket[key]
        lines.append(
            f"| {phi:.2f} | {cost:g} | {ic:g} | "
            f"{np.mean(m['sharpe']):.1f} | {np.mean(m['shrunk_sharpe']):.1f} | "
            f"{np.mean(m['sharpe_minus_turnover']):.1f} | "
            f"{np.mean(m['shrunk_sharpe_minus_turnover']):.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bars", type=int, default=2520)
    ap.add_argument("--train-window", type=int, default=504)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--phis", type=float, nargs="+", default=[0.0, 0.1])
    ap.add_argument("--cost-bps", type=float, nargs="+", default=[2.0, 5.0])
    ap.add_argument("--impact-coefs", type=float, nargs="+", default=[0.0, 0.0005, 0.001])
    ap.add_argument("--shrink-intensity", type=float, default=0.5)
    ap.add_argument("--turnover-penalty", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build(args)
    md = summarise(report)
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / "selector_ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    (art / "selector_ablation.md").write_text(md)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
