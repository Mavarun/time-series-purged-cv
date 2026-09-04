#!/usr/bin/env python3
"""Run the purged vs naive CV research slice and print honest metrics.

Default data path: synthetic AR(1) returns (reproducible, offline).
Optional: --yfinance to pull daily bars (Yahoo Finance via yfinance).

This script does NOT claim production accuracy or live PnL.
"""

from __future__ import annotations

import argparse
import json
import sys

from purged_cv.data import load_yfinance_daily_dataset, make_synthetic_ar_dataset
from purged_cv.metrics import compare_naive_vs_purged_cv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bars", type=int, default=800)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--n-lags", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--embargo-pct", type=float, default=0.01)
    p.add_argument("--model", choices=["logistic", "ridge"], default="logistic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--yfinance", action="store_true", help="Use SPY daily via yfinance")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--sample-weights", action="store_true")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.yfinance:
        ds = load_yfinance_daily_dataset(
            ticker=args.ticker, horizon=args.horizon, n_lags=args.n_lags
        )
    else:
        ds = make_synthetic_ar_dataset(
            n_bars=args.n_bars,
            horizon=args.horizon,
            n_lags=args.n_lags,
            seed=args.seed,
        )

    result = compare_naive_vs_purged_cv(
        ds.X,
        ds.y,
        ds.label_start_times,
        ds.label_end_times,
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct,
        model_kind=args.model,
        shuffle_naive=True,
        random_state=args.seed,
        forward_returns=ds.forward_returns,
        use_sample_weights=args.sample_weights,
    )

    payload = {
        "data_meta": ds.meta,
        "feature_names": ds.feature_names,
        "n_samples": int(len(ds.y)),
        "cv": result.to_dict(),
        "disclaimer": (
            "Research diagnostic only. Not production accuracy. Not live PnL. "
            "Cite data source in data_meta.citation."
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("=== Purged vs Naive CV slice ===")
        print(f"data: {ds.meta.get('source')} | samples={len(ds.y)} | horizon={ds.meta.get('horizon')}")
        print(f"citation: {ds.meta.get('citation')}")
        print(f"model: {result.model_kind} | splits: {result.n_splits}")
        print(f"naive  KFold accuracy:  mean={result.naive_mean:.4f}  std={result.naive_std:.4f}  folds={result.naive_scores}")
        print(f"purged KFold accuracy:  mean={result.purged_mean:.4f}  std={result.purged_std:.4f}  folds={result.purged_scores}")
        print(f"gap (naive - purged):   {result.gap_naive_minus_purged:.4f}")
        print("disclaimer:", payload["disclaimer"])
        for note in result.notes:
            print(f"- {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
