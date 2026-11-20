"""Slice 13 protocol: does the two-sided ACF embargo remove slice 12's k-NN residual? (frozen 2026-11-20)

Slice 12 measured a real leak left by the package default (``pct1_after``) for the k-NN at
n 500, 20 folds (25-row test folds), shock phi 0.9: +0.0110 (s.e. 0.0042). Slice 7 built a
two-sided label-ACF embargo (``acf_both``) for persistent labels; slice 8b measured its cost on
a real edge. This slice asks whether ``acf_both`` removes the residual in that one cell.
Written before any score of this grid was computed. A smoke run may only use seeds outside the
evaluation range (``SMOKE_SEEDS``).

Cell: k-NN, n 500, 20 folds, horizon 20, feature phi 0.98, shock phi 0.9; seeds 10400-10879
(480, disjoint from slices 4-12). Arms ``purge_only``, ``pct1_after`` (default), ``acf_both``
(candidate) and the no-purge control. Leak test: mean optimism more than 2 s.e. above zero.

Pre-stated calls:

1. **Reproduction**: the control and the default must both leak (slice 12's residual must
   reappear on fresh seeds). If the default does not leak, the README reports that and no
   candidate call is made.
2. **Candidate removes the residual** iff ``acf_both`` is leak-free **and** the paired per-seed
   difference (default - candidate) is more than 2 s.e. above zero.
3. Budget: the candidate's mean training rows must be >= 90% of the default's; otherwise the
   call in 2 is reported as "removes it, over the row budget".

The package default does not change in this slice. If 2 passes within budget, the README
documents ``acf_both`` as the setting for k-NN with short test folds and persistent shocks.

Expectation written before the run: ``acf_both`` is leak-free, but the paired reduction is under
2 s.e. (call 2 fails on power).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N_SAMPLES = 500
N_SPLITS = 20
HORIZON = 20
FEATURE_PHI = 0.98
SHOCK_PHI = 0.9
SEEDS = tuple(range(10400, 10880))
SMOKE_SEEDS = (99990, 99991)
ARMS: dict[str, dict] = {
    "purge_only": {"embargo_pct": 0.0},
    "pct1_after": {"embargo_pct": 0.01},
    "acf_both": {"embargo_bars": "acf", "embargo_side": "both"},
}
CONTROL = "contiguous_no_purge"
DEFAULT_ARM = "pct1_after"
CANDIDATE = "acf_both"
SE_MULT = 2.0
MIN_ROW_SHARE = 0.90


def one_seed(seed: int) -> dict:
    """CV accuracy and total training rows per arm on one null draw."""
    ds = make_autocorrelated_null_dataset(N_SAMPLES, HORIZON, FEATURE_PHI, 3, seed, shock_phi=SHOCK_PHI)
    model = make_sign_classifier("knn")
    splitters = {name: PurgedKFold(N_SPLITS, ds.label_start_times, ds.label_end_times, **kw) for name, kw in ARMS.items()}
    splitters[CONTROL] = KFold(N_SPLITS, shuffle=False)
    acc, rows = {}, {}
    for name, cv in splitters.items():
        scores, n_tr = [], 0
        for tr, te in cv.split(ds.X, ds.y):
            scores.append(float((clone(model).fit(ds.X[tr], ds.y[tr]).predict(ds.X[te]) == ds.y[te]).mean()))
            n_tr += len(tr)
        acc[name], rows[name] = float(np.mean(scores)), int(n_tr)
    return {"accuracy": acc, "train_rows": rows}


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize(results: list[dict]) -> dict:
    arms = {}
    for name in (*ARMS, CONTROL):
        m, s = _mean_se(np.array([r["accuracy"][name] - 0.5 for r in results]))
        arms[name] = {"optimism": m, "se": s, "leak": bool(m > SE_MULT * s),
                      "mean_train_rows": float(np.mean([r["train_rows"][name] for r in results]))}
    d, ds = _mean_se(np.array([r["accuracy"][DEFAULT_ARM] - r["accuracy"][CANDIDATE] for r in results]))
    return {"arms": arms, "paired_diff": {"mean": d, "se": ds}}


def call(summary: dict) -> dict:
    a, p = summary["arms"], summary["paired_diff"]
    reproduced = a[CONTROL]["leak"] and a[DEFAULT_ARM]["leak"]
    share = a[CANDIDATE]["mean_train_rows"] / a[DEFAULT_ARM]["mean_train_rows"]
    removes = reproduced and not a[CANDIDATE]["leak"] and p["mean"] > SE_MULT * p["se"]
    return {"reproduced": bool(reproduced), "candidate_leak_free": bool(not a[CANDIDATE]["leak"]),
            "paired_reduction_significant": bool(p["mean"] > SE_MULT * p["se"]),
            "removes_residual": bool(removes), "row_share": float(share),
            "within_budget": bool(share >= MIN_ROW_SHARE)}
