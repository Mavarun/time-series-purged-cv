"""Slice 8 protocol: does purging stop being enough at many folds or short samples? (frozen 2026-11-19)

Slice 7 found no leak in any purged arm at 5 folds and n = 1500, and left open whether that
holds when the number of fold boundaries grows relative to n, which is when the residual
boundary leak (label autocorrelation beyond the label window) should be largest. Written
before any CV score of this grid was computed. A 2-seed smoke run of the script (seeds 8000-8001) is
allowed only to check that it runs; no arm, rule or threshold may change after it.

Null: slice 7's ``make_autocorrelated_null_dataset`` with ``feature_phi`` = 0.98, 3 features,
k-NN, horizon 20 and persistent shocks. True accuracy is 0.5 for every arm.

Grid: n in {500, 1500} x n_splits in {5, 10, 20} x shock_phi in {0, 0.9}; seeds 8000-8099
(disjoint from slices 4-7). Arms: ``purge_only``, ``pct1_after`` (package default) and
``acf_both`` (slice 7 candidate), plus ``contiguous_no_purge`` as an in-grid positive
control (it is known to leak; if it does not, the cell cannot see a leak).

Leak test per cell and arm: mean optimism (CV accuracy - 0.5) more than 2 s.e. above zero.

Pre-stated calls (the package default does not change in this slice either way):

1. **Default holds**: ``pct1_after`` is documented as sufficient at many folds iff it is
   leak-free in every cell with ``n_splits >= 10``.
2. **Candidate**: ``acf_both`` is documented as the recommended setting for many folds iff
   ``pct1_after`` leaks in >= 2 cells with ``n_splits >= 10`` AND ``acf_both`` is leak-free
   in all of those cells AND keeps >= 90% of the default's train rows in every cell.
3. **Sensitivity**: the grid only counts as informative if ``contiguous_no_purge`` leaks in
   at least half of the 12 cells; otherwise both calls are reported as "not testable".
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N_SAMPLES = (500, 1500)
N_SPLITS = (5, 10, 20)
SHOCK_PHIS = (0.0, 0.9)
HORIZON = 20
FEATURE_PHI = 0.98
SEEDS = tuple(range(8000, 8100))
ARMS: dict[str, dict] = {
    "purge_only": {"embargo_pct": 0.0},
    "pct1_after": {"embargo_pct": 0.01},
    "acf_both": {"embargo_bars": "acf", "embargo_side": "both"},
}
CONTROL = "contiguous_no_purge"
DEFAULT_ARM = "pct1_after"
CANDIDATE = "acf_both"
SE_MULT = 2.0
MANY_FOLDS = 10
MIN_DEFAULT_FAILS = 2
MIN_KEEP_SHARE = 0.90
MIN_CONTROL_SHARE = 0.5


def one_seed(n: int, n_splits: int, shock_phi: float, seed: int) -> dict:
    """CV accuracy and total train rows per arm (and the control) on one null draw."""
    ds = make_autocorrelated_null_dataset(n, HORIZON, FEATURE_PHI, 3, seed, shock_phi=shock_phi)
    model = make_sign_classifier("knn")
    acc, rows = {}, {}
    splitters = {name: PurgedKFold(n_splits, ds.label_start_times, ds.label_end_times, **kw)
                 for name, kw in ARMS.items()}
    splitters[CONTROL] = KFold(n_splits, shuffle=False)
    for name, cv in splitters.items():
        scores, n_tr = [], 0
        for tr, te in cv.split(ds.X, ds.y):
            m = clone(model).fit(ds.X[tr], ds.y[tr])
            scores.append(float((m.predict(ds.X[te]) == ds.y[te]).mean()))
            n_tr += len(tr)
        acc[name], rows[name] = float(np.mean(scores)), n_tr
    return {"accuracy": acc, "train_rows": rows}


def run_job(job: tuple[int, int, float, int]):
    n, k, phi, seed = job
    return n, k, phi, one_seed(n, k, phi, seed)


def summarize(results: list[dict]) -> dict[str, dict]:
    out = {}
    base = np.array([r["train_rows"][DEFAULT_ARM] for r in results], float)
    for name in (*ARMS, CONTROL):
        opt = np.array([r["accuracy"][name] - 0.5 for r in results])
        se = float(opt.std(ddof=1) / np.sqrt(len(opt))) if len(opt) > 1 else float("nan")
        out[name] = {"optimism": float(opt.mean()), "se": se, "leak": bool(opt.mean() > SE_MULT * se),
                     "keep_vs_default": float((np.array([r["train_rows"][name] for r in results], float) / base).mean())}
    return out


def call(cells: dict[tuple[int, int, float], dict[str, dict]]) -> dict:
    many = {k: c for k, c in cells.items() if k[1] >= MANY_FOLDS}
    control_leaks = sum(c[CONTROL]["leak"] for c in cells.values())
    testable = control_leaks >= MIN_CONTROL_SHARE * len(cells)
    default_fails = sum(c[DEFAULT_ARM]["leak"] for c in many.values())
    cand_clean = all(not c[CANDIDATE]["leak"] for c in many.values())
    min_keep = min(c[CANDIDATE]["keep_vs_default"] for c in cells.values())
    return {
        "control_leaks": int(control_leaks), "n_cells": len(cells), "testable": bool(testable),
        "default_fails_many_folds": int(default_fails), "n_many_fold_cells": len(many),
        "default_holds": bool(testable and default_fails == 0),
        "candidate_leak_free_many_folds": bool(cand_clean), "candidate_min_keep": float(min_keep),
        "recommend_candidate": bool(testable and default_fails >= MIN_DEFAULT_FAILS and cand_clean
                                    and min_keep >= MIN_KEEP_SHARE),
    }
