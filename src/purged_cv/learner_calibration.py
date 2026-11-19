"""Slice 9 protocol: do the purge results hold beyond k-NN? (frozen 2026-11-19)

Slices 7-8 found that purging alone removes the boundary leak and that the 1% default is
sufficient, but only ever with a k-NN, which slice 8 lists as its main weakness. This slice
re-runs the slice-8 null with three learners. Written before any CV score of this grid was
computed. A 2-seed smoke run (seeds 9000-9001) is allowed only to check that it runs; no
arm, rule or threshold may change after it.

Null: slice 7's ``make_autocorrelated_null_dataset`` with ``feature_phi`` = 0.98, 3 features,
horizon 20, n = 1500. True accuracy is 0.5 for every arm.

Grid: learner in {logistic, knn, forest} x n_splits in {5, 20} x shock_phi in {0, 0.9};
seeds 9000-9059 (disjoint from slices 4-8). Arms: ``purge_only`` and ``pct1_after`` (package
default), plus ``contiguous_no_purge`` as the in-grid positive control. The k-NN rows repeat
slice 8's design on fresh seeds and act as a reproduction check.

Leak test per cell and arm: mean optimism (CV accuracy - 0.5) more than 2 s.e. above zero.

Pre-stated calls, per learner (the package default does not change in this slice):

1. **Testable**: the learner's 4 cells can see a leak iff the no-purge control leaks in >= 2
   of them. A learner that is not testable is reported as such; it is not evidence either way.
2. **Default holds** for a testable learner iff ``pct1_after`` is leak-free in all 4 cells.
3. **Claim widened**: the README may drop "k-NN only" from the slice 7-8 weakness iff the
   default holds for every testable learner AND at least 2 learners are testable.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N_SAMPLES = 1500
LEARNERS = ("logistic", "knn", "forest")
N_SPLITS = (5, 20)
SHOCK_PHIS = (0.0, 0.9)
HORIZON = 20
FEATURE_PHI = 0.98
SEEDS = tuple(range(9000, 9060))
ARMS: dict[str, dict] = {
    "purge_only": {"embargo_pct": 0.0},
    "pct1_after": {"embargo_pct": 0.01},
}
CONTROL = "contiguous_no_purge"
DEFAULT_ARM = "pct1_after"
SE_MULT = 2.0
MIN_CONTROL_LEAKS = 2
MIN_TESTABLE_LEARNERS = 2


def one_seed(learner: str, n_splits: int, shock_phi: float, seed: int) -> dict[str, float]:
    """CV accuracy per arm (and the control) for one learner on one null draw."""
    ds = make_autocorrelated_null_dataset(N_SAMPLES, HORIZON, FEATURE_PHI, 3, seed, shock_phi=shock_phi)
    model = make_sign_classifier(learner, random_state=seed) if learner == "forest" else make_sign_classifier(learner)
    splitters = {name: PurgedKFold(n_splits, ds.label_start_times, ds.label_end_times, **kw)
                 for name, kw in ARMS.items()}
    splitters[CONTROL] = KFold(n_splits, shuffle=False)
    acc = {}
    for name, cv in splitters.items():
        scores = [float((clone(model).fit(ds.X[tr], ds.y[tr]).predict(ds.X[te]) == ds.y[te]).mean())
                  for tr, te in cv.split(ds.X, ds.y)]
        acc[name] = float(np.mean(scores))
    return acc


def run_job(job: tuple[str, int, float, int]):
    learner, k, phi, seed = job
    return learner, k, phi, one_seed(learner, k, phi, seed)


def summarize(results: list[dict[str, float]]) -> dict[str, dict]:
    out = {}
    for name in (*ARMS, CONTROL):
        opt = np.array([r[name] - 0.5 for r in results])
        se = float(opt.std(ddof=1) / np.sqrt(len(opt))) if len(opt) > 1 else float("nan")
        out[name] = {"optimism": float(opt.mean()), "se": se, "leak": bool(opt.mean() > SE_MULT * se)}
    return out


def call(cells: dict[tuple[str, int, float], dict[str, dict]]) -> dict:
    per = {}
    for learner in LEARNERS:
        mine = [c for k, c in cells.items() if k[0] == learner]
        control = sum(c[CONTROL]["leak"] for c in mine)
        testable = control >= MIN_CONTROL_LEAKS
        default_leaks = sum(c[DEFAULT_ARM]["leak"] for c in mine)
        per[learner] = {"control_leaks": int(control), "n_cells": len(mine), "testable": bool(testable),
                        "default_leaks": int(default_leaks), "default_holds": bool(testable and default_leaks == 0)}
    testable = [x for x in LEARNERS if per[x]["testable"]]
    widen = len(testable) >= MIN_TESTABLE_LEARNERS and all(per[x]["default_holds"] for x in testable)
    return {"per_learner": per, "n_testable": len(testable), "widen_claim": bool(widen)}
