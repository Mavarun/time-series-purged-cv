"""Slice 11 protocol: how big is the logistic leak at n 1500? (frozen 2026-11-20)

Slice 10 measured a logistic leak at n 500 / 20 folds (no-purge control +0.028 / +0.054) and
found the default leak-free. At n 1500 the leak was still only bounded: slice 9's control
read +0.019 at shock phi 0.9 with an s.e. of 0.011 (60 seeds). This slice measures it with
eight times slice 9's seeds. Written before any CV score of this grid was computed; a
1-seed timing run on dev seed 1 (not in the grid) is the only score seen.

Null: ``make_autocorrelated_null_dataset`` with ``feature_phi`` = 0.98, 3 features, horizon
20, n = 1500. True accuracy is 0.5 for every arm.

Grid: logistic only x shock_phi in {0, 0.9}; 20 folds; seeds 9400-9879 (480, disjoint from
slices 4-10). Arms ``purge_only`` and ``pct1_after`` (package default) and the
``contiguous_no_purge`` control, as in slice 10.

Leak test per cell and arm: mean optimism more than 2 s.e. above zero.

Pre-stated calls:

1. **Measured** iff the control leaks in >= 1 of the 2 cells; the n 1500 leak is then reported
   with its s.e. If it does not leak, the bound (largest control optimism + 2 s.e.) replaces
   slice 9's +0.040.
2. **Default holds at n 1500** iff measured and ``pct1_after`` is leak-free in both cells.
3. **Shrinks with n** (reported, decides nothing about the default): the control optimism is
   below slice 10's n 500 value in both cells (+0.0279 at phi 0, +0.0541 at phi 0.9).

Expectation written before the run: the control leaks at phi 0.9 only (about +0.015), the
default holds, and the leak shrinks with n.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.logistic_power import ARMS, CONTROL, DEFAULT_ARM, FEATURE_PHI, HORIZON, N_SPLITS, SE_MULT, summarize
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N_SAMPLES = 1500
SHOCK_PHIS = (0.0, 0.9)
SEEDS = tuple(range(9400, 9880))
SLICE10_CONTROL = {0.0: 0.0279, 0.9: 0.0541}
SLICE9_BOUND = 0.040

__all__ = ["ARMS", "CONTROL", "DEFAULT_ARM", "SEEDS", "SHOCK_PHIS", "call", "one_seed", "run_job", "summarize"]


def one_seed(shock_phi: float, seed: int, n: int = N_SAMPLES) -> dict[str, float]:
    ds = make_autocorrelated_null_dataset(n, HORIZON, FEATURE_PHI, 3, seed, shock_phi=shock_phi)
    model = make_sign_classifier("logistic")
    splitters = {name: PurgedKFold(N_SPLITS, ds.label_start_times, ds.label_end_times, **kw) for name, kw in ARMS.items()}
    splitters[CONTROL] = KFold(N_SPLITS, shuffle=False)
    acc = {}
    for name, cv in splitters.items():
        scores = [float((clone(model).fit(ds.X[tr], ds.y[tr]).predict(ds.X[te]) == ds.y[te]).mean())
                  for tr, te in cv.split(ds.X, ds.y)]
        acc[name] = float(np.mean(scores))
    return acc


def run_job(job: tuple[float, int]):
    phi, seed = job
    return phi, one_seed(phi, seed)


def call(cells: dict[float, dict[str, dict]]) -> dict:
    leaks = sum(c[CONTROL]["leak"] for c in cells.values())
    measured = leaks >= 1
    holds = measured and not any(c[DEFAULT_ARM]["leak"] for c in cells.values())
    shrinks = all(cells[phi][CONTROL]["optimism"] < SLICE10_CONTROL[phi] for phi in cells)
    bound = max(c[CONTROL]["optimism"] + SE_MULT * c[CONTROL]["se"] for c in cells.values())
    return {"measured": bool(measured), "default_holds": bool(holds), "shrinks_with_n": bool(shrinks),
            "control_leaks": int(leaks), "default_leaks": int(sum(c[DEFAULT_ARM]["leak"] for c in cells.values())),
            "control_upper_bound": float(bound)}
