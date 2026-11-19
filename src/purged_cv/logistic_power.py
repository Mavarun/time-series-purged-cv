"""Slice 10 protocol: can a leak be seen for logistic regression at all? (frozen 2026-11-20)

Slice 9 could not test logistic regression: its no-purge control never leaked at n 1500
(+0.019 at 20 folds, shock phi 0.9, under 2 s.e. of 0.0107). Slice 8 found the k-NN leak
about four times larger at n 500 than at n 1500, and the s.e. falls with more seeds. This
slice gives the control its best chance: n 500, 20 folds (25-row test folds), and four times
the seeds. Written before any CV score of this grid was computed. A 2-seed smoke run (seeds
9100-9101) is allowed only to check that it runs; nothing may change after it.

Null: ``make_autocorrelated_null_dataset`` with ``feature_phi`` = 0.98, 3 features, horizon
20, n = 500. True accuracy is 0.5 for every arm.

Grid: learner in {logistic, knn} x shock_phi in {0, 0.9}; 20 folds; seeds 9100-9339 (240,
disjoint from slices 4-9). Arms ``purge_only`` and ``pct1_after`` (package default), plus the
``contiguous_no_purge`` control. The k-NN cells reproduce slice 8's n 500 / 20-fold cells on
fresh seeds (there: control +0.064 / +0.100, purged arms leak-free).

Leak test per cell and arm: mean optimism more than 2 s.e. above zero.

Pre-stated calls:

1. **Logistic testable** iff its control leaks in >= 1 of its 2 cells.
2. **Logistic default holds** iff it is testable and ``pct1_after`` is leak-free in both cells.
3. **Reproduction**: the k-NN control must leak in both cells and the k-NN default in none;
   if not, the grid is reported as failing its reproduction check and no logistic call is made.
4. If logistic is still not testable at this power, the README says so and stops calling it
   "untestable at 60 seeds": the bound is reported as the largest control optimism + 2 s.e.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N_SAMPLES = 500
LEARNERS = ("logistic", "knn")
N_SPLITS = 20
SHOCK_PHIS = (0.0, 0.9)
HORIZON = 20
FEATURE_PHI = 0.98
SEEDS = tuple(range(9100, 9340))
ARMS: dict[str, dict] = {"purge_only": {"embargo_pct": 0.0}, "pct1_after": {"embargo_pct": 0.01}}
CONTROL = "contiguous_no_purge"
DEFAULT_ARM = "pct1_after"
SE_MULT = 2.0


def one_seed(learner: str, shock_phi: float, seed: int) -> dict[str, float]:
    ds = make_autocorrelated_null_dataset(N_SAMPLES, HORIZON, FEATURE_PHI, 3, seed, shock_phi=shock_phi)
    model = make_sign_classifier(learner)
    splitters = {name: PurgedKFold(N_SPLITS, ds.label_start_times, ds.label_end_times, **kw) for name, kw in ARMS.items()}
    splitters[CONTROL] = KFold(N_SPLITS, shuffle=False)
    acc = {}
    for name, cv in splitters.items():
        scores = [float((clone(model).fit(ds.X[tr], ds.y[tr]).predict(ds.X[te]) == ds.y[te]).mean())
                  for tr, te in cv.split(ds.X, ds.y)]
        acc[name] = float(np.mean(scores))
    return acc


def run_job(job: tuple[str, float, int]):
    learner, phi, seed = job
    return learner, phi, one_seed(learner, phi, seed)


def summarize(results: list[dict[str, float]]) -> dict[str, dict]:
    out = {}
    for name in (*ARMS, CONTROL):
        opt = np.array([r[name] - 0.5 for r in results])
        se = float(opt.std(ddof=1) / np.sqrt(len(opt))) if len(opt) > 1 else float("nan")
        out[name] = {"optimism": float(opt.mean()), "se": se, "leak": bool(opt.mean() > SE_MULT * se)}
    return out


def call(cells: dict[tuple[str, float], dict[str, dict]]) -> dict:
    knn = [c for k, c in cells.items() if k[0] == "knn"]
    lg = [c for k, c in cells.items() if k[0] == "logistic"]
    reproduced = all(c[CONTROL]["leak"] for c in knn) and not any(c[DEFAULT_ARM]["leak"] for c in knn)
    testable = sum(c[CONTROL]["leak"] for c in lg) >= 1
    holds = testable and not any(c[DEFAULT_ARM]["leak"] for c in lg)
    bound = max(c[CONTROL]["optimism"] + SE_MULT * c[CONTROL]["se"] for c in lg)
    return {"reproduced": bool(reproduced),
            "logistic_testable": bool(reproduced and testable),
            "logistic_default_holds": bool(reproduced and holds),
            "logistic_control_leaks": int(sum(c[CONTROL]["leak"] for c in lg)),
            "logistic_default_leaks": int(sum(c[DEFAULT_ARM]["leak"] for c in lg)),
            "logistic_control_upper_bound": float(bound)}
