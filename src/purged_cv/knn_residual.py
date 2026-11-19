"""Slice 12 protocol: is the purged k-NN residual at 25-row test folds real? (frozen 2026-11-20)

Slice 10's purged k-NN cells (n 500, 20 folds) sat at +0.0074 / +0.0078, 1.6 and 1.3 s.e. above
zero: not a leak under the 2-s.e. rule, but the largest purged k-NN optimism in slices 8-10, so a
small residual leak at short test folds could not be ruled out. This slice re-runs exactly those
two k-NN cells with twice slice 10's seeds. Written before any score of this grid was computed.

Grid: k-NN x shock_phi in {0, 0.9}; n 500; 20 folds; seeds 9900-10379 (480, disjoint from slices
4-11). Arms ``purge_only`` and ``pct1_after`` (package default), and the no-purge control.

Leak test per cell and arm: mean optimism more than 2 s.e. above zero.

Pre-stated calls:

1. **Reproduction**: the control must leak in both cells, else no call is made.
2. **Residual is real** iff the default (``pct1_after``) leaks in >= 1 cell. Then the README must
   say the package default leaves a measurable leak at 25-row test folds for k-NN, with its size;
   the default itself does not change in this slice (a larger embargo would need its own
   protocol).
3. Otherwise the default is **leak-free at this power** and slice 10's caveat is closed, with the
   largest default optimism + 2 s.e. reported as the bound.

Expectation written before the run: the residual is real at phi 0.9 (about +0.007).
"""

from __future__ import annotations

from purged_cv.logistic_power import CONTROL, DEFAULT_ARM, SE_MULT, one_seed, summarize

SHOCK_PHIS = (0.0, 0.9)
SEEDS = tuple(range(9900, 10380))

__all__ = ["CONTROL", "DEFAULT_ARM", "SEEDS", "SHOCK_PHIS", "call", "run_job", "summarize"]


def run_job(job: tuple[float, int]):
    phi, seed = job
    return phi, one_seed("knn", phi, seed)


def call(cells: dict[float, dict[str, dict]]) -> dict:
    reproduced = all(c[CONTROL]["leak"] for c in cells.values())
    leaks = sum(c[DEFAULT_ARM]["leak"] for c in cells.values())
    bound = max(c[DEFAULT_ARM]["optimism"] + SE_MULT * c[DEFAULT_ARM]["se"] for c in cells.values())
    return {"reproduced": bool(reproduced), "residual_real": bool(reproduced and leaks >= 1),
            "default_leaks": int(leaks), "purge_only_leaks": int(sum(c["purge_only"]["leak"] for c in cells.values())),
            "default_upper_bound": float(bound)}
