"""Slice 7 protocol: does the embargo remove the leak that purging leaves? (frozen 2026-11-10)

Written before the grid was run. Disclosure: while writing ``acf_embargo_bars`` its
output (not any CV score) was printed for 6 seeds (0-5) per cell below; those seeds are
outside the grid's seeds. A 2-seed smoke run of the script (seeds 7000-7001) was made to
check it runs; no arm, rule or threshold was changed after it.

Null: ``make_autocorrelated_null_dataset`` with persistent bar shocks (``shock_phi``),
n = 1500, ``feature_phi`` = 0.98, 3 features, k-NN, 5 contiguous folds. Features are
independent of labels, so every arm's true accuracy is 0.5; CV accuracy above 0.5 is leak.

Grid: shock_phi in {0, 0.5, 0.8, 0.9} x horizon in {5, 20}; seeds 7000-7099 (disjoint from
slices 4-6). Arms (purge is always on):

- ``purge_only``: no embargo;
- ``pct1_after``: the package default, ``embargo_pct = 0.01`` (15 bars) after the test fold;
- ``pct5_after``: ``embargo_pct = 0.05`` (75 bars) after;
- ``acf_after``: ``embargo_bars = "acf"`` after;
- ``acf_both``: ``embargo_bars = "acf"`` on both sides.

Per cell and arm: mean optimism (CV accuracy - 0.5) with its s.e. over seeds, and the mean
share of the default arm's train rows kept. An arm is *leak-free* in a cell when its mean
optimism is at most 2 s.e. above zero.

Pre-stated call: ``acf_both`` is documented as the recommended setting for persistent labels
only if (a) it is leak-free in all 8 cells, (b) the default ``pct1_after`` is *not*
leak-free in at least 2 of the 6 ``shock_phi > 0`` cells, and (c) on the 2 ``shock_phi = 0``
cells it keeps at least 95% of the default's train rows. The package default
(``embargo_pct = 0.01``, after) does not change in this slice either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from purged_cv.leakage import make_autocorrelated_null_dataset
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

SHOCK_PHIS = (0.0, 0.5, 0.8, 0.9)
HORIZONS = (5, 20)
SEEDS = tuple(range(7000, 7100))
N_SAMPLES = 1500
FEATURE_PHI = 0.98
N_SPLITS = 5
ARMS: dict[str, dict] = {
    "purge_only": {"embargo_pct": 0.0},
    "pct1_after": {"embargo_pct": 0.01},
    "pct5_after": {"embargo_pct": 0.05},
    "acf_after": {"embargo_bars": "acf"},
    "acf_both": {"embargo_bars": "acf", "embargo_side": "both"},
}
DEFAULT_ARM = "pct1_after"
CANDIDATE = "acf_both"
SE_MULT = 2.0
MIN_DEFAULT_FAILS = 2
MIN_KEEP_SHARE = 0.95


@dataclass
class SeedResult:
    accuracy: dict[str, float]
    train_rows: dict[str, int]
    embargo: dict[str, int]


def one_seed(shock_phi: float, horizon: int, seed: int, n_samples: int = N_SAMPLES) -> SeedResult:
    """CV accuracy, total train rows and embargo length per arm on one null draw."""
    ds = make_autocorrelated_null_dataset(n_samples, horizon, FEATURE_PHI, 3, seed, shock_phi=shock_phi)
    model = make_sign_classifier("knn")
    acc, rows, emb = {}, {}, {}
    for name, kw in ARMS.items():
        cv = PurgedKFold(N_SPLITS, ds.label_start_times, ds.label_end_times, **kw)
        scores, n_tr = [], 0
        for tr, te in cv.split(ds.X, ds.y):
            m = clone(model).fit(ds.X[tr], ds.y[tr])
            scores.append(float((m.predict(ds.X[te]) == ds.y[te]).mean()))
            n_tr += len(tr)
        acc[name], rows[name], emb[name] = float(np.mean(scores)), n_tr, int(cv.last_embargo_ or 0)
    return SeedResult(acc, rows, emb)


def summarize(results: list[SeedResult]) -> dict[str, dict]:
    """Per arm: mean optimism, its s.e., leak-free flag, kept share of default train rows, median embargo."""
    out = {}
    base = np.array([r.train_rows[DEFAULT_ARM] for r in results], float)
    for name in ARMS:
        opt = np.array([r.accuracy[name] - 0.5 for r in results])
        se = float(opt.std(ddof=1) / np.sqrt(len(opt))) if len(opt) > 1 else float("nan")
        keep = np.array([r.train_rows[name] for r in results], float) / base
        out[name] = {
            "optimism": float(opt.mean()),
            "se": se,
            "leak_free": bool(opt.mean() <= SE_MULT * se),
            "keep_vs_default": float(keep.mean()),
            "median_embargo": float(np.median([r.embargo[name] for r in results])),
        }
    return out


def call(cells: dict[tuple[float, int], dict[str, dict]]) -> dict:
    """The pre-stated recommendation rule over the 8 summarized cells."""
    a = all(c[CANDIDATE]["leak_free"] for c in cells.values())
    fails = sum(not c[DEFAULT_ARM]["leak_free"] for (phi, _h), c in cells.items() if phi > 0)
    b = fails >= MIN_DEFAULT_FAILS
    keep = [c[CANDIDATE]["keep_vs_default"] for (phi, _h), c in cells.items() if phi == 0]
    c_ok = bool(keep) and min(keep) >= MIN_KEEP_SHARE
    return {"candidate_leak_free_everywhere": a, "default_fails_in_persistent_cells": fails,
            "candidate_min_keep_at_phi0": min(keep) if keep else None, "recommend": bool(a and b and c_ok)}


def run_job(job: tuple[float, int, int]) -> tuple[float, int, SeedResult]:
    """``(shock_phi, horizon, seed) -> (shock_phi, horizon, SeedResult)``; picklable for process pools."""
    phi, h, seed = job
    return phi, h, one_seed(phi, h, seed)
