"""Slice 8b protocol: what does the embargo cost when there *is* an edge? (frozen 2026-11-19)

Slice 7 and slice 8 only measure optimism on a null. An embargo also throws away the train
rows closest to each test fold, which on a real, persistent edge are the most informative
ones, so it could make CV *pessimistic*. Written before any score of this grid was computed.

Data (``make_signal_dataset``): slice 7's persistent features (``feature_phi`` = 0.98, 3
features) and AR(1) shocks (``shock_phi``), plus a real edge: the bar shock at t gets
``beta * X[t-1, 0]`` added, so the forward-sum sign label depends on the first feature.
``beta`` = 0.15 was fixed by checking that the plain sign rule 1[X[t, 0] > 0] scores ~0.6 on
long draws (a test pins > 0.58); no CV score was looked at to choose it.

Truth: a model fitted on the n study rows is scored on the next 5000 rows of the same path
(after a ``horizon``-bar gap), which is what CV is meant to estimate. Bias = CV - truth.

Grid: n = 500, folds in {5, 20}, shock_phi in {0, 0.9}, horizon 20, seeds 8100-8199
(disjoint from slices 4-8). Arms: slice 8's ``purge_only``, ``pct1_after`` and ``acf_both``.

Pre-stated call: the embargo is documented as *cheap on a real edge* iff, in every cell,
both ``pct1_after`` and ``acf_both`` have mean CV accuracy no more than 0.005 below
``purge_only`` (their cost), measured on the same seeds. Otherwise the README states the
cost per cell. Nothing in the package changes either way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from purged_cv.data import LabeledDataset
from purged_cv.fold_count_calibration import ARMS
from purged_cv.model import make_sign_classifier
from purged_cv.split import PurgedKFold

N = 500
FRESH = 5000
FOLDS = (5, 20)
SHOCK_PHIS = (0.0, 0.9)
HORIZON = 20
BETA = 0.15
FEATURE_PHI = 0.98
SEEDS = tuple(range(8100, 8200))
REF_ARM = "purge_only"
MAX_COST = 0.005


def make_signal_dataset(n_samples: int, horizon: int = HORIZON, beta: float = BETA, shock_phi: float = 0.0,
                        feature_phi: float = FEATURE_PHI, n_features: int = 3, seed: int = 0) -> LabeledDataset:
    """Persistent features, AR(1) shocks and a real edge ``beta * X[t-1, 0]`` in every bar shock."""
    if beta < 0 or not 0 <= shock_phi < 1 or not 0 <= feature_phi < 1:
        raise ValueError("need beta >= 0 and phis in [0, 1)")
    rng = np.random.default_rng(seed)
    m = n_samples + horizon + 1
    z = rng.normal(size=(m, n_features))
    X = np.empty_like(z)
    X[0] = z[0]
    s = np.sqrt(1.0 - feature_phi**2)
    for t in range(1, m):
        X[t] = feature_phi * X[t - 1] + s * z[t]
    e = rng.normal(size=m)
    if shock_phi > 0:
        innov = e * np.sqrt(1.0 - shock_phi**2)
        e = np.empty_like(innov)
        e[0] = innov[0] / np.sqrt(1.0 - shock_phi**2)
        for t in range(1, m):
            e[t] = shock_phi * e[t - 1] + innov[t]
    shocks = e.copy()
    shocks[1:] += beta * X[:-1, 0]
    csum = np.concatenate([[0.0], np.cumsum(shocks)])
    idx = np.arange(n_samples)
    fwd = csum[idx + 1 + horizon] - csum[idx + 1]
    return LabeledDataset(
        X=X[:n_samples], y=(fwd > 0).astype(int), forward_returns=fwd,
        label_start_times=pd.Series(idx, name="label_start"), label_end_times=pd.Series(idx + horizon, name="label_end"),
        feature_names=[f"ar_feature_{k}" for k in range(n_features)],
        meta={"source": "autocorrelated_signal", "horizon": horizon, "beta": beta, "shock_phi": shock_phi, "seed": seed},
    )


def one_seed(n_splits: int, shock_phi: float, seed: int) -> dict:
    full = make_signal_dataset(N + HORIZON + FRESH, shock_phi=shock_phi, seed=seed)
    X, y = full.X[:N], full.y[:N]
    fresh = slice(N + HORIZON, N + HORIZON + FRESH)
    model = make_sign_classifier("knn")
    truth = float((clone(model).fit(X, y).predict(full.X[fresh]) == full.y[fresh]).mean())
    starts, ends = full.label_start_times[:N], full.label_end_times[:N]
    acc = {}
    for name, kw in ARMS.items():
        cv = PurgedKFold(n_splits, starts, ends, **kw)
        acc[name] = float(np.mean([(clone(model).fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
                                   for tr, te in cv.split(X, y)]))
    return {"truth": truth, "accuracy": acc}


def run_job(job):
    k, phi, seed = job
    return k, phi, one_seed(k, phi, seed)


def summarize(results: list[dict]) -> dict:
    truth = np.array([r["truth"] for r in results])
    out = {"truth": float(truth.mean())}
    ref = np.array([r["accuracy"][REF_ARM] for r in results])
    for name in ARMS:
        a = np.array([r["accuracy"][name] for r in results])
        d = ref - a
        out[name] = {"cv": float(a.mean()), "bias": float((a - truth).mean()),
                     "bias_se": float((a - truth).std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan"),
                     "cost_vs_purge_only": float(d.mean()),
                     "cost_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")}
    return out


def call(cells: dict) -> dict:
    costs = {f"{k}_{phi}": {a: c[a]["cost_vs_purge_only"] for a in ("pct1_after", "acf_both")} for (k, phi), c in cells.items()}
    worst = max(v for c in costs.values() for v in c.values())
    return {"max_cost": float(worst), "cheap": bool(worst <= MAX_COST), "costs": costs}
