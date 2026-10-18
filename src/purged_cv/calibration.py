"""Calibration of selection-bias hurdles: false positives on nulls, power on edges.

For each seed a fresh AR(1) series is drawn, the default 40-rule grid is
run net of costs, and the best full-sample trial is tested four ways:

* ``dsr_raw``: DSR with N = 40 independent trials (null Sharpe variance);
* ``dsr_participation``: DSR with N = eigenvalue participation ratio;
* ``dsr_simulated``: DSR against the mean of correlated zero-skill max
  draws (implied effective N reported);
* ``max_p``: max-statistic p-value against the same simulated null.

A trial "passes" when DSR > ``dsr_level`` (0.95) or ``max_p < alpha``.
On ``phi = 0`` every pass is a false positive; on ``phi > 0`` the lag-1
momentum rule has a real edge, so the pass rate is power. Nothing is
fitted: the grid, costs and thresholds are fixed before any data is drawn,
and the trial correlation is a property of the rules, not of a label.
"""

from __future__ import annotations

import numpy as np

from purged_cv.selection import default_rule_grid, make_ar_returns, net_returns_from_positions, rule_positions
from purged_cv.sharpe import deflated_sharpe_for_selection, max_sharpe_pvalue

METHODS = ("dsr_raw", "dsr_participation", "dsr_simulated", "max_p")


def selection_tests_one_seed(
    phi: float,
    seed: int,
    cost_bps: float = 2.0,
    n: int = 2520,
    vol: float = 0.01,
    grid=None,
    n_sims: int = 4000,
) -> dict:
    """Run every hurdle on one simulated series; returns scores and effective N."""
    grid = grid or default_rule_grid()
    r = make_ar_returns(n, phi, vol, seed)
    P = rule_positions(r, grid)
    _, _, net = net_returns_from_positions(r, P, cost_bps)
    warm = max(L for L, _, _ in grid) + 1
    M = net[warm:]
    out = {"phi": phi, "seed": seed, "cost_bps": cost_bps}
    for mode in ("raw", "participation", "simulated"):
        rep = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode=mode, n_sims=n_sims, seed=seed)
        out[f"dsr_{mode}"] = rep.dsr
        out[f"n_eff_{mode}"] = rep.n_effective_trials
        out["best_trial"] = list(grid[rep.best_index])
    out["max_p"] = max_sharpe_pvalue(M, n_sims=n_sims, seed=seed)
    return out


def calibration_table(
    phis=(0.0, 0.05, 0.1),
    seeds=range(200),
    cost_bps=(0.0, 2.0),
    dsr_level: float = 0.95,
    alpha: float = 0.05,
    **kwargs,
) -> list[dict]:
    """Pass rates per (phi, cost) for each method, with binomial std errors."""
    rows = []
    for phi in phis:
        for c in cost_bps:
            runs = [selection_tests_one_seed(phi, s, cost_bps=c, **kwargs) for s in seeds]
            k = len(runs)
            row = {"phi": float(phi), "cost_bps": float(c), "n_seeds": k}
            for m in METHODS:
                vals = np.array([rr[m] for rr in runs])
                passed = vals < alpha if m == "max_p" else vals > dsr_level
                rate = float(passed.mean())
                row[m] = rate
                row[f"{m}_se"] = float(np.sqrt(rate * (1 - rate) / k))
            for mode in ("participation", "simulated"):
                row[f"mean_n_eff_{mode}"] = float(np.mean([rr[f"n_eff_{mode}"] for rr in runs]))
            row["frac_best_is_lag1_long"] = float(
                np.mean([rr["best_trial"][0] == 1 and rr["best_trial"][1] == 1 for rr in runs])
            )
            rows.append(row)
    return rows
