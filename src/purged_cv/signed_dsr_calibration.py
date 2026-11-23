"""Slice 14 protocol: does a mirror-aware trial count fix the liberal participation DSR? (frozen 2026-11-24)

Slice 6 found the participation-N DSR the only hurdle above 7% on short nulls (7.7% normal and
10.3% GARCH-t at 252 bars, 7.0% at 504 normal). Its docstring already names the cause it
suspects: the default grid holds every rule in both directions, the participation ratio counts
a rule and its mirror as one direction, yet ``max(x, -x)`` is a two-draw max.
``n_trials_mode="participation_signed"`` counts each mirrored direction twice.

Dev check (seeds 0-99, 252 / 504 bars, normal / GARCH-t nulls; not in the grid below): the
participation null rate was 3-10% and the signed one 0-2%. That is the only score seen.

Grid: the slice-6 setup (40-rule grid, 2 bps, null Sharpe variance, DSR > 0.95, max-stat p < 0.05,
4000 null draws) at n in {252, 504, 1260} x returns {normal, garch, regime} x phi in {0, 0.1};
300 fresh seeds 11000-11299 (slices 4-13 used 0-199, 5000-5299, 7000-10879).

Pre-stated calls:

1. **H1 (fixes the null):** the signed DSR's false-positive rate is <= 0.063 (5% + 1 binomial
   s.e. at 300 seeds) in all 6 null cells with n <= 504.
2. **H2 (keeps power):** in the 9 ``phi = 0.1`` cells, the signed DSR's power is >= the
   max-statistic test's minus 0.03 in >= 6 cells. The max-statistic test is the reference
   because slice 6 found it the hurdle with a clean null.
3. **Recommend** ``participation_signed`` over ``participation`` in the README iff H1 and H2
   both pass. The package default (``raw``) does not change either way.

Expectation written before the run: H1 passes; H2 fails, because doubling N on a 40-rule grid
should make the signed DSR at least as strict as the simulated one, which slice 6 found
conservative (0-1.7% null) at the cost of power.
"""

from __future__ import annotations

import numpy as np

from purged_cv.calibration import INNOVATIONS, MIN_SCORED_BARS
from purged_cv.selection import default_rule_grid, net_returns_from_positions, rule_positions
from purged_cv.sharpe import deflated_sharpe_for_selection, max_sharpe_pvalue

NS = (252, 504, 1260)
RETURNS = ("normal", "garch", "regime")
PHIS = (0.0, 0.1)
SEEDS = tuple(range(11000, 11300))
DEV_SEEDS = tuple(range(100))
COST_BPS = 2.0
VOL = 0.01
N_SIMS = 4000
DSR_LEVEL = 0.95
ALPHA = 0.05
NULL_CAP = 0.063
POWER_SLACK = 0.03
POWER_CELLS_NEEDED = 6
METHODS = ("dsr_participation", "dsr_signed", "dsr_simulated", "max_p")


def one_seed(n: int, returns: str, phi: float, seed: int) -> dict:
    grid = default_rule_grid()
    r = INNOVATIONS[returns](n, phi, VOL, seed)
    _, _, net = net_returns_from_positions(r, rule_positions(r, grid), COST_BPS)
    M = net[max(L for L, _, _ in grid) + 1:]
    if len(M) < MIN_SCORED_BARS:
        raise ValueError("too few scored bars")
    out = {"seed": seed}
    for mode, key in (("participation", "dsr_participation"), ("participation_signed", "dsr_signed"),
                      ("simulated", "dsr_simulated")):
        rep = deflated_sharpe_for_selection(M, sr_variance="null", n_trials_mode=mode, n_sims=N_SIMS, seed=seed)
        out[key] = rep.dsr
        out[f"n_eff_{key[4:]}"] = rep.n_effective_trials
    out["max_p"] = max_sharpe_pvalue(M, n_sims=N_SIMS, seed=seed)
    return out


def run_job(job):
    n, returns, phi, seed = job
    return (n, returns, phi), one_seed(n, returns, phi, seed)


def summarize(rows: list[dict]) -> dict:
    k = len(rows)
    cell = {"n_seeds": k}
    for m in METHODS:
        v = np.array([r[m] for r in rows])
        rate = float((v < ALPHA).mean() if m == "max_p" else (v > DSR_LEVEL).mean())
        cell[m] = rate
        cell[f"{m}_se"] = float(np.sqrt(rate * (1 - rate) / k))
    for key in ("participation", "signed", "simulated"):
        cell[f"mean_n_eff_{key}"] = float(np.mean([r[f"n_eff_{key}"] for r in rows]))
    return cell


def call(cells: dict) -> dict:
    """``cells`` maps (n, returns, phi) -> summarize() output."""
    null_short = {c: v for c, v in cells.items() if c[2] == 0.0 and c[0] <= 504}
    h1 = bool(len(null_short) == 6 and all(v["dsr_signed"] <= NULL_CAP for v in null_short.values()))
    power = {c: v for c, v in cells.items() if c[2] == 0.1}
    ok = [c for c, v in power.items() if v["dsr_signed"] >= v["max_p"] - POWER_SLACK]
    h2 = bool(len(power) == 9 and len(ok) >= POWER_CELLS_NEEDED)
    return {"H1": h1, "H2": h2, "power_cells_ok": len(ok), "recommend_signed": bool(h1 and h2)}
