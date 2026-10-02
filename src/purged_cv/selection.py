"""Strategy-selection overfitting diagnostics with transaction costs.

Pipeline (all synthetic, seeded, offline)
-----------------------------------------
1. Returns: AR(1) daily returns ``r_t = phi r_{t-1} + sigma_e e_t``.
   ``phi = 0`` is the null (no strategy has an edge); ``phi > 0`` plants a
   genuine short-horizon momentum edge.
2. Trial grid: causal sign rules. For lookback ``L`` and direction
   ``d in {+1, -1}`` the position decided at the close of ``t`` is
   ``d * sign(mean(r_{t-L+1..t}))`` (zeroed when ``|z| < threshold``) and earns
   ``r_{t+1}``. No parameter is fitted to future data.
3. Costs: ``cost_bps`` per unit of absolute position change, charged on the
   bar the trade happens. All Sharpe numbers below are **net of costs**.
4. Diagnostics on the ``T x N`` net-return matrix:
   * full-sample best trial (the number a naive backtest would report);
   * DSR of that best trial, deflated by ``N`` trials, with both the
     cross-sectional and the null Sharpe variance (see
     :func:`purged_cv.sharpe.deflated_sharpe_for_selection`; the grid holds
     long and short mirror images, so the cross-sectional DSR over-deflates
     when a real edge exists);
   * PBO via CSCV;
   * walk-forward OOS: every ``step`` bars pick the trial with the best
     trailing ``train_window`` net Sharpe, hold it for the next ``step``
     bars; the stitched OOS position path is re-costed (switching trades are
     charged too).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from purged_cv.overfitting import probability_of_backtest_overfitting
from purged_cv.sharpe import deflated_sharpe_for_selection, sharpe_ratio


def make_ar_returns(n: int = 2520, phi: float = 0.0, vol: float = 0.01, seed: int = 0) -> np.ndarray:
    """Stationary AR(1) returns with unconditional std ``vol``."""
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must be in (-1, 1)")
    rng = np.random.default_rng(seed)
    sigma_e = vol * np.sqrt(1.0 - phi**2)
    e = rng.normal(0.0, sigma_e, size=n)
    r = np.empty(n)
    r[0] = rng.normal(0.0, vol)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return r


def default_rule_grid(
    lookbacks=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89),
    thresholds=(0.0, 0.5),
) -> list[tuple[int, int, float]]:
    """(lookback, direction, z-threshold) trials; 40 by default."""
    return [(L, d, z) for L in lookbacks for d in (1, -1) for z in thresholds]


def rule_positions(returns: np.ndarray, grid) -> np.ndarray:
    """``T x N`` positions; row ``t`` is held over ``r_{t}`` (decided at t-1)."""
    r = np.asarray(returns, dtype=float)
    T = len(r)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    vol = r.std(ddof=1) if T > 1 else 1.0
    P = np.zeros((T, len(grid)))
    for j, (L, d, z) in enumerate(grid):
        decided = np.zeros(T)
        idx = np.arange(L - 1, T)
        mean = (csum[idx + 1] - csum[idx + 1 - L]) / L
        zscore = mean / (vol / np.sqrt(L))
        sig = np.sign(mean) * (np.abs(zscore) >= z)
        decided[idx] = d * sig
        P[1:, j] = decided[:-1]  # decision at close t earns r_{t+1}
    return P


def net_returns_from_positions(
    returns: np.ndarray, positions: np.ndarray, cost_bps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gross, cost, and net returns; cost = cost_bps/1e4 * |delta position|."""
    r = np.asarray(returns, dtype=float)
    P = np.asarray(positions, dtype=float)
    r2 = r[:, None] if P.ndim == 2 else r
    gross = P * r2
    prev = np.zeros_like(P)
    prev[1:] = P[:-1]
    costs = np.abs(P - prev) * cost_bps / 1e4
    return gross, costs, gross - costs


def walk_forward_select(
    returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float,
    train_window: int = 504,
    step: int = 63,
) -> dict:
    """Rolling selection by trailing net Sharpe; returns OOS net series."""
    _, _, net = net_returns_from_positions(returns, positions, cost_bps)
    T = len(returns)
    if train_window >= T:
        raise ValueError("train_window must be shorter than the series")
    oos_pos = np.zeros(T)
    chosen = []
    for start in range(train_window, T, step):
        window = net[start - train_window : start]
        srs = window.mean(0) / np.maximum(window.std(0, ddof=1), 1e-12)
        j = int(np.argmax(srs))
        chosen.append(j)
        oos_pos[start : start + step] = positions[start : start + step, j]
    g, c, n = net_returns_from_positions(returns, oos_pos, cost_bps)
    sl = slice(train_window, T)
    return {
        "oos_net": n[sl],
        "oos_gross": g[sl],
        "oos_costs": c[sl],
        "chosen": np.asarray(chosen),
    }


@dataclass
class SelectionDiagnostics:
    phi: float
    cost_bps: float
    n_obs: int
    n_trials: int
    best_trial: tuple
    is_best_gross_sharpe_ann: float
    is_best_net_sharpe_ann: float
    psr_vs_zero: float
    dsr_cross_sectional: float
    dsr_null: float
    pbo: float
    pbo_prob_oos_loss: float
    wf_oos_net_sharpe_ann: float
    wf_oos_gross_sharpe_ann: float
    wf_oos_cost_drag_bps_per_day: float
    wf_n_obs: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best_trial"] = list(self.best_trial)
        return d


def run_selection_diagnostics(
    n: int = 2520,
    phi: float = 0.0,
    vol: float = 0.01,
    cost_bps: float = 5.0,
    seed: int = 0,
    grid=None,
    n_partitions: int = 16,
    train_window: int = 504,
    step: int = 63,
) -> SelectionDiagnostics:
    """Run the full pipeline (see module docstring)."""
    grid = grid or default_rule_grid()
    r = make_ar_returns(n, phi, vol, seed)
    P = rule_positions(r, grid)
    gross, _, net = net_returns_from_positions(r, P, cost_bps)
    # drop the warm-up rows where the longest lookback has no signal yet
    warm = max(L for L, _, _ in grid) + 1
    rep = deflated_sharpe_for_selection(net[warm:], sr_variance="cross_sectional")
    rep_null = deflated_sharpe_for_selection(net[warm:], sr_variance="null")
    pbo = probability_of_backtest_overfitting(net[warm:], n_partitions=n_partitions)
    wf = walk_forward_select(r, P, cost_bps, train_window, step)
    ann = np.sqrt(252.0)
    return SelectionDiagnostics(
        phi=phi,
        cost_bps=cost_bps,
        n_obs=int(n - warm),
        n_trials=len(grid),
        best_trial=tuple(grid[rep.best_index]),
        is_best_gross_sharpe_ann=float(sharpe_ratio(gross[warm:, rep.best_index]) * ann),
        is_best_net_sharpe_ann=rep.best_sharpe_annualised,
        psr_vs_zero=rep.psr_vs_zero,
        dsr_cross_sectional=rep.dsr,
        dsr_null=rep_null.dsr,
        pbo=pbo.pbo,
        pbo_prob_oos_loss=pbo.prob_oos_loss,
        wf_oos_net_sharpe_ann=float(sharpe_ratio(wf["oos_net"]) * ann),
        wf_oos_gross_sharpe_ann=float(sharpe_ratio(wf["oos_gross"]) * ann),
        wf_oos_cost_drag_bps_per_day=float(wf["oos_costs"].mean() * 1e4),
        wf_n_obs=int(len(wf["oos_net"])),
    )
