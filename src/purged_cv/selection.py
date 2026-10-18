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
   * the same DSR against a correlation-aware hurdle (simulated effective
     N) and a max-statistic p-value (see :mod:`purged_cv.calibration`);
   * PBO via CSCV;
   * walk-forward OOS: every ``step`` bars pick the trial with the best
     trailing ``train_window`` net Sharpe, hold it for the next ``step``
     bars; the stitched OOS position path is re-costed (switching trades are
     charged too).
   * enhanced selectors (optional): James-Stein shrinkage of trailing Sharpes
     toward the cross-sectional mean, and a turnover penalty
     ``Sharpe - turnover_penalty * mean(|delta position|)`` so noisy
     high-turnover winners are less likely to be picked; optional
     square-root impact costs on top of linear bps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from purged_cv.overfitting import probability_of_backtest_overfitting
from purged_cv.sharpe import deflated_sharpe_for_selection, max_sharpe_pvalue, sharpe_ratio


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


def expanding_vol(returns: np.ndarray, min_periods: int = 20) -> np.ndarray:
    """Causal per-bar std: ``std(r[:t+1])`` with a warm-up floor at ``min_periods``.

    Replaces the old full-sample return std used to z-score lookback means
    (a mild look-ahead). Bars before ``min_periods`` reuse the std of the
    first ``min_periods`` observations so early signals stay defined.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    out = np.empty(T)
    if T == 0:
        return out
    ss = 0.0
    s = 0.0
    for t in range(T):
        s += r[t]
        ss += r[t] * r[t]
        n = t + 1
        if n >= 2:
            var = (ss - s * s / n) / (n - 1)
            out[t] = float(np.sqrt(max(var, 0.0)))
        else:
            out[t] = 0.0
    floor = out[min(min_periods, T) - 1]
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.std(r, ddof=1)) if T > 1 else 1.0
    out[: min(min_periods, T)] = floor
    out[out <= 0] = floor if floor > 0 else 1.0
    return out


def rule_positions(
    returns: np.ndarray, grid, *, vol_mode: str = "expanding"
) -> np.ndarray:
    """``T x N`` positions; row ``t`` is held over ``r_{t}`` (decided at t-1).

    ``vol_mode``:
      * ``expanding`` (default) — causal expanding std for z-thresholds.
      * ``full_sample`` — previous behaviour (look-ahead; kept for ablation).
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    if vol_mode == "expanding":
        vol_path = expanding_vol(r)
    elif vol_mode == "full_sample":
        vol_path = np.full(T, r.std(ddof=1) if T > 1 else 1.0)
    else:
        raise ValueError(f"unknown vol_mode: {vol_mode!r}")
    P = np.zeros((T, len(grid)))
    for j, (L, d, z) in enumerate(grid):
        decided = np.zeros(T)
        idx = np.arange(L - 1, T)
        mean = (csum[idx + 1] - csum[idx + 1 - L]) / L
        # vol at decision time idx (close of bar), already causal under expanding
        zscore = mean / (vol_path[idx] / np.sqrt(L))
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



def net_returns_with_impact(
    returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float,
    impact_coef: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gross, cost, and net returns with optional square-root impact.

    Linear cost is ``cost_bps/1e4 * |delta position|``. When ``impact_coef`` > 0
    an extra ``impact_coef * sqrt(|delta position|)`` term (in return units) is
    charged on the same bar — a simple Almgren-Chriss-style temporary impact
    proxy that penalises large switches more than linear bps alone.
    """
    gross, linear, _ = net_returns_from_positions(returns, positions, cost_bps)
    P = np.asarray(positions, dtype=float)
    prev = np.zeros_like(P)
    prev[1:] = P[:-1]
    delta = np.abs(P - prev)
    impact = impact_coef * np.sqrt(delta)
    costs = linear + impact
    return gross, costs, gross - costs


def empirical_bayes_sharpes(net_window: np.ndarray) -> np.ndarray:
    """Normal-normal posterior means of per-period Sharpes (see ``eb_sharpe``)."""
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import skew as _skew

    w = np.asarray(net_window, dtype=float)
    T = w.shape[0]
    sd = w.std(0, ddof=1)
    live = sd > 0
    srs = np.zeros(w.shape[1])
    srs[live] = w[:, live].mean(0) / sd[live]
    g3 = np.zeros(w.shape[1])
    g4 = np.full(w.shape[1], 3.0)
    if live.any():
        g3[live] = _skew(w[:, live], axis=0)
        g4[live] = _kurt(w[:, live], axis=0, fisher=False)
    se2 = np.maximum(1.0 - g3 * srs + (g4 - 1.0) / 4.0 * srs**2, 1e-12) / (T - 1.0)
    se2[~live] = np.inf  # a never-trading rule carries no evidence
    m = float(srs[live].mean()) if live.any() else 0.0
    finite = np.isfinite(se2)
    tau2 = max(float(srs[live].var(ddof=1)) - float(se2[finite].mean()), 0.0) if live.sum() > 1 else 0.0
    weight = np.where(finite, tau2 / (tau2 + np.where(finite, se2, 1.0)), 0.0)
    post = m + weight * (srs - m)
    # deterministic tie-break by raw Sharpe when the posterior is flat
    return post + 1e-9 * srs


def trailing_scores(
    net_window: np.ndarray,
    positions_window: np.ndarray | None = None,
    *,
    mode: str = "sharpe",
    shrink_intensity: float = 0.0,
    turnover_penalty: float = 0.0,
) -> np.ndarray:
    """Per-trial score used by walk-forward selection.

    Modes
    -----
    sharpe
        trailing net Sharpe (status quo).
    shrunk_sharpe
        James-Stein shrink each trial's Sharpe toward the cross-sectional mean:
        ``(1 - intensity) * SR_j + intensity * mean(SR)``. ``intensity`` in
        [0, 1]; 0 recovers plain Sharpe. Because every trial gets the same
        intensity this is a positive affine map of the Sharpes, so for
        ``intensity < 1`` it **never changes the argmax** (walk-forward picks
        are identical to ``sharpe``); it only matters when combined with the
        turnover penalty, where it rescales Sharpe against the penalty. Use
        ``eb_sharpe`` for shrinkage that can re-rank trials.
    sharpe_minus_turnover
        ``SR_j - turnover_penalty * mean(|delta position_j|)`` over the window.
        Requires ``positions_window``.
    shrunk_sharpe_minus_turnover
        shrink first, then apply the turnover penalty.
    eb_sharpe
        empirical-Bayes (normal-normal) posterior mean of each trial's Sharpe
        with a **per-trial** standard error (Mertens / Lo, using the
        window's skew and kurtosis, so rarely-trading, fat-tailed rules get a
        larger error). Prior variance ``tau^2 = max(var(SR) - mean(se^2), 0)``
        is estimated from the window's own cross-section; trial ``j`` keeps
        weight ``tau^2 / (tau^2 + se_j^2)``. Because weights differ by trial
        this *can* re-rank trials, unlike ``shrunk_sharpe``. When no
        dispersion beyond noise is detected (``tau^2 = 0``) every posterior
        equals the mean and ties are broken by raw Sharpe.
    eb_sharpe_minus_turnover
        ``eb_sharpe`` then the turnover penalty.
    """
    if mode not in {
        "sharpe",
        "shrunk_sharpe",
        "sharpe_minus_turnover",
        "shrunk_sharpe_minus_turnover",
        "eb_sharpe",
        "eb_sharpe_minus_turnover",
    }:
        raise ValueError(f"unknown walk-forward score mode: {mode!r}")
    w = np.asarray(net_window, dtype=float)
    if w.ndim != 2:
        raise ValueError("net_window must be T x N")
    srs = w.mean(0) / np.maximum(w.std(0, ddof=1), 1e-12)
    if mode in ("shrunk_sharpe", "shrunk_sharpe_minus_turnover"):
        intensity = float(np.clip(shrink_intensity, 0.0, 1.0))
        srs = (1.0 - intensity) * srs + intensity * float(srs.mean())
    if mode in ("eb_sharpe", "eb_sharpe_minus_turnover"):
        srs = empirical_bayes_sharpes(w)
    if mode in ("sharpe_minus_turnover", "shrunk_sharpe_minus_turnover", "eb_sharpe_minus_turnover"):
        if positions_window is None:
            raise ValueError("positions_window required for turnover-penalised modes")
        P = np.asarray(positions_window, dtype=float)
        prev = np.zeros_like(P)
        prev[1:] = P[:-1]
        turnover = np.abs(P - prev).mean(0)
        srs = srs - float(turnover_penalty) * turnover
    return srs


def walk_forward_select(
    returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float,
    train_window: int = 504,
    step: int = 63,
    *,
    mode: str = "sharpe",
    shrink_intensity: float = 0.5,
    turnover_penalty: float = 5.0,
    impact_coef: float = 0.0,
) -> dict:
    """Rolling selection by a trailing score; returns OOS net series.

    Default ``mode="sharpe"`` preserves the original behaviour. Other modes
    use :func:`trailing_scores`. Costs on the stitched OOS path use
    :func:`net_returns_with_impact` so an impact coefficient can be studied
    without changing the linear-bps baseline.
    """
    _, _, net = net_returns_with_impact(returns, positions, cost_bps, impact_coef)
    T = len(returns)
    if train_window >= T:
        raise ValueError("train_window must be shorter than the series")
    oos_pos = np.zeros(T)
    chosen = []
    for start in range(train_window, T, step):
        slw = slice(start - train_window, start)
        srs = trailing_scores(
            net[slw],
            positions[slw],
            mode=mode,
            shrink_intensity=shrink_intensity,
            turnover_penalty=turnover_penalty,
        )
        j = int(np.argmax(srs))
        chosen.append(j)
        oos_pos[start : start + step] = positions[start : start + step, j]
    g, c, n = net_returns_with_impact(returns, oos_pos, cost_bps, impact_coef)
    sl = slice(train_window, T)
    return {
        "oos_net": n[sl],
        "oos_gross": g[sl],
        "oos_costs": c[sl],
        "chosen": np.asarray(chosen),
        "mode": mode,
        "shrink_intensity": float(shrink_intensity),
        "turnover_penalty": float(turnover_penalty),
        "impact_coef": float(impact_coef),
        "n_switches": int(np.sum(np.diff(chosen) != 0)) if len(chosen) > 1 else 0,
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
    dsr_null_simulated: float = float("nan")
    n_effective_trials_simulated: float = float("nan")
    max_stat_pvalue: float = float("nan")

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
    rep_sim = deflated_sharpe_for_selection(
        net[warm:], sr_variance="null", n_trials_mode="simulated", seed=seed
    )
    max_p = max_sharpe_pvalue(net[warm:], n_sims=10000, seed=seed)
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
        dsr_null_simulated=rep_sim.dsr,
        n_effective_trials_simulated=rep_sim.n_effective_trials,
        max_stat_pvalue=max_p,
    )


def compare_walk_forward_modes(
    returns: np.ndarray,
    positions: np.ndarray,
    cost_bps: float,
    train_window: int = 504,
    step: int = 63,
    impact_coef: float = 0.0,
    shrink_intensity: float = 0.5,
    turnover_penalty: float = 5.0,
) -> dict:
    """Walk-forward OOS net Sharpe for each selector mode (same path costing)."""
    modes = (
        "sharpe",
        "shrunk_sharpe",
        "sharpe_minus_turnover",
        "shrunk_sharpe_minus_turnover",
        "eb_sharpe",
        "eb_sharpe_minus_turnover",
    )
    ann = np.sqrt(252.0)
    out = {}
    from purged_cv.sharpe import sharpe_ratio

    for mode in modes:
        wf = walk_forward_select(
            returns,
            positions,
            cost_bps,
            train_window,
            step,
            mode=mode,
            shrink_intensity=shrink_intensity,
            turnover_penalty=turnover_penalty,
            impact_coef=impact_coef,
        )
        out[mode] = {
            "chosen": [int(j) for j in wf["chosen"]],
            "oos_net_sharpe_ann": float(sharpe_ratio(wf["oos_net"]) * ann),
            "oos_gross_sharpe_ann": float(sharpe_ratio(wf["oos_gross"]) * ann),
            "oos_cost_drag_bps_per_day": float(wf["oos_costs"].mean() * 1e4),
            "n_switches": int(wf["n_switches"]),
            "n_obs": int(len(wf["oos_net"])),
        }
    return out


def run_selector_ablation(
    n: int = 2520,
    phi: float = 0.0,
    vol: float = 0.01,
    cost_bps: float = 5.0,
    seed: int = 0,
    grid=None,
    train_window: int = 504,
    step: int = 63,
    impact_coefs: tuple[float, ...] = (0.0, 0.0005, 0.001),
    shrink_intensity: float = 0.5,
    turnover_penalty: float = 5.0,
) -> dict:
    """Null and edge ablations of enhanced walk-forward selectors + impact."""
    grid = grid or default_rule_grid()
    r = make_ar_returns(n, phi, vol, seed)
    P = rule_positions(r, grid)
    by_impact = {}
    for ic in impact_coefs:
        by_impact[str(ic)] = compare_walk_forward_modes(
            r,
            P,
            cost_bps,
            train_window,
            step,
            impact_coef=ic,
            shrink_intensity=shrink_intensity,
            turnover_penalty=turnover_penalty,
        )
    return {
        "phi": phi,
        "cost_bps": cost_bps,
        "seed": seed,
        "n": n,
        "n_trials": len(grid),
        "shrink_intensity": shrink_intensity,
        "turnover_penalty": turnover_penalty,
        "by_impact": by_impact,
    }
