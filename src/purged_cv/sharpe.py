"""Probabilistic and Deflated Sharpe ratios.

References
----------
* Bailey, D. H., López de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk* 15(2). (PSR, minimum track record length.)
* Bailey, D. H., López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  *Journal of Portfolio Management* 40(5).

All Sharpe ratios here are **per period** (not annualised) unless a function
says otherwise; ``kurtosis`` is the raw (non-excess) kurtosis, 3 for normal.

PSR(SR*) = Phi( (SR - SR*) sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) )

Expected maximum Sharpe of N independent zero-skill trials with cross-trial
Sharpe variance V (false-strategy theorem):

E[max SR] ~= sqrt(V) [ (1 - gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) ]

DSR = PSR(E[max SR]): the probability the selected strategy's true Sharpe
exceeds what the best of N skill-less trials would show by luck.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import kurtosis as _kurt
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_GAMMA = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-period Sharpe ratio, mean / sample std (ddof=1)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def sharpe_std_error(sr: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """Std error of an estimated per-period Sharpe (Mertens / Lo, non-normal)."""
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1.0)
    return float(np.sqrt(max(var, 0.0)))


def probabilistic_sharpe_ratio(
    sr: float,
    n_obs: int,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true SR > sr_benchmark) given an observed per-period ``sr``."""
    se = sharpe_std_error(sr, n_obs, skew, kurtosis)
    if se == 0:
        return float(sr > sr_benchmark)
    return float(norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, trials_sr_variance: float) -> float:
    """Expected max per-period Sharpe among ``n_trials`` zero-skill trials."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    if trials_sr_variance < 0:
        raise ValueError("trials_sr_variance must be >= 0")
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(trials_sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr: float,
    n_obs: int,
    n_trials: int,
    trials_sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR = PSR with benchmark E[max SR] over ``n_trials`` (per-period units)."""
    sr0 = expected_max_sharpe(n_trials, trials_sr_variance)
    return probabilistic_sharpe_ratio(sr, n_obs, sr0, skew, kurtosis)


def min_track_record_length(
    sr: float,
    sr_benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for PSR(sr_benchmark) >= ``confidence``."""
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(confidence)
    return float(1.0 + (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) * (z / (sr - sr_benchmark)) ** 2)


def trial_correlation(trial_returns: np.ndarray) -> np.ndarray:
    """``N x N`` Pearson correlation of trial return columns.

    Constant columns (e.g. a rule that never trades) get zero off-diagonal
    correlation and a unit diagonal so the matrix stays a valid correlation.
    """
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 1:
        raise ValueError("trial_returns must be T x N")
    sd = M.std(0, ddof=1)
    Z = np.zeros_like(M)
    live = sd > 0
    Z[:, live] = (M[:, live] - M[:, live].mean(0)) / sd[live]
    C = (Z.T @ Z) / (M.shape[0] - 1.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def effective_trials_participation(trial_returns: np.ndarray) -> float:
    """Eigenvalue participation ratio ``(sum l)^2 / sum l^2`` of the trial correlation.

    Counts the number of roughly independent *directions* spanned by the
    trials: N for uncorrelated trials, 1 for perfectly (anti-)correlated
    ones. It is a dimension count, not a max-statistic count: a rule and its
    exact mirror image span one direction, yet ``max(x, -x) = |x|`` is a
    larger draw than the max of one trial, so this measure understates the
    selection hurdle on mirrored grids (anti-conservative).
    """
    C = trial_correlation(trial_returns)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    return float(lam.sum() ** 2 / np.sum(lam**2))


def null_max_sharpe_draws(
    trial_returns: np.ndarray,
    trials_sr_variance: float | None = None,
    n_sims: int = 4000,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo draws of the best per-period Sharpe among zero-skill trials.

    Under H0 every trial has true Sharpe 0 and the estimated Sharpes are
    jointly ~ ``N(0, V * C)`` with ``C`` the trial correlation (Sharpe
    estimates of near-zero-Sharpe series inherit the return correlation) and
    ``V`` the per-trial sampling variance (default ``1 / (T - 1)``). Unlike
    the closed-form E[max SR], this keeps the grid's correlation and mirror
    structure, so correlated trials are not counted as independent.
    """
    M = np.asarray(trial_returns, dtype=float)
    T, N = M.shape
    var = 1.0 / (T - 1.0) if trials_sr_variance is None else float(trials_sr_variance)
    if var < 0:
        raise ValueError("trials_sr_variance must be >= 0")
    C = trial_correlation(M)
    w, V = np.linalg.eigh(C)
    L = V * np.sqrt(np.clip(w, 0.0, None))
    Z = np.random.default_rng(seed).standard_normal((int(n_sims), N)) @ L.T
    return Z.max(1) * np.sqrt(var)


def _expected_max_sharpe_continuous(n: float, var: float) -> float:
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def implied_effective_trials(expected_max: float, trials_sr_variance: float) -> float:
    """Invert the false-strategy formula: the N whose E[max SR] equals ``expected_max``.

    Treats N as continuous (bisection on log N). Returns 1.0 when the
    target is not positive (no selection inflation).
    """
    if trials_sr_variance <= 0:
        raise ValueError("trials_sr_variance must be > 0")
    if expected_max <= 0:
        return 1.0
    lo, hi = np.log(1.0 + 1e-9), np.log(1e12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _expected_max_sharpe_continuous(np.exp(mid), trials_sr_variance) < expected_max:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


@dataclass
class SelectionReport:
    """DSR diagnostics for picking the best column of a ``T x N`` trial matrix."""

    best_index: int
    n_trials: int
    n_obs: int
    best_sharpe: float
    best_sharpe_annualised: float
    trials_sr_std: float
    expected_max_sharpe: float
    psr_vs_zero: float
    dsr: float
    skew: float
    kurtosis: float

    def to_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe_for_selection(
    trial_returns: np.ndarray,
    periods_per_year: float = 252.0,
    sr_variance: str = "cross_sectional",
) -> SelectionReport:
    """Select the max-Sharpe trial and deflate it by the number of trials.

    ``sr_variance`` chooses the variance used in E[max SR]:

    * ``"cross_sectional"`` (Bailey & López de Prado 2014): sample variance of
      the trials' per-period Sharpes. Appropriate when trials are a broad
      search over unrelated ideas. If the grid contains a genuine edge *and*
      its mirror image (e.g. long/short versions of one rule), that dispersion
      is real skill, not luck, and this choice over-deflates sharply.
    * ``"null"``: sampling variance of a zero-skill Sharpe estimate,
      ``1 / (T - 1)``. Asks "could the best of N skill-less trials look this
      good?" without letting real dispersion inflate the hurdle.

    Skew/kurtosis come from the selected series. Trials are treated as
    independent; correlated trials overstate N, which is conservative.
    """
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("trial_returns must be T x N with N >= 2")
    srs = np.array([sharpe_ratio(M[:, j]) for j in range(M.shape[1])])
    b = int(np.argmax(srs))
    r = M[:, b]
    g3 = float(_skew(r))
    g4 = float(_kurt(r, fisher=False))
    n = M.shape[0]
    if sr_variance == "cross_sectional":
        var_sr = float(srs.var(ddof=1))
    elif sr_variance == "null":
        var_sr = 1.0 / (n - 1.0)
    else:
        raise ValueError("sr_variance must be 'cross_sectional' or 'null'")
    return SelectionReport(
        best_index=b,
        n_trials=M.shape[1],
        n_obs=n,
        best_sharpe=float(srs[b]),
        best_sharpe_annualised=float(srs[b] * np.sqrt(periods_per_year)),
        trials_sr_std=float(np.sqrt(var_sr)),
        expected_max_sharpe=expected_max_sharpe(M.shape[1], var_sr),
        psr_vs_zero=probabilistic_sharpe_ratio(srs[b], n, 0.0, g3, g4),
        dsr=deflated_sharpe_ratio(srs[b], n, M.shape[1], var_sr, g3, g4),
        skew=g3,
        kurtosis=g4,
    )
