"""Stationary bootstrap and White's reality check for the best of correlated trials.

The slice-4 max-statistic p-value (``sharpe.max_sharpe_pvalue``) draws the
null Sharpes from ``N(0, C / (T - 1))``: Gaussian sampling error, iid bars.
Real returns cluster in volatility and have fat tails, and rule returns built
from them inherit serial dependence, so the iid-normal Sharpe variance can be
wrong in either direction. White's (2000) reality check replaces the Gaussian
null with a resampling one:

* resample whole rows of the ``T x N`` trial-return matrix with the
  **stationary bootstrap** of Politis & Romano (1994): blocks start at uniform
  random bars and have geometric lengths with mean ``mean_block``, wrapping
  around the end, so serial dependence and the cross-trial correlation are
  both kept and the resampled series is stationary;
* statistic ``V = max_j sqrt(T) * SR_j``; bootstrap draws
  ``V*_b = max_j sqrt(T) * (SR*_bj - SR_j)`` (each trial re-centred on its own
  full-sample Sharpe, i.e. the least-favourable null that every trial has
  zero true Sharpe);
* ``p = (1 + #{V*_b >= V}) / (1 + B)``.

``mean_block`` is fixed a priori (default 10 bars) rather than tuned on the
calibration seeds. Hansen's SPA refinement (dropping very poor trials from
the null) is not implemented.
"""

from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float = 10.0, seed: int | np.random.Generator = 0
) -> np.ndarray:
    """``(n_boot, n)`` row indices of Politis-Romano stationary-bootstrap samples."""
    if n < 1 or n_boot < 1:
        raise ValueError("n and n_boot must be >= 1")
    if mean_block < 1.0:
        raise ValueError("mean_block must be >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    p = 1.0 / float(mean_block)
    new = rng.random((n_boot, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    t = np.arange(n)
    # position of the most recent block start at or before t
    last = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    rows = np.arange(n_boot)[:, None]
    return (starts[rows, last] + (t - last)) % n


def _sharpes(M: np.ndarray) -> np.ndarray:
    """Per-column Sharpe along axis -2 (works on ``(T, N)`` and ``(B, T, N)``)."""
    mu = M.mean(axis=-2)
    sd = M.std(axis=-2, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)


def reality_check_draws(
    trial_returns: np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 10.0,
    seed: int = 0,
    chunk: int = 50,
) -> tuple[float, np.ndarray]:
    """Observed ``max_j sqrt(T) SR_j`` and its re-centred bootstrap null draws."""
    M = np.asarray(trial_returns, dtype=float)
    if M.ndim != 2 or M.shape[0] < 3:
        raise ValueError("trial_returns must be a (T >= 3, N) matrix")
    T = M.shape[0]
    sr = _sharpes(M)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot))
    for lo in range(0, int(n_boot), chunk):
        b = min(chunk, int(n_boot) - lo)
        idx = stationary_bootstrap_indices(T, b, mean_block, rng)
        draws[lo: lo + b] = np.sqrt(T) * (_sharpes(M[idx]) - sr).max(axis=1)
    return float(np.sqrt(T) * sr.max()), draws


def reality_check_pvalue(
    trial_returns: np.ndarray,
    n_boot: int = 1000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> float:
    """White's reality-check p-value for the best trial's Sharpe (stationary bootstrap)."""
    stat, draws = reality_check_draws(trial_returns, n_boot=n_boot, mean_block=mean_block, seed=seed)
    return float((1 + np.sum(draws >= stat)) / (1 + len(draws)))
