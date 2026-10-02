"""Probability of Backtest Overfitting (PBO) via CSCV.

Reference: Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J.
(2017). "The Probability of Backtest Overfitting." *Journal of Computational
Finance* 20(4). (SSRN 2326253, 2014.)

Combinatorially symmetric cross-validation (CSCV)
-------------------------------------------------
Given a ``T x N`` matrix of per-period returns for ``N`` strategy
configurations:

1. Cut the rows into ``S`` (even) contiguous blocks.
2. For each of the ``C(S, S/2)`` ways to pick half the blocks as in-sample
   (IS), the complement is out-of-sample (OOS).
3. Pick the configuration with the best IS metric, find its relative rank
   ``w = rank_OOS / (N + 1)`` among all configurations OOS, and record the
   logit ``lambda = ln(w / (1 - w))``.
4. ``PBO = P(lambda <= 0)``: the fraction of splits where the IS winner is at
   or below the OOS median.

The metric is a per-period Sharpe ratio (mean / std), computed from per-block
sufficient statistics so all ``C(S, S/2)`` splits are vectorised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy.stats import rankdata


@dataclass
class PBOResult:
    pbo: float
    logits: np.ndarray
    is_best_index: np.ndarray
    is_metric_of_best: np.ndarray
    oos_metric_of_best: np.ndarray
    prob_oos_loss: float
    degradation_slope: float
    n_splits: int
    n_partitions: int
    n_strategies: int

    def summary(self) -> dict:
        d = asdict(self)
        for key in ("logits", "is_best_index", "is_metric_of_best", "oos_metric_of_best"):
            d.pop(key)
        d["median_logit"] = float(np.median(self.logits))
        d["mean_is_sharpe_of_best"] = float(np.mean(self.is_metric_of_best))
        d["mean_oos_sharpe_of_best"] = float(np.mean(self.oos_metric_of_best))
        return d


def _block_stats(returns: np.ndarray, n_partitions: int):
    T = returns.shape[0] - returns.shape[0] % n_partitions
    blocks = returns[:T].reshape(n_partitions, T // n_partitions, returns.shape[1])
    counts = np.full(n_partitions, T // n_partitions, dtype=float)
    sums = blocks.sum(axis=1)
    sumsq = (blocks**2).sum(axis=1)
    return counts, sums, sumsq


def _sharpe_from_stats(count: float, s: np.ndarray, ss: np.ndarray) -> np.ndarray:
    mean = s / count
    var = (ss - count * mean**2) / max(count - 1.0, 1.0)
    std = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, mean / std, 0.0)
    return sr


def probability_of_backtest_overfitting(
    returns: np.ndarray,
    n_partitions: int = 16,
) -> PBOResult:
    """Estimate PBO with CSCV on a ``T x N`` matrix of strategy returns.

    Trailing rows that do not fill a whole block are dropped. Ties in OOS
    performance get average ranks.
    """
    M = np.asarray(returns, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns must be a T x N matrix with N >= 2")
    if n_partitions < 2 or n_partitions % 2:
        raise ValueError("n_partitions must be an even integer >= 2")
    if M.shape[0] < 2 * n_partitions:
        raise ValueError("need at least 2 rows per partition")
    if not np.isfinite(M).all():
        raise ValueError("returns must be finite")

    counts, sums, sumsq = _block_stats(M, n_partitions)
    n = M.shape[1]
    half = n_partitions // 2
    all_blocks = np.arange(n_partitions)

    logits, best_idx, is_best, oos_best = [], [], [], []
    for is_blocks in combinations(range(n_partitions), half):
        is_blocks = np.array(is_blocks)
        oos_blocks = np.setdiff1d(all_blocks, is_blocks)
        is_sr = _sharpe_from_stats(
            counts[is_blocks].sum(), sums[is_blocks].sum(0), sumsq[is_blocks].sum(0)
        )
        oos_sr = _sharpe_from_stats(
            counts[oos_blocks].sum(), sums[oos_blocks].sum(0), sumsq[oos_blocks].sum(0)
        )
        b = int(np.argmax(is_sr))
        rank = rankdata(oos_sr)[b]  # 1 = worst, n = best
        w = rank / (n + 1.0)
        logits.append(np.log(w / (1.0 - w)))
        best_idx.append(b)
        is_best.append(is_sr[b])
        oos_best.append(oos_sr[b])

    logits_a = np.asarray(logits)
    is_a = np.asarray(is_best)
    oos_a = np.asarray(oos_best)
    if np.ptp(is_a) > 0:
        slope = float(np.polyfit(is_a, oos_a, 1)[0])
    else:
        slope = float("nan")
    return PBOResult(
        pbo=float(np.mean(logits_a <= 0.0)),
        logits=logits_a,
        is_best_index=np.asarray(best_idx),
        is_metric_of_best=is_a,
        oos_metric_of_best=oos_a,
        prob_oos_loss=float(np.mean(oos_a < 0.0)),
        degradation_slope=slope,
        n_splits=len(logits_a),
        n_partitions=n_partitions,
        n_strategies=n,
    )
