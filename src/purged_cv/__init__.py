"""Purged and embargoed time-series cross-validation (Lopez de Prado style)."""

from purged_cv.split import PurgedKFold
from purged_cv.cpcv import CombinatorialPurgedKFold
from purged_cv.embargo import embargo_train_indices, purge_train_indices
from purged_cv.overfitting import probability_of_backtest_overfitting
from purged_cv.sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from purged_cv.metrics import compare_naive_vs_purged_cv
from purged_cv.data import make_synthetic_ar_dataset, make_horizon_labels
from purged_cv.selection import (
    expanding_vol,
    net_returns_with_impact,
    run_selector_ablation,
    trailing_scores,
    walk_forward_select,
)

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "embargo_train_indices",
    "purge_train_indices",
    "compare_naive_vs_purged_cv",
    "probability_of_backtest_overfitting",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "make_synthetic_ar_dataset",
    "make_horizon_labels",
    "expanding_vol",
    "net_returns_with_impact",
    "run_selector_ablation",
    "trailing_scores",
    "walk_forward_select",
]

__version__ = "0.1.0"
