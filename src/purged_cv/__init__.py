"""Purged and embargoed time-series cross-validation (Lopez de Prado style)."""

from purged_cv.split import PurgedKFold
from purged_cv.embargo import embargo_train_indices, purge_train_indices
from purged_cv.metrics import compare_naive_vs_purged_cv
from purged_cv.data import make_synthetic_ar_dataset, make_horizon_labels

__all__ = [
    "PurgedKFold",
    "embargo_train_indices",
    "purge_train_indices",
    "compare_naive_vs_purged_cv",
    "make_synthetic_ar_dataset",
    "make_horizon_labels",
]

__version__ = "0.1.0"
