"""Purged and embargoed time-series cross-validation (Lopez de Prado style)."""

from purged_cv.split import PurgedKFold
from purged_cv.embargo import embargo_train_indices, purge_train_indices

__all__ = ["PurgedKFold", "embargo_train_indices", "purge_train_indices"]

__version__ = "0.1.0"
