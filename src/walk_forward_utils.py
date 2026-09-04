"""Utilities for efficient leakage-safe walk-forward model retraining."""
from __future__ import annotations


def retrain_dates(n_rows: int, min_train: int, step: int, retrain_every: int):
    """Yield (start, end) prediction windows while retraining periodically.

    Each model is fit only on rows strictly before ``start``. Predictions after
    the first fit reuse that model until the next retraining date. This keeps
    the procedure genuinely out-of-sample while avoiding a model fit for every
    small prediction window.
    """
    if step <= 0 or retrain_every <= 0:
        raise ValueError("step and retrain_every must be positive")
    i = min_train
    while i < n_rows:
        end = min(i + step, n_rows)
        yield i, end
        i += step
