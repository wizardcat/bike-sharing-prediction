# metrics.py — pure metric helpers shared by training and evaluation.
#
# The important rule here: every metric is computed from non-negative *count*
# predictions obtained with the same inverse transform (``to_counts``). This
# keeps validation and test numbers comparable.

from __future__ import annotations

import numpy as np


def to_counts(pred_log: np.ndarray) -> np.ndarray:
    """Invert ``log1p`` prediction and clip negative counts to 0."""
    return np.clip(np.expm1(np.asarray(pred_log, dtype=float)), 0.0, None)


def rmsle_from_counts(y_true_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """Root mean squared log error of non-negative count predictions.

    Equals ``sqrt(mean((log1p(y_true) - log1p(clip(y_pred, 0))) ** 2))``, i.e.
    RMSE on the ``log1p`` scale with predictions clipped the same way they are
    clipped before being served.
    """
    y_true = np.log1p(np.asarray(y_true_counts, dtype=float))
    y_pred = np.log1p(np.clip(np.asarray(pred_counts, dtype=float), 0.0, None))
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def log_mae_from_counts(y_true_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """Mean absolute error on the ``log1p`` scale (secondary metric)."""
    y_true = np.log1p(np.asarray(y_true_counts, dtype=float))
    y_pred = np.log1p(np.clip(np.asarray(pred_counts, dtype=float), 0.0, None))
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_counts(y_true_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """RMSE in rentals per hour (interpretable secondary metric)."""
    y_true = np.asarray(y_true_counts, dtype=float)
    y_pred = np.asarray(pred_counts, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_counts(y_true_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """MAE in rentals per hour (interpretable secondary metric)."""
    y_true = np.asarray(y_true_counts, dtype=float)
    y_pred = np.asarray(pred_counts, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
