import numpy as np
import pytest

from metrics import (
    log_mae_from_counts,
    mae_counts,
    rmse_counts,
    rmsle_from_counts,
    to_counts,
)


def test_to_counts_inverts_log1p_and_clips_negative():
    counts = to_counts(np.array([0.0, np.log1p(5.0), -5.0]))
    assert np.allclose(counts, [0.0, 5.0, 0.0])


def test_rmsle_is_zero_for_perfect_predictions():
    y = np.array([0.0, 3.0, 250.0])
    assert rmsle_from_counts(y, y) == pytest.approx(0.0)


def test_rmsle_equals_rmse_on_log1p_scale():
    y_true = np.array([10.0, 100.0])
    y_pred = np.array([20.0, 50.0])
    expected = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    assert rmsle_from_counts(y_true, y_pred) == pytest.approx(expected)


def test_rmsle_treats_negative_predictions_as_zero():
    # Clip-to-zero must happen before log1p, otherwise log1p(-3) is NaN.
    assert rmsle_from_counts(np.array([0.0]), np.array([-3.0])) == pytest.approx(0.0)


def test_log_mae_and_count_scale_metrics():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([5.0, 25.0])
    assert log_mae_from_counts(y_true, y_pred) == pytest.approx(
        np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
    )
    # errors are -5 and +5, so both RMSE and MAE are 5
    assert rmse_counts(y_true, y_pred) == pytest.approx(5.0)
    assert mae_counts(y_true, y_pred) == pytest.approx(5.0)
