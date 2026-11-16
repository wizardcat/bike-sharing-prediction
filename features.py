# features.py — feature engineering shared by train.py and predict.py.
# One implementation for both paths prevents train/serve skew: the encoder is
# always the same function, and only the column list (learned on the training
# split) is passed in at inference time.

from __future__ import annotations

import pandas as pd

# All model inputs are calendar fields. They are encoded as strings so that
# one-hot encoding treats them as categories (hr=3 is not "three times hour 1").
CATEGORICAL_FEATURES: list[str] = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
]


def make_features(
    frame: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode the calendar features.

    Parameters
    ----------
    frame:
        DataFrame containing at least ``CATEGORICAL_FEATURES``.
    columns:
        Fixed column list learned on the training split. Pass it for the
        validation/test splits or at serving time so the matrix always has the
        same shape and order as the model expects. ``None`` fits the encoder.

    Returns
    -------
    A float DataFrame with one column per observed category value.
    """
    encoded = pd.get_dummies(frame[CATEGORICAL_FEATURES].astype(str), dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded
