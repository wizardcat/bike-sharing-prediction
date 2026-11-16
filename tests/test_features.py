import pandas as pd

from features import CATEGORICAL_FEATURES, make_features


def hour_row(**overrides):
    row = {
        "season": 1,
        "yr": 0,
        "mnth": 1,
        "hr": 0,
        "holiday": 0,
        "weekday": 0,
        "workingday": 0,
    }
    row.update(overrides)
    return row


def test_fitted_encoder_gives_one_hot_column_per_feature():
    frame = pd.DataFrame([hour_row(), hour_row(season=2, hr=8, weekday=6)])
    X = make_features(frame)
    assert X.shape[0] == 2
    # every row has exactly one active column per input feature
    assert (X.sum(axis=1) == len(CATEGORICAL_FEATURES)).all()
    assert list(X.dtypes.unique()) == [float]


def test_reindex_restores_training_columns_and_order():
    train_columns = list(make_features(pd.DataFrame([hour_row()])).columns)
    # month 7 never appears in the "training" frame above
    X = make_features(pd.DataFrame([hour_row(mnth=7)]), train_columns)
    assert list(X.columns) == train_columns
    assert X.shape[0] == 1
    # the unseen month contributes no active column
    assert X.sum(axis=1).iloc[0] == len(CATEGORICAL_FEATURES) - 1


def test_column_order_is_whatever_the_caller_passed():
    columns = ["season_4", "hr_23"]
    X = make_features(pd.DataFrame([hour_row(season=4, hr=23)]), columns)
    assert list(X.columns) == columns
    assert X.iloc[0].tolist() == [1.0, 1.0]
