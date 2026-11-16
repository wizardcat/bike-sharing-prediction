import os

import pytest

MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_model.bin")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="model artifact missing; run `pipenv run python train.py` first",
)

VALID_HOUR = {
    "season": 4,
    "yr": 1,
    "mnth": 10,
    "hr": 8,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import predict

    return TestClient(predict.app)


@pytest.fixture(scope="module")
def api():
    import predict

    return predict


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_one_non_negative_int_per_hour(client):
    payload = {"hours": [VALID_HOUR, {**VALID_HOUR, "hr": 0}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(isinstance(value, int) and value >= 0 for value in predictions)


def test_predict_learns_the_commute_peak(client):
    # Sanity check on model behaviour, not on exact values.
    payload = {"hours": [{**VALID_HOUR, "hr": 8}, {**VALID_HOUR, "hr": 3}]}
    morning_peak, night = client.post("/predict", json=payload).json()["predictions"]
    assert morning_peak > night


def test_predict_rejects_out_of_range_hour(client):
    response = client.post("/predict", json={"hours": [{**VALID_HOUR, "hr": 99}]})
    assert response.status_code == 422


def test_predict_rejects_unknown_season(client):
    response = client.post("/predict", json={"hours": [{**VALID_HOUR, "season": 9}]})
    assert response.status_code == 422


def test_predict_rejects_empty_request(client):
    response = client.post("/predict", json={"hours": []})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_HOUR.items() if k != "workingday"}
    response = client.post("/predict", json={"hours": [incomplete]})
    assert response.status_code == 422


def test_feature_preparation_matches_model_features(api):
    X = api.prepare_features([VALID_HOUR])
    assert list(X.columns) == api.FEATURES
    # a valid request must activate exactly one column per input feature
    assert X.sum(axis=1).iloc[0] == 7
