# predict.py — FastAPI web service for bike-sharing demand predictions.
# Loads models/bike_demand_model.bin (produced by train.py) and returns the
# predicted number of rentals for each requested hour.

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Load model once when the server starts
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_model.bin")

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

FEATURES = artifact["features"]
MODEL = artifact["model"]

CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
]

app = FastAPI(title="Bike Sharing Demand Prediction API")


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class HourInput(BaseModel):
    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int


class PredictionRequest(BaseModel):
    hours: list[HourInput]


class PredictionResponse(BaseModel):
    predictions: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_features(rows):
    """Convert request rows into the same one-hot matrix used during training."""
    frame = pd.DataFrame(rows)[CATEGORICAL_FEATURES].astype(str)
    encoded = pd.get_dummies(frame, dtype=float).reindex(
        columns=FEATURES, fill_value=0.0
    )
    return encoded


def predict_counts(rows):
    """Return predicted rental counts for a list of feature dictionaries."""
    X = prepare_features(rows)
    pred_log = MODEL.predict(xgb.DMatrix(X))
    counts = np.clip(np.expm1(pred_log), 0, None)
    return counts.round().astype(int).tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Health-check / landing endpoint."""
    return {
        "service": "bike-sharing-demand-prediction",
        "model_path": MODEL_PATH,
        "num_features": len(FEATURES),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """Predict rental counts for one or more hours."""
    rows = [hour.model_dump() for hour in payload.hours]
    return {"predictions": predict_counts(rows)}


# ---------------------------------------------------------------------------
# CLI sanity check (can also run `python predict.py` for a quick test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        {
            "season": 4,
            "yr": 1,
            "mnth": 10,
            "hr": 0,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
        },
        {
            "season": 4,
            "yr": 1,
            "mnth": 10,
            "hr": 8,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
        },
    ]
    print("sample predictions:", predict_counts(sample))
