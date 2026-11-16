# predict.py — FastAPI web service for bike-sharing demand predictions.
# Loads models/bike_demand_model.bin (produced by train.py) and returns the
# predicted number of rentals for each requested hour.

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import make_features

# ---------------------------------------------------------------------------
# Load model once when the server starts
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/bike_demand_model.bin")

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

FEATURES = artifact["features"]
MODEL = artifact["model"]
# Older artifacts saved before metadata was added do not have this key.
METADATA = artifact.get("metadata", {})

app = FastAPI(title="Bike Sharing Demand Prediction API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request/response schemas
#
# The ranges match the values that appear in the training data. Rejecting
# out-of-range values here is what keeps the one-hot matrix meaningful: an
# unknown category would otherwise be encoded as an all-zero row and would be
# silently predicted as if it were a valid hour.
# ---------------------------------------------------------------------------
class HourInput(BaseModel):
    season: int = Field(ge=1, le=4, description="1=spring, 2=summer, 3=fall, 4=winter")
    yr: int = Field(ge=0, le=1, description="0=2011, 1=2012")
    mnth: int = Field(ge=1, le=12, description="Month, 1-12")
    hr: int = Field(ge=0, le=23, description="Hour of the day, 0-23")
    holiday: int = Field(ge=0, le=1, description="1 if the day is a holiday")
    weekday: int = Field(ge=0, le=6, description="Day of week, 0-6")
    workingday: int = Field(ge=0, le=1, description="1 if neither weekend nor holiday")


class PredictionRequest(BaseModel):
    hours: list[HourInput] = Field(min_length=1, description="One entry per hour")


class PredictionResponse(BaseModel):
    predictions: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prepare_features(rows: list[dict]) -> pd.DataFrame:
    """Convert request rows into the same one-hot matrix used during training."""
    return make_features(pd.DataFrame(rows), FEATURES)


def predict_counts(rows: list[dict]) -> list[int]:
    """Return predicted rental counts for a list of feature dictionaries."""
    X = prepare_features(rows)
    pred_log = MODEL.predict(xgb.DMatrix(X))
    counts = np.clip(np.expm1(pred_log), 0, None)
    return counts.round().astype(int).tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Simple 200 OK check used by the platform and by smoke tests."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    """Landing endpoint with model details."""
    return {
        "service": "bike-sharing-demand-prediction",
        "model_path": MODEL_PATH,
        "num_features": len(FEATURES),
        "trained_at": METADATA.get("created_at"),
        "git_sha": METADATA.get("git_sha"),
        "test_metrics": METADATA.get("metrics", {}).get("test"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict:
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
