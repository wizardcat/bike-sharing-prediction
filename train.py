# train.py — Train and save the bike-sharing demand model.
# This script follows the same steps as notebook.ipynb but runs from the terminal.
# It produces models/bike_demand_model.bin, which predict.py loads for serving.

import hashlib
import json
import os
import pickle
import platform
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from features import CATEGORICAL_FEATURES, make_features
from metrics import (
    log_mae_from_counts,
    mae_counts,
    rmse_counts,
    rmsle_from_counts,
    to_counts,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "bike-sharing-demand-calendar-xgb"
SEED = 42
SEEDS = [0, 1, 2, 3, 4]
DATA_PATH = "data/hour.csv"
MODEL_PATH = "models/bike_demand_model.bin"
METADATA_PATH = "models/bike_demand_model.metadata.json"

TRAIN_END = pd.Timestamp("2012-06-30")
VALID_END = pd.Timestamp("2012-09-30")

PRIMARY_METRIC = "RMSLE = RMSE of log1p(counts), predictions clipped at 0"

np.random.seed(SEED)
os.makedirs("models", exist_ok=True)


def git_sha():
    """Get current git commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def dataset_hash(path):
    """SHA-256 of the raw data file for reproducibility logging."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def score(y_true_counts, preds_log):
    """Return (RMSLE, log-scale MAE) for log-space predictions.

    Predictions are converted to non-negative counts first, so every model is
    scored by exactly the same transform that the served model uses.
    """
    counts = to_counts(preds_log)
    return (
        rmsle_from_counts(y_true_counts, counts),
        log_mae_from_counts(y_true_counts, counts),
    )


def add_result(name, scores):
    """Store mean +- std of the metrics over the seeds of one model."""
    rmsle = [s[0] for s in scores]
    log_mae = [s[1] for s in scores]
    results.append(
        {
            "model": name,
            "rmsle_mean": float(np.mean(rmsle)),
            "rmsle_std": float(np.std(rmsle)),
            "log_mae_mean": float(np.mean(log_mae)),
            "log_mae_std": float(np.std(log_mae)),
        }
    )


def train_booster(dtrain, dvalid, params, max_rounds=500, early_stopping_rounds=50):
    """Train one XGBoost booster with early stopping on the validation set."""
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def predict_booster(booster, dmatrix):
    """Predict with the best iteration kept by early stopping."""
    return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Bike Sharing Demand Prediction — training script")
print("=" * 60)
print("python      :", platform.python_version())
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost     :", xgb.__version__)
print("git sha     :", git_sha())
print("data sha256 :", dataset_hash(DATA_PATH))
print()

df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
df["log_cnt"] = np.log1p(df["cnt"])

# Treat all model inputs as categorical, so one-hot encoding does not impose order.
for col in CATEGORICAL_FEATURES:
    df[col] = df[col].astype(str)

# ---------------------------------------------------------------------------
# 2. Chronological split
# ---------------------------------------------------------------------------
train = df[df["dteday"] <= TRAIN_END].copy()
valid = df[(df["dteday"] > TRAIN_END) & (df["dteday"] <= VALID_END)].copy()
test = df[df["dteday"] > VALID_END].copy()

for name, part in [("train", train), ("valid", valid), ("test", test)]:
    print(
        f"{name:5s}: {len(part):5d} rows | "
        f"{part['dteday'].min().date()} -> {part['dteday'].max().date()} | "
        f"mean cnt {part['cnt'].mean():6.1f}"
    )
print()

# ---------------------------------------------------------------------------
# 3. Feature matrices
# ---------------------------------------------------------------------------
X_train = make_features(train)
FEATURES = list(X_train.columns)  # column order is fixed by the training split
X_valid = make_features(valid, FEATURES)
X_test = make_features(test, FEATURES)

y_train = train["log_cnt"].to_numpy()
y_valid = valid["log_cnt"].to_numpy()

print("features    :", len(FEATURES))
print("train shape :", X_train.shape)
print("valid shape :", X_valid.shape)
print("test shape  :", X_test.shape)
print()

# ---------------------------------------------------------------------------
# 4. Baselines
# ---------------------------------------------------------------------------
results = []

y_valid_counts = valid["cnt"].to_numpy()

train_mean = float(y_train.mean())
pred_global_mean = np.full(len(y_valid), train_mean)
add_result("baseline: global mean", [score(y_valid_counts, pred_global_mean)])

profile = train.groupby(["weekday", "hr"])["log_cnt"].mean()
valid_keys = list(zip(valid["weekday"], valid["hr"]))
pred_profile = np.array([profile.get(key, train_mean) for key in valid_keys])
add_result("baseline: weekday x hour mean", [score(y_valid_counts, pred_profile)])

# ---------------------------------------------------------------------------
# 5. Linear regression
# ---------------------------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
pred_lin = lin_model.predict(X_valid)
add_result("linear regression", [score(y_valid_counts, pred_lin)])

# ---------------------------------------------------------------------------
# 6. Decision tree
# ---------------------------------------------------------------------------
tree_scores = []
for seed in SEEDS:
    tree = DecisionTreeRegressor(max_depth=8, random_state=seed)
    tree.fit(X_train, y_train)
    tree_scores.append(score(y_valid_counts, tree.predict(X_valid)))

add_result("decision tree (max_depth=8)", tree_scores)

# ---------------------------------------------------------------------------
# 7. Random forest
# ---------------------------------------------------------------------------
forest_scores = []
for seed in SEEDS:
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    forest_scores.append(score(y_valid_counts, forest.predict(X_valid)))

add_result("random forest (100 trees)", forest_scores)

# ---------------------------------------------------------------------------
# 8. XGBoost hyperparameter tuning
# ---------------------------------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

base_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 4,
    "seed": SEED,
}

etas = [0.05, 0.1]
max_depths = [4, 6, 8]

tuning_rows = []
tuning_scores = {}
print("tuning XGBoost (this may take ~30-60 seconds)...")
for eta in etas:
    for max_depth in max_depths:
        grid_scores = []
        best_iterations = []
        for seed in SEEDS:
            params = dict(base_params, eta=eta, max_depth=max_depth, seed=seed)
            booster = train_booster(dtrain, dvalid, params)
            grid_scores.append(score(y_valid_counts, predict_booster(booster, dvalid)))
            best_iterations.append(booster.best_iteration + 1)
        tuning_scores[(eta, max_depth)] = grid_scores
        tuning_rows.append(
            {
                "eta": eta,
                "max_depth": max_depth,
                "rmsle_mean": float(np.mean([s[0] for s in grid_scores])),
                "rmsle_std": float(np.std([s[0] for s in grid_scores])),
                "rounds_mean": int(np.mean(best_iterations)),
            }
        )

tuning_table = (
    pd.DataFrame(tuning_rows).sort_values("rmsle_mean").reset_index(drop=True)
)
print(tuning_table.round(4).to_string(index=False))
print()

best_row = tuning_table.iloc[0]
best_params = dict(
    base_params,
    eta=float(best_row["eta"]),
    max_depth=int(best_row["max_depth"]),
)
best_rounds = int(best_row["rounds_mean"])

add_result(
    "xgboost (tuned)",
    tuning_scores[(float(best_row["eta"]), int(best_row["max_depth"]))],
)

print("chosen: eta =", best_params["eta"], "| max_depth =", best_params["max_depth"])
print("boosting rounds to use for final fit:", best_rounds)
print(
    "validation RMSLE: %.4f +- %.4f" % (best_row["rmsle_mean"], best_row["rmsle_std"])
)
print()

# ---------------------------------------------------------------------------
# 9. Model comparison on validation
# ---------------------------------------------------------------------------
comparison = pd.DataFrame(results).sort_values("rmsle_mean").reset_index(drop=True)
comparison["rmsle"] = (
    comparison["rmsle_mean"].map("{:.4f}".format)
    + " +- "
    + comparison["rmsle_std"].map("{:.4f}".format)
)
comparison["log_mae"] = (
    comparison["log_mae_mean"].map("{:.4f}".format)
    + " +- "
    + comparison["log_mae_std"].map("{:.4f}".format)
)
print("validation RMSLE comparison:")
print(comparison[["model", "rmsle", "log_mae"]].to_string(index=False))
print()

# Validation metrics of the selected configuration, kept for the run metadata.
best_validation = {
    "RMSLE": float(best_row["rmsle_mean"]),
    "RMSLE_std": float(best_row["rmsle_std"]),
    "log_MAE": float(
        comparison.loc[comparison["model"] == "xgboost (tuned)", "log_mae_mean"].iloc[0]
    ),
}

# ---------------------------------------------------------------------------
# 10. Final model on train + validation, evaluated once on test
# ---------------------------------------------------------------------------
train_valid = pd.concat([train, valid])
X_full = make_features(train_valid, FEATURES)
y_full = train_valid["log_cnt"].to_numpy()

final_model = xgb.train(
    best_params,
    xgb.DMatrix(X_full, label=y_full),
    num_boost_round=best_rounds,
)

pred_test_log = final_model.predict(xgb.DMatrix(X_test))
pred_test_counts = to_counts(pred_test_log)
test_counts = test["cnt"].to_numpy()

test_metrics = {
    "RMSLE": rmsle_from_counts(test_counts, pred_test_counts),
    "log_MAE": log_mae_from_counts(test_counts, pred_test_counts),
    "RMSE_counts": rmse_counts(test_counts, pred_test_counts),
    "MAE_counts": mae_counts(test_counts, pred_test_counts),
    "mean_actual_counts": float(np.mean(test_counts)),
}
print("test period metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.3f}")
print()

# ---------------------------------------------------------------------------
# 11. Save model artifact + run metadata
# ---------------------------------------------------------------------------
metadata = {
    "experiment": EXPERIMENT_NAME,
    "created_at": datetime.now().astimezone().isoformat(),
    "primary_metric": PRIMARY_METRIC,
    "dataset": {
        "path": DATA_PATH,
        "sha256": dataset_hash(DATA_PATH),
        "rows": len(df),
    },
    "split": {
        "strategy": "chronological",
        "train": [str(train["dteday"].min().date()), str(TRAIN_END.date())],
        "validation": [str(valid["dteday"].min().date()), str(VALID_END.date())],
        "test": [str(test["dteday"].min().date()), str(test["dteday"].max().date())],
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
    },
    "seeds": {"global_seed": SEED, "seed_list": SEEDS},
    "hyperparameters": {
        "eta": float(best_params["eta"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_weight": int(base_params["min_child_weight"]),
        "subsample": float(base_params["subsample"]),
        "colsample_bytree": float(base_params["colsample_bytree"]),
        "boosting_rounds": best_rounds,
        "objective": base_params["objective"],
    },
    "metrics": {"validation": best_validation, "test": test_metrics},
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
    },
    "git_sha": git_sha(),
}

with open(MODEL_PATH, "wb") as f_out:
    pickle.dump(
        {"features": FEATURES, "model": final_model, "metadata": metadata}, f_out
    )

with open(METADATA_PATH, "w", encoding="utf-8") as f_out:
    json.dump(metadata, f_out, indent=2)

print("saved model to:", MODEL_PATH)
print("saved run metadata to:", METADATA_PATH)
print("model size    :", os.path.getsize(MODEL_PATH), "bytes")
print("training finished at:", metadata["created_at"])
