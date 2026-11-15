# train.py — Train and save the bike-sharing demand model.
# This script follows the same steps as notebook.ipynb but runs from the terminal.
# It produces models/bike_demand_model.bin, which predict.py loads for serving.

import hashlib
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SEED = 42
SEEDS = [0, 1, 2, 3, 4]
DATA_PATH = "data/hour.csv"
MODEL_PATH = "models/bike_demand_model.bin"

TRAIN_END = pd.Timestamp("2012-06-30")
VALID_END = pd.Timestamp("2012-09-30")

CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
]

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


def rmse(y_true, y_pred):
    """RMSE; when both inputs are log1p(cnt) this is RMSLE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_features(frame, columns=None):
    """One-hot encode the calendar features using a fixed column set."""
    encoded = pd.get_dummies(frame[CATEGORICAL_FEATURES], dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded


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
y_test = test["log_cnt"].to_numpy()

print("features    :", len(FEATURES))
print("train shape :", X_train.shape)
print("valid shape :", X_valid.shape)
print("test shape  :", X_test.shape)
print()

# ---------------------------------------------------------------------------
# 4. Baselines
# ---------------------------------------------------------------------------
results = []

train_mean = float(y_train.mean())
pred_global_mean = np.full(len(y_valid), train_mean)
results.append(
    {
        "model": "baseline: global mean",
        "rmsle_mean": rmse(y_valid, pred_global_mean),
        "rmsle_std": 0.0,
    }
)

profile = train.groupby(["weekday", "hr"])["log_cnt"].mean()
valid_keys = list(zip(valid["weekday"], valid["hr"]))
pred_profile = np.array([profile.get(key, train_mean) for key in valid_keys])
results.append(
    {
        "model": "baseline: weekday x hour mean",
        "rmsle_mean": rmse(y_valid, pred_profile),
        "rmsle_std": 0.0,
    }
)

# ---------------------------------------------------------------------------
# 5. Linear regression
# ---------------------------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
pred_lin = lin_model.predict(X_valid)
results.append(
    {
        "model": "linear regression",
        "rmsle_mean": rmse(y_valid, pred_lin),
        "rmsle_std": 0.0,
    }
)

# ---------------------------------------------------------------------------
# 6. Decision tree
# ---------------------------------------------------------------------------
tree_scores = []
for seed in SEEDS:
    tree = DecisionTreeRegressor(max_depth=8, random_state=seed)
    tree.fit(X_train, y_train)
    tree_scores.append(rmse(y_valid, tree.predict(X_valid)))

results.append(
    {
        "model": "decision tree (max_depth=8)",
        "rmsle_mean": float(np.mean(tree_scores)),
        "rmsle_std": float(np.std(tree_scores)),
    }
)

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
    forest_scores.append(rmse(y_valid, forest.predict(X_valid)))

results.append(
    {
        "model": "random forest (100 trees)",
        "rmsle_mean": float(np.mean(forest_scores)),
        "rmsle_std": float(np.std(forest_scores)),
    }
)

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
print("tuning XGBoost (this may take ~30-60 seconds)...")
for eta in etas:
    for max_depth in max_depths:
        scores = []
        best_iterations = []
        for seed in SEEDS:
            params = dict(base_params, eta=eta, max_depth=max_depth, seed=seed)
            booster = train_booster(dtrain, dvalid, params)
            scores.append(rmse(y_valid, predict_booster(booster, dvalid)))
            best_iterations.append(booster.best_iteration + 1)
        tuning_rows.append(
            {
                "eta": eta,
                "max_depth": max_depth,
                "rmsle_mean": float(np.mean(scores)),
                "rmsle_std": float(np.std(scores)),
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

results.append(
    {
        "model": "xgboost (tuned)",
        "rmsle_mean": float(best_row["rmsle_mean"]),
        "rmsle_std": float(best_row["rmsle_std"]),
    }
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
print("validation RMSLE comparison:")
print(comparison[["model", "rmsle"]].to_string(index=False))
print()

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
pred_test_cnt = np.clip(np.expm1(pred_test_log), 0, None)

test_metrics = {
    "RMSLE (log scale)": float(
        np.sqrt(mean_squared_error(np.log1p(test["cnt"]), np.log1p(pred_test_cnt)))
    ),
    "RMSE (counts)": float(np.sqrt(mean_squared_error(test["cnt"], pred_test_cnt))),
    "MAE (counts)": float(mean_absolute_error(test["cnt"], pred_test_cnt)),
}
print("test period metrics:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.3f}")
print()

# ---------------------------------------------------------------------------
# 11. Save model artifact
# ---------------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f_out:
    pickle.dump({"features": FEATURES, "model": final_model}, f_out)

print("saved model to:", MODEL_PATH)
print("model size    :", os.path.getsize(MODEL_PATH), "bytes")
print("training finished at:", datetime.now().astimezone().isoformat())
