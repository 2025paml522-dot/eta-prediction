"""Compute held-out metrics for a trained run and produce the model
comparison report referenced in the submission checklist (M3)."""

import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.config import (
    PROCESSED_TRIPS_FILE, FEATURE_COLUMNS, TARGET_COL, RANDOM_STATE,
    MODELS_DIR, BEST_MODEL_PATH, MODEL_METADATA_PATH, EXPERIMENTS_LOG,
    REFERENCE_STATS_PATH,
)



def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/raw/trips_raw.csv")
    return df

def get_candidate_models() -> dict:
    """Model zoo: a simple linear baseline vs. tree ensembles, as called
    for by the brief ('linear regression vs. gradient boosting')."""
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, random_state=42,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_experiment(row: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(EXPERIMENTS_LOG):
        df_row.to_csv(EXPERIMENTS_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(EXPERIMENTS_LOG, mode="w", header=True, index=False)


def log_experiment(row: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(EXPERIMENTS_LOG):
        df_row.to_csv(EXPERIMENTS_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(EXPERIMENTS_LOG, mode="w", header=True, index=False)


def train_and_compare(test_size: float = 0.2) -> dict:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["train_time_sec"] = round(train_time, 3)

        results[name] = metrics
        fitted_models[name] = model

        log_experiment({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": name,
            "params": json.dumps(model.get_params()),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "train_time_sec": metrics["train_time_sec"],
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        print(f"[train] {name:20s} MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
              f"R2={metrics['r2']:.4f}  ({train_time:.2f}s)")

    # best model = lowest RMSE on held-out test set
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_model = fitted_models[best_name]

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
        "metrics": results[best_name],
        "all_results": results,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Reference feature statistics -> baseline for M5 drift monitoring
    ref_stats = {
        col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
        for col in FEATURE_COLUMNS
    }
    ref_stats["_target"] = {"mean": float(y_train.mean()), "std": float(y_train.std())}
    with open(REFERENCE_STATS_PATH, "w") as f:
        json.dump(ref_stats, f, indent=2)

    print(f"\n[train] BEST MODEL: {best_name} -> saved to {BEST_MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    train_and_compare()

