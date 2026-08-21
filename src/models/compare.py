"""
compare.py

Standalone script to train and compare the 4 candidate ETA-prediction models
(linear_regression, ridge_regression, random_forest, gradient_boosting) as
independent, top-level MLflow runs.

Unlike train.py (which nests all 4 candidates under one parent "training
session" run), this script logs each model as its own flat run in the same
experiment. That makes it simpler to select all 4 in the MLflow UI's Runs
table and hit "Compare" without needing to expand a parent run first.

This script does NOT save a best model to disk or touch the model registry --
it's purely for side-by-side experimentation/comparison. Use train.py when
you're ready to actually pick and persist a production model.

Usage:
    python compare.py
    python compare.py --test-size 0.25
"""
import argparse
import json
import time
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.config import (
    PROCESSED_TRIPS_FILE, FEATURE_COLUMNS, TARGET_COL, RANDOM_STATE,
)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "eta-prediction-comparison"


def get_candidate_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_models(test_size: float = 0.2) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv(PROCESSED_TRIPS_FILE)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    batch_tag = f"compare-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    summary_rows = []

    for name, model in get_candidate_models().items():
        # Each model gets its own top-level run (no nesting), so all 4
        # appear side-by-side in the Runs table and can be multi-selected
        # for the built-in "Compare" view.
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("comparison_batch", batch_tag)
            mlflow.set_tag("model_name", name)

            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            metrics["train_time_sec"] = round(train_time, 3)

            mlflow.log_param("model_name", name)
            mlflow.log_params(model.get_params())
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))

            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])
            mlflow.log_metric("train_time_sec", metrics["train_time_sec"])

            mlflow.sklearn.log_model(model, artifact_path=name)

            summary_rows.append({"model_name": name, **metrics})

            print(f"[compare] {name:20s} MAE={metrics['mae']:.3f}  "
                  f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}  "
                  f"({train_time:.2f}s)")

    summary_df = pd.DataFrame(summary_rows).sort_values("rmse")
    print("\n[compare] Summary (sorted by RMSE, best first):")
    print(summary_df.to_string(index=False))
    print(f"\n[compare] Batch tag for this run: {batch_tag}")
    print("[compare] Open the MLflow UI, filter by tags.comparison_batch = "
          f"'{batch_tag}' to isolate just these 4 runs, select all, and click Compare.")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare candidate ETA models in MLflow.")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of data to hold out for testing (default: 0.2)")
    args = parser.parse_args()

    compare_models(test_size=args.test_size)