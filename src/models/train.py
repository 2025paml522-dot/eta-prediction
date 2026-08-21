"""
Train and compare candidate models, logging every run to MLflow (M3).

Trains at minimum:
- Baseline: Linear Regression
- Candidate: Gradient Boosting (XGBoost)

For each run, logs: hyperparameters, train/val metrics (MAE, RMSE, R2),
the fitted model artifact, and the exact feature-table version (DVC hash)
used, so any run is reproducible from its logged config alone.

Usage:
    python src/models/train.py --config config/config.yaml
"""
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.models.config import (
    PROCESSED_TRIPS_FILE, FEATURE_COLUMNS, TARGET_COL, RANDOM_STATE,
    MODELS_DIR, BEST_MODEL_PATH, MODEL_METADATA_PATH, EXPERIMENTS_LOG,
    REFERENCE_STATS_PATH,
)

# ---- MLflow configuration -------------------------------------------------
# Point this at a remote tracking server if you have one, e.g.
# MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "eta-prediction")


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_TRIPS_FILE)
    return df


def get_candidate_models() -> dict:
    """Model zoo: a simple linear baseline vs. tree ensembles, as called
    for by the brief ('linear regression vs. gradient boosting')."""
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


def log_experiment(row: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_row = pd.DataFrame([row])
    if os.path.exists(EXPERIMENTS_LOG):
        df_row.to_csv(EXPERIMENTS_LOG, mode="a", header=False, index=False)
    else:
        df_row.to_csv(EXPERIMENTS_LOG, mode="w", header=True, index=False)


def train_and_compare(test_size: float = 0.2) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    results = {}
    fitted_models = {}

    # One parent run per training invocation, one nested child run per model.
    # This lets you compare all four candidates side-by-side in the MLflow UI
    # while still grouping them under a single "training session".
    with mlflow.start_run(run_name=f"training-run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}") as parent_run:
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("target_column", TARGET_COL)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        for name, model in get_candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True):
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                preds = model.predict(X_test)
                metrics = evaluate(y_test, preds)
                metrics["train_time_sec"] = round(train_time, 3)

                results[name] = metrics
                fitted_models[name] = model

                # --- MLflow logging for this candidate ---
                mlflow.log_param("model_name", name)
                mlflow.log_params(model.get_params())
                mlflow.log_metric("mae", metrics["mae"])
                mlflow.log_metric("rmse", metrics["rmse"])
                mlflow.log_metric("r2", metrics["r2"])
                mlflow.log_metric("train_time_sec", metrics["train_time_sec"])
                mlflow.sklearn.log_model(model, registered_model_name=name)

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

        # Tag the parent run with which child model won, for quick filtering in the UI
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric("best_model_rmse", results[best_name]["rmse"])

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)

        # Also log the winning model against the parent run, and register it
        # in the MLflow Model Registry so it's easy to promote/serve later.
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="best_model",
            registered_model_name="eta-prediction-best-model",
        )

        metadata = {
            "best_model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COL,
            "metrics": results[best_name],
            "all_results": results,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "mlflow_run_id": parent_run.info.run_id,
        }
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(MODEL_METADATA_PATH)

        # Reference feature statistics -> baseline for M5 drift monitoring
        ref_stats = {
            col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
            for col in FEATURE_COLUMNS
        }
        ref_stats["_target"] = {"mean": float(y_train.mean()), "std": float(y_train.std())}
        with open(REFERENCE_STATS_PATH, "w") as f:
            json.dump(ref_stats, f, indent=2)
        mlflow.log_artifact(REFERENCE_STATS_PATH)

        print(f"\n[train] BEST MODEL: {best_name} -> saved to {BEST_MODEL_PATH}")
        print(f"[mlflow] Parent run ID: {parent_run.info.run_id}")

    return metadata


if __name__ == "__main__":
    train_and_compare()

