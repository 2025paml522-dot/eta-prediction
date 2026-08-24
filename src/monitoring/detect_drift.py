"""
Drift detection for the ETA prediction service .

"""
import argparse
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25

NUMERIC_FEATURES = ["distance_km", "temperature_c", "precipitation_mm"]
CATEGORICAL_FEATURES = ["weather_condition", "traffic_level"]


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for a continuous feature, bucketed using the reference distribution's
    own percentiles so bins reflect where the reference data actually sits."""
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)  # guard against duplicate percentiles

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.clip(ref_counts / len(reference), 1e-4, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature, comparing category proportions directly."""
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    categories = set(reference.unique()) | set(current.unique())
    ref_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=1e-4).clip(lower=1e-4)
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=1e-4).clip(lower=1e-4)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_rolling_mae(recent_df: pd.DataFrame) -> dict:
    """MAE on predictions where ground truth has been filled in via /feedback."""
    known = recent_df.dropna(subset=["actual_duration_min"])
    if len(known) == 0:
        return {"n_with_ground_truth": 0, "rolling_mae": None}

    mae = float((known["actual_duration_min"] - known["predicted_duration_min"]).abs().mean())
    return {"n_with_ground_truth": int(len(known)), "rolling_mae": round(mae, 3)}


def detect_drift(reference_df: pd.DataFrame, recent_df: pd.DataFrame) -> dict:
    feature_psi = {}

    for col in NUMERIC_FEATURES:
        if col in reference_df.columns and col in recent_df.columns:
            feature_psi[col] = round(psi_numeric(reference_df[col], recent_df[col]), 4)

    for col in CATEGORICAL_FEATURES:
        if col in reference_df.columns and col in recent_df.columns:
            feature_psi[col] = round(psi_categorical(reference_df[col], recent_df[col]), 4)

    flagged_features = {
        col: psi for col, psi in feature_psi.items() if psi >= PSI_WARN_THRESHOLD
    }
    alert_features = {
        col: psi for col, psi in feature_psi.items() if psi >= PSI_ALERT_THRESHOLD
    }

    mae_stats = compute_rolling_mae(recent_df)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_reference_rows": len(reference_df),
        "n_recent_rows": len(recent_df),
        "feature_psi": feature_psi,
        "flagged_features": flagged_features,
        "alert_features": alert_features,
        "drift_detected": len(alert_features) > 0,
        **mae_stats,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference data (e.g. training set)")
    parser.add_argument("--recent", required=True, help="Path to recent predictions log (CSV)")
    parser.add_argument("--out", default="reports/drift_report.json")
    args = parser.parse_args()

    if args.reference.endswith(".parquet"):
        reference_df = pd.read_parquet(args.reference)
    else:
        reference_df = pd.read_csv(args.reference)

    recent_df = pd.read_csv(args.recent)

    report = detect_drift(reference_df, recent_df)

    print(json.dumps(report, indent=2))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[detect_drift] report written to {args.out}")

    if report["drift_detected"]:
        print(f"[detect_drift] ALERT: drift detected in {list(report['alert_features'].keys())}")

def run_drift_check(
    reference_path: str = None,
    recent_path: str = None,
    out_path: str = None,
) -> dict:
    """
    No-argument-friendly wrapper around detect_drift(), used by
    dashboard/app.py's "Run drift check now" button.

    Writes a report in the schema dashboard/app.py's Monitoring & Drift
    tab actually reads (n_predictions_total, max_psi, feature_drift_psi,
    retrain_recommended, reasons, performance_drift.{live_mae,test_mae,
    n_labeled_samples,degradation_ratio}) -- this is a DIFFERENT shape
    than detect_drift()'s own return value (which uses feature_psi,
    n_recent_rows, rolling_mae, etc. -- that's the schema the CLI and
    reports/drift_report.json use, kept unchanged since it's already
    verified working). This function translates between the two.
    """
    from src.models.config import PROCESSED_TRIPS_FILE, PREDICTIONS_LOG, DRIFT_REPORT_PATH, EXPERIMENTS_LOG

    if reference_path is None:
        reference_path = str(PROCESSED_TRIPS_FILE)
    if recent_path is None:
        recent_path = str(PREDICTIONS_LOG)
    if out_path is None:
        out_path = str(DRIFT_REPORT_PATH)

    if not os.path.exists(recent_path):
        report = {
            "status": "no_predictions_logged_yet",
            "n_predictions_total": 0,
            "max_psi": 0,
            "feature_drift_psi": {},
            "retrain_recommended": False,
            "reasons": [],
            "performance_drift": {
                "live_mae": "—", "test_mae": "—",
                "n_labeled_samples": 0, "degradation_ratio": "—",
            },
        }
    else:
        if str(reference_path).endswith(".parquet"):
            reference_df = pd.read_parquet(reference_path)
        else:
            reference_df = pd.read_csv(reference_path)

        recent_df = pd.read_csv(recent_path)

        raw_report = detect_drift(reference_df, recent_df)

        # Look up the best model's training-time test MAE from the
        # experiments log, for the performance_drift comparison.
        test_mae = "—"
        try:
            exp_df = pd.read_csv(EXPERIMENTS_LOG)
            if len(exp_df) > 0 and "mae" in exp_df.columns:
                test_mae = round(float(exp_df.sort_values("mae").iloc[0]["mae"]), 4)
        except Exception:
            pass  # experiments log missing/unreadable -- leave as "—" rather than fail the whole check

        feature_psi = raw_report.get("feature_psi", {})
        max_psi = round(max(feature_psi.values()), 4) if feature_psi else 0
        alert_features = raw_report.get("alert_features", {})
        rolling_mae = raw_report.get("rolling_mae")
        n_labeled = raw_report.get("n_with_ground_truth", 0)

        degradation_ratio = "—"
        if rolling_mae is not None and isinstance(test_mae, (int, float)) and test_mae:
            degradation_ratio = round(rolling_mae / test_mae, 3)

        report = {
            "status": "ok",
            "n_predictions_total": raw_report.get("n_recent_rows", 0),
            "max_psi": max_psi,
            "feature_drift_psi": feature_psi,
            "retrain_recommended": len(alert_features) > 0,
            "reasons": list(alert_features.keys()),
            "performance_drift": {
                "live_mae": rolling_mae if rolling_mae is not None else "—",
                "test_mae": test_mae,
                "n_labeled_samples": n_labeled,
                "degradation_ratio": degradation_ratio,
            },
        }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    main()
