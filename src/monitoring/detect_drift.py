"""
Drift detection for the ETA prediction service .

"""
import argparse
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


if __name__ == "__main__":
    main()
