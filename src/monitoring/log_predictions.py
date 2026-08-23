"""
Logs every prediction the API serves, for later drift detection (M5).

Design: append-only CSV, one row per prediction, matching TripRequest's
fields plus prediction metadata. actual_duration_min starts as None and
is filled in later via the API's /feedback endpoint once the real trip
outcome is known -- that's what lets detect_drift.py compute rolling MAE.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "predictions_log.csv"


def log_prediction(
    request_id: str,
    trip_fields: dict,
    prediction: float,
    model_name: str,
    log_path: str = None,
) -> None:
    if log_path is None:
        log_path = DEFAULT_LOG_PATH

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = dict(trip_fields)
    row.update({
        "request_id": request_id,
        "predicted_duration_min": prediction,
        "actual_duration_min": None,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
    })

    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(log_path)
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)