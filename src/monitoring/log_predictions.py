"""
Logs every prediction the API serves, for later drift detection (M5).

Design: append-only JSONL (one JSON object per line). This format is
easy to append to safely from a running API process, and pandas can
read it directly with pd.read_json(path, lines=True) for the drift
analysis in detect_drift.py.

NOTE: actual_duration_min starts as None. A separate process (not yet
built) will need to backfill it once real trip outcomes are known --
that's a design gap to raise with the team, not solved here.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # src/monitoring/ -> src/ -> repo root


def log_prediction(
    features: dict,
    predicted_duration_min: float,
    model_version: str,
    log_path: str = None,
) -> str:
    """
    Append one prediction record to the log. Returns the request_id.
    """
    if log_path is None:
        log_path = REPO_ROOT / "data" / "monitoring" / "predictions.jsonl"

    request_id = str(uuid.uuid4())

    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "features": features,
        "predicted_duration_min": predicted_duration_min,
        "actual_duration_min": None,
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return request_id