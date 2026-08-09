"""Great Expectations suite definitions for the trip dataset (M2)."""
import pandas as pd

REQUIRED_COLUMNS = [
    "trip_id", "pickup_datetime", "pickup_lat", "pickup_lon",
    "dropoff_lat", "dropoff_lon", "distance_km", "weather_condition",
    "temperature_c", "precipitation_mm", "traffic_level",
    "passenger_count", "duration_min",
]

NYC_LAT_RANGE = (40.4, 41.1)
NYC_LON_RANGE = (-74.5, -73.3)
MIN_DURATION_MIN, MAX_DURATION_MIN = 1, 180

CRITICAL_FIELDS = [
    "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
    "duration_min", "distance_km",
]


def validate(raw_df: pd.DataFrame):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    d = raw_df.copy()
    d["pickup_datetime"] = pd.to_datetime(d["pickup_datetime"], errors="coerce")

    checks = {
        "timestamps_parseable": d["pickup_datetime"].notna(),
        "no_missing_critical_fields": d[CRITICAL_FIELDS].notna().all(axis=1),
        "gps_within_nyc": (
            d["pickup_lat"].between(*NYC_LAT_RANGE) & d["dropoff_lat"].between(*NYC_LAT_RANGE)
            & d["pickup_lon"].between(*NYC_LON_RANGE) & d["dropoff_lon"].between(*NYC_LON_RANGE)
        ),
        "duration_in_range": d["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN),
        "distance_positive": d["distance_km"] > 0,
    }
    checks_df = pd.DataFrame(checks)
    overall_valid = checks_df.all(axis=1)

    def reasons(row):
        return ";".join(name for name, passed in row.items() if not passed)
    reject_reason = checks_df.apply(reasons, axis=1)

    valid_df = d[overall_valid].copy()
    rejected_df = d[~overall_valid].copy()
    rejected_df["reject_reason"] = reject_reason[~overall_valid]

    summary_df = pd.DataFrame({
        "check": list(checks.keys()),
        "passed": [int(checks_df[c].sum()) for c in checks],
        "failed": [int((~checks_df[c]).sum()) for c in checks],
    })

    return valid_df, rejected_df, summary_df