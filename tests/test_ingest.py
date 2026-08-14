"""Unit tests for schema validation (Day 4 logic)."""
import pandas as pd
from validate_schema import validate


def _base_row(**overrides):
    row = {
        "trip_id": "T1", "pickup_datetime": "2024-03-14 09:00:00",
        "pickup_lat": 40.75, "pickup_lon": -73.98,
        "dropoff_lat": 40.76, "dropoff_lon": -73.99,
        "distance_km": 5.0, "weather_condition": "clear",
        "temperature_c": 20.0, "precipitation_mm": 0.1,
        "traffic_level": "medium", "passenger_count": 2, "duration_min": 15.0,
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    valid, rejected, _ = validate(pd.DataFrame([_base_row()]))
    assert len(valid) == 1 and len(rejected) == 0


def test_bad_gps_rejected():
    valid, rejected, _ = validate(pd.DataFrame([_base_row(dropoff_lat=200.0)]))
    assert len(valid) == 0
    assert "gps_within_nyc" in rejected.iloc[0]["reject_reason"]


def test_negative_distance_rejected():
    valid, rejected, _ = validate(pd.DataFrame([_base_row(distance_km=-5.0)]))
    assert len(valid) == 0
    assert "distance_positive" in rejected.iloc[0]["reject_reason"]


def test_bad_timestamp_rejected():
    valid, rejected, _ = validate(pd.DataFrame([_base_row(pickup_datetime="not-a-timestamp")]))
    assert len(valid) == 0
    assert "timestamps_parseable" in rejected.iloc[0]["reject_reason"]


def test_missing_column_raises():
    df = pd.DataFrame([_base_row()]).drop(columns=["distance_km"])
    try:
        validate(df)
        assert False, "should have raised"
    except ValueError:
        pass