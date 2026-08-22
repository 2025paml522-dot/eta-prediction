"""Unit tests for feature engineering functions (haversine, time buckets)."""
import pandas as pd
from build_features import build_features


def _base_row(**overrides):
    row = {
        "trip_id": "T1",
        "pickup_datetime": pd.Timestamp("2024-03-14 09:00:00"),  # a Thursday
        "distance_km": 5.0,
        "weather_condition": "clear",
        "traffic_level": "medium",
        "temperature_c": 20.0,
        "precipitation_mm": 0.0,
        "duration_min": 15.0,
    }
    row.update(overrides)
    return row


def test_hour_of_day_extracted_correctly():
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-14 14:30:00"))])
    result = build_features(df)
    assert result["hour_of_day"].iloc[0] == 14


def test_weekday_is_not_weekend():
    # 2024-03-14 is a Thursday
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-14 09:00:00"))])
    result = build_features(df)
    assert result["is_weekend"].iloc[0] == False
    assert result["day_of_week"].iloc[0] == 3  # Thursday = 3


def test_saturday_is_weekend():
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-16 09:00:00"))])
    result = build_features(df)
    assert result["is_weekend"].iloc[0] == True


def test_morning_rush_hour_flagged():
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-14 08:00:00"))])
    result = build_features(df)
    assert result["is_rush_hour"].iloc[0] == True


def test_evening_rush_hour_flagged():
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-14 17:00:00"))])
    result = build_features(df)
    assert result["is_rush_hour"].iloc[0] == True


def test_midday_not_rush_hour():
    df = pd.DataFrame([_base_row(pickup_datetime=pd.Timestamp("2024-03-14 13:00:00"))])
    result = build_features(df)
    assert result["is_rush_hour"].iloc[0] == False


def test_original_columns_preserved():
    df = pd.DataFrame([_base_row()])
    result = build_features(df)
    for col in ["distance_km", "weather_condition", "traffic_level", "duration_min"]:
        assert col in result.columns