"""Pydantic request/response models for the /predict endpoint (M4)."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    pickup_datetime: datetime = Field(..., description="ISO timestamp of pickup, e.g. 2024-06-07T18:30:00")
    pickup_lat: float = Field(..., ge=-90, le=90)
    pickup_lon: float = Field(..., ge=-180, le=180)
    dropoff_lat: float = Field(..., ge=-90, le=90)
    dropoff_lon: float = Field(..., ge=-180, le=180)
    distance_km: float = Field(..., gt=0, le=200)
    weather_condition: Literal["clear", "cloudy", "rain", "storm", "fog"] = "clear"
    temperature_c: float = 20.0
    precipitation_mm: float = 0.0
    traffic_level: Literal["low", "medium", "high", "severe"] = "medium"
    passenger_count: int = Field(1, ge=1, le=8)

    @field_validator("distance_km")
    @classmethod
    def sane_distance(cls, v):
        if v <= 0:
            raise ValueError("distance_km must be positive")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2024-06-07T18:30:00",
                "pickup_lat": 40.75,
                "pickup_lon": -73.98,
                "dropoff_lat": 40.77,
                "dropoff_lon": -73.96,
                "distance_km": 4.2,
                "weather_condition": "rain",
                "temperature_c": 18.5,
                "precipitation_mm": 3.1,
                "traffic_level": "high",
                "passenger_count": 2,
            }
        }


class PredictionResponse(BaseModel):
    predicted_duration_min: float
    model_name: str
    model_version: str
    request_id: str
    predicted_at_utc: str


class ActualDurationUpdate(BaseModel):
    request_id: str
    actual_duration_min: float = Field(..., gt=0)


class HealthResponse(BaseModel):
    status: str
    model_name: Optional[str] = None
    model_trained_at: Optional[str] = None