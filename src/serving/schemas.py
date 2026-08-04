"""Pydantic request/response models for the /predict endpoint (M4)."""
from pydantic import BaseModel, Field

class TripRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    pickup_datetime: str
    weather_condition: str = Field(description="clear, rain, snow, fog, etc.")

class ETAResponse(BaseModel):
    predicted_duration_seconds: float
    model_version: str
    request_id: str
