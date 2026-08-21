"""
M4 - Loads the trained model + metadata once at API startup, and turns a
single TripRequest into the exact feature vector the model expects
(reusing the same encoding logic as training, so there is no train/serve skew).
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

from src.models.config import (
    BEST_MODEL_PATH, MODEL_METADATA_PATH, MODELS_DIR, FEATURE_COLUMNS, CATEGORICAL_MAPS,
)

from src.serving.schemas import TripRequest


class ModelService:
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.zone_vocab = {}
        self.load()

    def load(self):
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {BEST_MODEL_PATH}. Run `python -m src.train` first."
            )
        self.model = joblib.load(BEST_MODEL_PATH)

        with open(MODEL_METADATA_PATH) as f:
            self.metadata = json.load(f)

        zone_vocab_path = os.path.join(MODELS_DIR, "zone_vocab.joblib")
        self.zone_vocab = joblib.load(zone_vocab_path) if os.path.exists(zone_vocab_path) else {}

        print(f"[model_loader] loaded model '{self.metadata.get('best_model_name')}' "
              f"trained at {self.metadata.get('trained_at_utc')}")

    def build_features(self, trip: TripRequest) -> pd.DataFrame:
        hour = trip.pickup_datetime.hour
        weekday = trip.pickup_datetime.weekday()


        row = {
            "distance_km": trip.distance_km,
            "hour_of_day": hour,
            "day_of_week": weekday,
            "is_weekend": int(weekday >= 5),
            "is_rush_hour": int(hour in [7, 8, 9, 17, 18, 19]),
            "temperature_c": trip.temperature_c,
            "precipitation_mm": trip.precipitation_mm,
            "traffic_level_enc": CATEGORICAL_MAPS["traffic_level"].get(trip.traffic_level, -1),
            "weather_condition_enc": CATEGORICAL_MAPS["weather_condition"].get(trip.weather_condition, -1),
            "passenger_count": trip.passenger_count,
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def predict(self, trip: TripRequest) -> float:
        X = self.build_features(trip)
        pred = self.model.predict(X)[0]
        return float(max(pred, 0.5))


# Singleton instance reused across requests
model_service = ModelService()
