import os

# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Data ----
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VERSIONS_DIR = os.path.join(DATA_DIR, "versions")

RAW_TRIPS_FILE = os.path.join(RAW_DATA_DIR, "trips_raw.csv")
PROCESSED_TRIPS_FILE = os.path.join(PROCESSED_DATA_DIR, "trips_processed.csv")

# ---- Models / experiments ----
MODELS_DIR = os.path.join(ROOT_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "best_model_metadata.json")
EXPERIMENTS_LOG = os.path.join(MODELS_DIR, "experiments_log.csv")

# ---- Serving / monitoring logs ----
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions_log.csv")
DRIFT_REPORT_PATH = os.path.join(LOGS_DIR, "drift_report.json")
REFERENCE_STATS_PATH = os.path.join(MODELS_DIR, "reference_feature_stats.json")

# ---- Feature schema ----
# Raw columns expected from the ingestion source
RAW_SCHEMA = {
    "trip_id": "object",
    "pickup_datetime": "object",
    "pickup_lat": "float64",
    "pickup_lon": "float64",
    "dropoff_lat": "float64",
    "dropoff_lon": "float64",
    "distance_km": "float64",
    "weather_condition": "object",
    "temperature_c": "float64",
    "precipitation_mm": "float64",
    "traffic_level": "object",
    "passenger_count": "int64",
    "duration_min": "float64",  # target
}

TARGET_COL = "duration_min"

# Final feature columns used for model training / inference
FEATURE_COLUMNS = [
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "temperature_c",
    "precipitation_mm",
    "traffic_level_enc",
    "weather_condition_enc",
    "passenger_count",
]

CATEGORICAL_MAPS = {
    "traffic_level": {"low": 0, "medium": 1, "high": 2, "severe": 3},
    "weather_condition": {"clear": 0, "cloudy": 1, "rain": 2, "storm": 3, "fog": 4},
}

RANDOM_STATE = 42

# Valid NYC-ish bounding box used for GPS sanity checks (swap for your city)
LAT_BOUNDS = (40.4, 41.1)
LON_BOUNDS = (-74.5, -73.5)

if __name__ == "__main__":
    print(repr(PROCESSED_TRIPS_FILE))
    print(os.path.exists(PROCESSED_TRIPS_FILE))
    print(repr(ROOT_DIR))