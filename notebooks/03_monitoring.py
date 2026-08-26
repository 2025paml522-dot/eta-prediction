# %%
import sys
from pathlib import Path

# Add the local monitoring module directory to the import path at runtime.
# This keeps the notebook working when launched from the project root or notebook folder.
monitoring_dir = Path(__file__).resolve().parent.parent / "src" / "monitoring"
sys.path.insert(0, str(monitoring_dir))

from log_predictions import log_prediction  # type: ignore[reportMissingImports]

request_id = log_prediction(
    features={
        "distance_km": 5.2,
        "weather_condition": "rain",
        "traffic_level": "high",
        "hour_of_day": 17,
        "is_rush_hour": True,
    },
    predicted_duration_min=22.4,
    model_version="dummy-v0",
)
print("Logged prediction:", request_id)
# %%
import pandas as pd
df = pd.read_json("../data/monitoring/predictions.jsonl", lines=True)
print(df.shape)
print(df.to_string())
# %%
