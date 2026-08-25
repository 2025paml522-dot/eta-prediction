# Data Sources

## Primary dataset: `trips_raw.csv`

20,000 synthetic trip records, already including weather
(`weather_condition`, `temperature_c`, `precipitation_mm`) and traffic
(`traffic_level`) as columns — no separate weather dataset or join
required.

**Schema:**
```
trip_id, pickup_datetime, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
distance_km, weather_condition, temperature_c, precipitation_mm,
traffic_level, passenger_count, duration_min
```

Versioned with DVC, stored on a shared Google Drive remote (synced via
Google Drive for Desktop, not DVC's `gdrive://` OAuth backend — see
`.dvc/config` for the remote path). To get the data after cloning:

```bash
dvc pull
```

*(Requires Google Drive for Desktop installed and signed into the
shared account, so the remote path in `.dvc/config` resolves to a real
local folder.)*

## Known, deliberately-seeded data quality issues

The raw file contains ~2% intentionally invalid rows, useful for
exercising the validation pipeline (`src/data/validate_schema.py`):

| Issue | Rows | Handled by |
|---|---|---|
| Missing `pickup_lat` | ~90 | `no_missing_critical_fields` check |
| `pickup_datetime` = `"not-a-timestamp"` | ~108 | `timestamps_parseable` check |
| `distance_km` = `-5.0` (impossible) | ~102 | `distance_positive` check |
| `dropoff_lat` = `200.0` (impossible) | ~100 | `gps_within_nyc` check |
| Legitimately outside NYC bounding box | ~1,700 | `gps_within_nyc` check |

Running `python src/data/ingest.py --config config/config.yaml` against
the raw file produces **18,098 valid rows, 1,902 rejected** — rejected
rows are quarantined with a `reject_reason` column
(`data/interim/rejected/`), not silently dropped.

## Derived features

`src/features/build_features.py` adds, on top of the raw columns:

| Feature | Derived from |
|---|---|
| `hour_of_day`, `day_of_week`, `is_weekend`, `is_rush_hour` | `pickup_datetime` |
| `hour_sin`, `hour_cos` | cyclical encoding of `hour_of_day` |
| `traffic_level_enc`, `weather_condition_enc` | ordinal-mapped from the raw categorical columns |

Output: `data/processed/train.parquet`, also DVC-versioned.

## Model training data

`src/models/train.py` trains four model families (Linear Regression,
Ridge, Random Forest, Gradient Boosting) against `train.parquet`,
compares them by MAE, and saves the best one to
`models/best_model.joblib`. Current best: **Gradient Boosting**
(MAE ≈ 1.59, R² ≈ 0.97).
