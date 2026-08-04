"""
Feature engineering for ETA prediction (M2).

Builds:
- Time features: hour_of_day, day_of_week, is_weekend, is_rush_hour
- Spatial features: haversine distance, pickup/drop-off zone encoding
- Weather join (external source, keyed on timestamp + zone)

Writes a versioned feature table to data/processed/ and tags it with DVC.

Usage:
    python src/features/build_features.py --config config/config.yaml
"""
