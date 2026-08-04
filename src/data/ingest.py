"""
Ingest raw trip data and run schema/quality validation (M2).

Responsibilities:
- Load raw CSV/API extract into a DataFrame.
- Validate schema: required columns, dtypes, value ranges.
- Flag/drop records with invalid GPS pings, missing timestamps, negative
  distances or durations.
- Write a clean, validated dataset to data/interim/ and register a new
  DVC-tracked version.

Usage:
    python src/data/ingest.py --config config/config.yaml
"""
import argparse

def load_raw(path: str):
    raise NotImplementedError

def validate_schema(df):
    """Apply Great Expectations / custom checks; raise on hard failures,
    log and quarantine on soft failures."""
    raise NotImplementedError

def main(config_path: str):
    raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
