import argparse
import sys

import pandas as pd
import yaml

from src.models.config import (
CATEGORICAL_MAPS,FEATURE_COLUMNS,TARGET_COL
)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 16, 17, 18])

    # ---- categorical encodings ----
    df["traffic_level_enc"] = df["traffic_level"].map(CATEGORICAL_MAPS["traffic_level"]).fillna(-1)
    df["weather_condition_enc"] = df["weather_condition"].map(CATEGORICAL_MAPS["weather_condition"]).fillna(-1)

    # ---- missing weather numeric values ----
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0)

    keep_cols = FEATURE_COLUMNS + ([TARGET_COL] if TARGET_COL in df.columns else [])
    return df[keep_cols]


def run(interim_path: str, out_path: str):
    print(f"[features] loading {interim_path}")
    df = pd.read_parquet(interim_path)
    print(f"[features] loaded {len(df):,} rows")

    features_df = build_features(df)

    features_df.to_parquet(out_path, index=False)
    print(f"[features] wrote {len(features_df):,} rows -> {out_path}")
    print(f"[features] columns: {list(features_df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--interim-path")
    parser.add_argument("--out-path")
    args = parser.parse_args()

    if args.interim_path and args.out_path:
        interim_path, out_path = args.interim_path, args.out_path
    elif args.config:
        cfg = yaml.safe_load(open(args.config))["data"]
        interim_path = cfg["interim_path"]
        out_path = cfg["processed_path"]
    else:
        print("Pass --config, or both --interim-path and --out-path", file=sys.stderr)
        sys.exit(1)

    run(interim_path, out_path)


if __name__ == "__main__":
    main()