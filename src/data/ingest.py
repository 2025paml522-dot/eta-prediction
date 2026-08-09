"""Load trips_raw.csv, validate it, write clean + quarantined output """
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from validate_schema import validate


def run(raw_path: str, out_dir: str) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "rejected").mkdir(parents=True, exist_ok=True)

    print(f"[ingest] loading {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[ingest] loaded {len(df):,} rows")

    valid_df, rejected_df, summary_df = validate(df)
    print(summary_df.to_string(index=False))

    out_path = out_dir / "trips_validated.parquet"
    try:
        valid_df.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_dir / "trips_validated.csv"
        valid_df.to_csv(out_path, index=False)
        print("[ingest] WARNING: pyarrow not installed, wrote CSV instead", file=sys.stderr)

    if len(rejected_df) > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rejected_df.to_csv(out_dir / "rejected" / f"rejected_{ts}.csv", index=False)

    print(f"[ingest] {len(valid_df):,} valid -> {out_path}, {len(rejected_df):,} rejected")
    return {"n_valid": len(valid_df), "n_rejected": len(rejected_df)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--raw-path")
    parser.add_argument("--out-dir", default="data/interim")
    args = parser.parse_args()

    if args.raw_path:
        raw_path = args.raw_path
    elif args.config:
        raw_path = yaml.safe_load(open(args.config))["data"]["raw_path"]
    else:
        print("Pass --config or --raw-path", file=sys.stderr)
        sys.exit(1)

    run(raw_path, args.out_dir)


if __name__ == "__main__":
    main()