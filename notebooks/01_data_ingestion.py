# %% [markdown]
# # 01 — Data Ingestion & Validation
# **Day 1–2 of Sprint 1 · Owner: Biju**
#
# Goal for this notebook: load the raw trip data, build a validation suite
# check-by-check, and end up with a `validate()` function + a runnable
# `ingest.py` — which we'll extract into `src/data/` once it works.
#
# Working directory assumption: this notebook lives in `notebooks/`, so raw
# data is at `../data/raw/` and outputs go to `../data/interim/`.

# %% [markdown]
# ## Step 1 — Load the raw file and just look at it

# %%
# %%
import pandas as pd

df = pd.read_csv("../data/raw/trips_raw.csv")
print(df.shape)
print(df.dtypes)
df.head()

# %% [markdown]
# Notice `pickup_datetime` and `dropoff_datetime` are `object` (plain strings),
# not real datetimes yet — that's the first thing Step 6 will fix.

# %% [markdown]
# ## Step 2 — Decide what "valid schema" means

# %%
REQUIRED_COLUMNS = [
    "trip_id", "pickup_datetime", "pickup_lat", "pickup_lon",
    "dropoff_lat", "dropoff_lon", "distance_km", "weather_condition",
    "temperature_c", "precipitation_mm", "traffic_level",
    "passenger_count", "duration_min",
]

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
print("Missing columns:", missing)

# %%
# Prove it catches a real problem: rename a column and re-check
typo_df = df.rename(columns={"vendor_id": "vendorId"})
missing_typo = [c for c in REQUIRED_COLUMNS if c not in typo_df.columns]
print("Missing columns after typo:", missing_typo)

# %% [markdown]
# ## Step 3 — First row-level check: missing critical fields

# %%
critical = [
    "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
    "duration_min", "distance_km",
]
no_missing = df[critical].notna().all(axis=1)
no_missing.value_counts()

# %% [markdown]
# `df[critical].notna()` gives a True/False grid (rows × those 7 columns).
# `.all(axis=1)` collapses each row to one bool — True only if *every* column
# in that row was True. This "grid → collapse to one bool per row" pattern
# repeats for every check below.

# %% [markdown]
# ## Step 4 — GPS bounds check

# %%
NYC_LAT_RANGE = (40.4, 41.1)
NYC_LON_RANGE = (-74.5, -73.3)

lat_ok = df["pickup_lat"].between(*NYC_LAT_RANGE) & df["dropoff_lat"].between(*NYC_LAT_RANGE)
lon_ok = df["pickup_lon"].between(*NYC_LON_RANGE) & df["dropoff_lon"].between(*NYC_LON_RANGE)
gps_ok = lat_ok & lon_ok
gps_ok.value_counts()

# %%
# Always look at the rows you're rejecting, not just the count
df[~gps_ok][["trip_id", "dropoff_lat"]].head(10)

# %% [markdown]
# ## Step 5 — Trip duration range check

# %%
MIN_DURATION_MIN, MAX_DURATION_MIN = 1, 180  # 1 min to 3 hrs

duration_ok = df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
duration_ok.value_counts()

# %% [markdown]
# **Design decision, not a fact:** why 10 seconds and not 0? A 1-second trip
# isn't a pandas error, but it's not a real taxi ride either. Write reasoning
# like this in a comment so teammates know a threshold was a deliberate call.

# %% [markdown]
# ## Step 6 — Parsing timestamps properly

# %%
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
timestamps_ok = df["pickup_datetime"].notna()
timestamps_ok.value_counts()

# %% [markdown]
# `errors="coerce"` is the key part: instead of crashing on a bad string,
# pandas turns it into `NaT` (Not a Time) and keeps going — which is what
# turns "unparseable timestamp" into something `.notna()` can catch.

# %% [markdown]
# ## Step 7 — Dropoff must be after pickup

# %%
distance_ok = df["distance_km"] > 0
distance_ok.value_counts()

# %% [markdown]
# ## Step 8 — Combine everything into one `validate()` function

# %%
def validate(raw_df: pd.DataFrame):
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    d = raw_df.copy()
    d["pickup_datetime"] = pd.to_datetime(d["pickup_datetime"], errors="coerce")

    checks = {
        "timestamps_parseable": d["pickup_datetime"].notna(),
        "no_missing_critical_fields": d[critical].notna().all(axis=1),
        "gps_within_nyc": (
            d["pickup_lat"].between(*NYC_LAT_RANGE) & d["dropoff_lat"].between(*NYC_LAT_RANGE)
            & d["pickup_lon"].between(*NYC_LON_RANGE) & d["dropoff_lon"].between(*NYC_LON_RANGE)
        ),
        "duration_in_range": d["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN),
        "distance_positive": d["distance_km"] > 0,
    }
    checks_df = pd.DataFrame(checks)
    overall_valid = checks_df.all(axis=1)

    def reasons(row):
        return ";".join(name for name, passed in row.items() if not passed)
    reject_reason = checks_df.apply(reasons, axis=1)

    valid_df = d[overall_valid].copy()
    rejected_df = d[~overall_valid].copy()
    rejected_df["reject_reason"] = reject_reason[~overall_valid]

    return valid_df, rejected_df, checks_df

# %%
# reload fresh, since df above already had timestamps parsed in Step 6
df_fresh = pd.read_csv("../data/raw/trips_raw.csv")
valid_df, rejected_df, checks_df = validate(df_fresh)
print(f"{len(valid_df)} valid, {len(rejected_df)} rejected")
rejected_df[["trip_id", "reject_reason"]].head(10)

# %% [markdown]
# ## Step 9 — Wire it into a runnable ingest step

# %%
from pathlib import Path
from datetime import datetime

def run(raw_path, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "rejected").mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(raw_path)
    valid_df, rejected_df, _ = validate(raw_df)

    valid_df.to_csv(out_dir / "trips_validated.csv", index=False)

    if len(rejected_df) > 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rejected_df.to_csv(out_dir / "rejected" / f"rejected_{ts}.csv", index=False)

    print(f"{len(valid_df)} valid, {len(rejected_df)} rejected -> written to {out_dir}")
    return valid_df, rejected_df

valid_df, rejected_df = run("../data/raw/trips_raw.csv", "../data/interim")

# %% [markdown]
# Go check `data/interim/rejected/rejected_*.csv` in a text editor — your own
# `reject_reason` column is sitting there. That auditability is the whole
# point of quarantining instead of silently dropping rows.

# %% [markdown]
# ## Step 10 — Try it yourself
#
# Add one more check: `passenger_count` should be between 1 and 6 (taxi
# capacity). Same pattern as every check above — write the boolean Series,
# add it to the `checks` dict inside `validate()`, re-run, and look at what
# gets newly rejected.

# %%
# Your turn -- write the passenger_count check here
# passenger_count_ok = df_fresh["passenger_count"].between(1, 6)
# passenger_count_ok.value_counts()

# %% [markdown]
# ---
# ## Next: extract into `src/data/`
#
# Once this notebook runs clean top-to-bottom, the `validate()` and `run()`
# functions above get copied into `src/data/validate_schema.py` and
# `src/data/ingest.py` respectively — same logic, just import-able and
# testable outside the notebook. That's tomorrow's task (Day 3).
#
# **Before committing this notebook:** clear outputs so diffs stay readable —
# ```
# jupyter nbconvert --clear-output --inplace notebooks/01_data_ingestion.ipynb
# ```
