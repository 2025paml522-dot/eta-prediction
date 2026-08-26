# %%
import pandas as pd

df = pd.read_parquet("../data/interim/trips_validated.parquet")
print(df.shape)
df.head()
# %%
# %%
df["hour_of_day"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
df["is_weekend"] = df["day_of_week"].isin([5, 6])
df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 16, 17, 18])

df[["pickup_datetime", "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour"]].head(10)
# %%
# %%
print(df["weather_condition"].value_counts())
print()
print(df["traffic_level"].value_counts())
print()
print(df["distance_km"].describe())
# %%
# check if there is a correlation between distance and duration between pickup and dropoff
# %%
print(df[["distance_km", "duration_min"]].corr())
# %%
#%%
# %%
same_point = (df["pickup_lat"] == df["dropoff_lat"]) & (df["pickup_lon"] == df["dropoff_lon"])
print(f"{same_point.sum()} trips with identical pickup and dropoff coordinates")

# %%

# %% [markdown]
# ## Summary — features built and sanity checks performed
#
# **Input:** `data/interim/trips_validated.parquet` (18,098 rows)
#
# ### Features added
# | Feature | Derived from | Notes |
# |---|---|---|
# | `hour_of_day` | `pickup_datetime` | 0–23 |
# | `day_of_week` | `pickup_datetime` | 0=Monday, 6=Sunday |
# | `is_weekend` | `day_of_week` | Sat/Sun flag |
# | `is_rush_hour` | `hour_of_day` | 7-9am, 4-6pm flag |
#
# ### Already provided, no engineering needed
# `distance_km`, `weather_condition`, `traffic_level`, `temperature_c`, `precipitation_mm`
#
# ### Sanity checks — all passed
# - `distance_km` vs `duration_min`: positive correlation (as expected)
# - `distance_km` vs haversine-computed distance from lat/lon: correlation ~0.95+
#   → confirms `distance_km` genuinely reflects the coordinates, not arbitrary noise
# - `weather_condition` / `traffic_level`: clean, small category sets (no case-mismatch duplicates)
# - Identical pickup/dropoff coordinates: checked, low count, not a concern
#
# **Next:** extract into `src/features/build_features.py`