"""
Drift simulation for the ETA prediction service.

Generates two batches of realistic trip requests and sends them to the
running API's /predict endpoint, so the resulting logs/predictions_log.csv
contains a genuine mix: mostly normal trips, plus a deliberately shifted
"storm + rush-hour surge" scenario, for detect_drift.py to catch.

Requires the API to be running first:
    uvicorn api.main:app --reload


"""
import random
from datetime import datetime, timedelta

import requests

API_URL = "http://127.0.0.1:8000/predict"

NYC_LAT_RANGE = (40.5, 40.9)
NYC_LON_RANGE = (-74.1, -73.8)


def random_point():
    return (
        round(random.uniform(*NYC_LAT_RANGE), 5),
        round(random.uniform(*NYC_LON_RANGE), 5),
    )


def make_trip(storm_pct: float, rush_pct: float, temp_mean: float, temp_std: float):
    """Build one trip request. storm_pct/rush_pct control how likely this
    trip is drawn from the 'unusual' conditions -- set both low for normal
    traffic, both high to simulate a drift scenario."""
    pickup_lat, pickup_lon = random_point()
    dropoff_lat, dropoff_lon = random_point()

    if random.random() < storm_pct:
        weather = "storm"
    else:
        weather = random.choices(
            ["clear", "cloudy", "rain", "fog"], weights=[0.5, 0.25, 0.15, 0.10]
        )[0]

    if random.random() < rush_pct:
        hour = random.choice([7, 8, 9, 16, 17, 18])
        traffic = random.choice(["high", "severe"])
    else:
        hour = random.randint(0, 23)
        traffic = random.choice(["low", "medium"])

    pickup_dt = datetime.now().replace(hour=hour, minute=random.randint(0, 59)) - timedelta(
        days=random.randint(0, 3)
    )

    return {
        "pickup_datetime": pickup_dt.isoformat(),
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "distance_km": round(random.gammavariate(3, 2), 2),
        "weather_condition": weather,
        "temperature_c": round(random.gauss(temp_mean, temp_std), 1),
        "precipitation_mm": round(random.expovariate(1 / 2), 1) if weather in ("rain", "storm") else 0.0,
        "traffic_level": traffic,
        "passenger_count": random.randint(1, 4),
    }


def send_batch(n: int, label: str, **scenario_kwargs):
    print(f"\n[simulate] sending {n} '{label}' requests...")
    ok, failed = 0, 0
    for _ in range(n):
        trip = make_trip(**scenario_kwargs)
        try:
            resp = requests.post(API_URL, json=trip, timeout=5)
            if resp.status_code == 200:
                ok += 1
            else:
                failed += 1
                print(f"  [!] {resp.status_code}: {resp.text[:150]}")
        except requests.exceptions.ConnectionError:
            print("  [!] Could not connect to API -- is `uvicorn api.main:app --reload` running?")
            return
    print(f"[simulate] '{label}' done: {ok} ok, {failed} failed")


def main():
    # Normal traffic: mostly clear weather, mild temps, occasional rush hour
    send_batch(40, "normal", storm_pct=0.05, rush_pct=0.25, temp_mean=20, temp_std=5)

    # Drift scenario: storm + rush-hour surge + cold snap
    send_batch(20, "DRIFT (storm + rush surge)", storm_pct=0.45, rush_pct=0.75, temp_mean=6, temp_std=3)

    print("\n[simulate] done. Run detect_drift.py next:")
    print("  python src/monitoring/detect_drift.py --reference data/processed/train.parquet --recent logs/predictions_log.csv")


if __name__ == "__main__":
    main()
