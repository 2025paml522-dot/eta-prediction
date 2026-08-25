"""
Basic load/latency test for the ETA prediction API.

Sends a batch of realistic requests to /predict and reports latency
percentiles (p50/p95/p99) plus throughput -- the "basic latency/
throughput awareness" .

Usage:
    python src/monitoring/load_test.py --n 100
    python src/monitoring/load_test.py --n 200 --url http://127.0.0.1:8000
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

# Reuse the same realistic trip-generation logic as simulate_drift.py,
# rather than duplicating it -- both files live in the same folder.
sys.path.insert(0, str(Path(__file__).parent))
from simulate_drift import make_trip  # noqa: E402


def run_load_test(base_url: str, n_requests: int, warmup: int = 3) -> dict:
    predict_url = f"{base_url}/predict"

    # A few unmeasured warmup requests -- the first request or two often
    # includes one-time costs (e.g. lazy imports, connection setup) that
    # would unfairly skew the latency numbers if included.
    print(f"[load_test] warming up ({warmup} requests, not counted)...")
    for _ in range(warmup):
        try:
            requests.post(predict_url, json=make_trip(0.1, 0.2, 20, 5), timeout=10)
        except requests.exceptions.ConnectionError:
            print(f"[load_test] Could not connect to {predict_url} -- is the API running?")
            sys.exit(1)

    print(f"[load_test] sending {n_requests} measured requests...")
    latencies_ms = []
    errors = 0

    for i in range(n_requests):
        trip = make_trip(storm_pct=0.1, rush_pct=0.3, temp_mean=18, temp_std=6)
        start = time.perf_counter()
        try:
            resp = requests.post(predict_url, json=trip, timeout=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                latencies_ms.append(elapsed_ms)
            else:
                errors += 1
                print(f"  [!] request {i}: HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"  [!] request {i}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{n_requests} done")

    if not latencies_ms:
        return {"error": "no successful requests -- check the API is running and reachable"}

    arr = np.array(latencies_ms)
    total_wall_time_sec = arr.sum() / 1000  # approximate, sequential requests

    report = {
        "n_requests_attempted": n_requests,
        "n_successful": len(latencies_ms),
        "n_errors": errors,
        "latency_ms": {
            "mean": round(float(arr.mean()), 2),
            "min": round(float(arr.min()), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(arr.max()), 2),
        },
        "approx_throughput_req_per_sec": round(len(latencies_ms) / total_wall_time_sec, 2) if total_wall_time_sec > 0 else None,
        "note": "Sequential (non-concurrent) requests. Reflects single-request "
                "latency and rough sequential throughput, not concurrent load "
                "capacity -- a tool like locust/hey would be needed for that.",
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=100, help="Number of measured requests")
    parser.add_argument("--out", default="reports/load_test_report.json")
    args = parser.parse_args()

    report = run_load_test(args.url, args.n)

    print("\n" + json.dumps(report, indent=2))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[load_test] report written to {args.out}")


if __name__ == "__main__":
    main()