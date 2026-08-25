# Build Log

A record of how this project was actually built, day by day. Written
retrospectively rather than as a plan — useful as evidence of
incremental progress, and as a reference for what decisions were made
and why.

## Sprint 1 — Data pipeline (Aug 8–14)

- **Day 1–3:** Built and tested data validation logic interactively
  (`notebooks/01_data_ingestion.py`, cell-based `.py` file, not a Jupyter
  notebook — see note below on tooling). Five checks: schema, missing
  fields, GPS bounds, duration range, distance positivity. Verified
  against `trips_raw.csv`, confirmed **18,098 valid / 1,902 rejected**.
- **Day 4:** Extracted the validated logic into `src/data/validate_schema.py`
  and `src/data/ingest.py`, added `tests/test_ingest.py`.
- **Day 5:** Set up DVC. Hit a `pathspec` version-compatibility bug on
  `dvc init` (fixed by pinning `pathspec==0.12.1`), then a Google OAuth
  wall on `dvc push` to a `gdrive://` remote — worked around by using
  Google Drive for Desktop (synced local folder) as a plain local DVC
  remote instead, avoiding the OAuth setup entirely.
- **Day 6–7:** Feature engineering (`src/features/build_features.py`) —
  time features, cyclical hour encoding, categorical encodings.
  `data/processed/train.parquet` versioned with DVC.

**Tooling note:** started with Jupyter notebooks, switched to plain
`.py` files with `# %%` cell markers (VS Code's Python Interactive
Window) after the first session — notebooks don't diff or merge
cleanly in git, and clearing outputs before every commit was an easy
step to forget. The `.py` files give the same cell-by-cell workflow
without the JSON-diff problem.

## Sprint 2 — Modeling, serving, monitoring (Aug 15 onward)

- Model training (`src/models/train.py`) compares Linear Regression,
  Ridge, Random Forest, and Gradient Boosting. Best model: **Gradient
  Boosting** (MAE ≈ 1.59, RMSE ≈ 2.03, R² ≈ 0.97), saved to
  `models/best_model.joblib`.
- API (`api/main.py`) serves `/predict`, `/health`, `/model/info`,
  `/feedback` via FastAPI. Verified with real requests (curl,
  `Invoke-RestMethod`, and Swagger UI).
- Monitoring (`src/monitoring/`): prediction logging
  (`log_predictions.py`), drift detection via PSI
  (`detect_drift.py`), and a drift-simulation script
  (`simulate_drift.py`) that generates a deliberate storm/rush-hour-surge
  scenario. Verified against real training data — correctly flagged
  `temperature_c`, `precipitation_mm`, and `weather_condition` as
  significantly drifted (PSI > 0.25 on all three).

## Team change (Aug 23)

Karthikeya stepped away from the project partway through Sprint 2.
Remaining work (serving, monitoring, and Karthikeya's originally-planned
tasks) was absorbed and completed solo. Timeline was extended to Aug 31
to accommodate this.

## Post-Sprint-2 hardening (Aug 23–31)

With extra time available, went back and closed several gaps rather
than leaving them as known limitations:

- **Streamlit dashboard** (`dashboard/app.py`) — added by course
  automation (`2025paml596-ops`, a legitimate collaborator, confirmed
  via GitHub's Manage Access page). Three tabs: live predictions, model
  comparison, and drift detection with a live "run check now" button.
  Fixed two real bugs in it: a missing `run_drift_check()` function the
  button called but didn't exist, and a schema mismatch between what
  that function wrote and what the dashboard's metrics display expected.
- **Docker** — fixed the Dockerfile (wasn't copying `models/`, so the
  container would have failed to start), and replaced the top-level
  `requirements.txt` with a slim `docker/requirements-api.txt`
  containing only what the API actually needs at runtime (FastAPI,
  pandas, scikit-learn, etc.) — the original approach was pulling in
  ~300MB of training-only dependencies (xgboost, the full MLflow
  server, DVC, Jupyter) into a container that never uses any of them,
  causing 30+ minute builds that timed out on a slower connection.
- **API tests** (`api/tests/test_api.py`) — 11 tests using FastAPI's
  `TestClient`, covering health/predict/feedback/model-info, valid and
  invalid input (422 cases), and a full predict→feedback round trip.
- **Load test** (`src/monitoring/load_test.py`) — basic sequential
  latency/throughput measurement against `/predict`. Result: p50 24ms,
  p95 45ms, ~36 req/sec sequential.
- **MLflow's run-tracking UI** never got fully working (a `meta.yaml`
  registration issue meant runs weren't queryable via
  `search_runs()`), but this didn't block anything — `train.py` writes
  a separate experiments log that the dashboard's Model Comparison tab
  reads directly, which serves the same evidence purpose and is what's
  used in the demo.

## Known limitations, stated plainly

- Drift detection and retraining are demonstrated and designed, not
  scheduled/automated — there's no cron job actually running
  `detect_drift.py` periodically or triggering retraining on its own.
- The load test is sequential, not concurrent — it measures
  single-request latency characteristics, not concurrent request
  handling capacity.
- MLflow's own run-tracking UI has a known registration bug (see
  above); worked around rather than fixed, since a working alternative
  (the dashboard) already existed.