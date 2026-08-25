# Ride ETA Prediction — End-to-End ML Pipeline

Predicts delivery/ride ETA (trip duration) from distance, time-of-day,
weather, and traffic — with a validated data pipeline, tracked
experiments, a deployed REST API + dashboard, and drift monitoring.

**Best model:** Gradient Boosting — MAE ≈ 1.59 min, RMSE ≈ 2.03 min, R² ≈ 0.97

## Architecture

![architecture](docs/images/architecture_diagram.png)

Four layers, matching the four project modules:

| Layer | What it does | Where |
|---|---|---|
| **Data** | Ingest, validate, version raw trip data | `src/data/` |
| **Features** | Time features, cyclical encoding, categorical encoding | `src/features/` |
| **Modeling** | Train + compare 4 models, track experiments | `src/models/` |
| **Serving** | REST API + interactive dashboard | `api/`, `dashboard/` |
| **Monitoring** | Prediction logging, drift detection, retraining trigger | `src/monitoring/` |

Full design rationale: [`docs/architecture-and-design.docx`](docs/architecture-and-design.docx)

## Quickstart

```powershell
git clone <repo-url>
cd eta-prediction
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt

dvc pull                        # fetches the dataset (requires Google Drive Desktop, see below)

python src\data\ingest.py --config config\config.yaml
python src\features\build_features.py --config config\config.yaml
python src\models\train.py --config config\config.yaml

uvicorn api.main:app --reload   # in one terminal
streamlit run dashboard\app.py  # in a second terminal
```

Then open:
- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

## Setup, in detail

### 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Windows notes, if you hit these:**
- `activate` blocked by execution policy → run once:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
- Long path errors during install → enable Windows long paths, or keep
  the repo path short (e.g. `C:\eta-prediction`, not deeply nested)

### 2. Get the data

The dataset (`data/raw/trips_raw.csv`) is versioned with DVC and stored
on a shared Google Drive folder (synced via **Google Drive for
Desktop**, not DVC's OAuth backend — simpler to set up, no Google Cloud
project needed):

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/), sign into the shared account
2. Let it sync, then:
   ```powershell
   dvc pull
   ```

See [`docs/data-sources.md`](docs/data-sources.md) for the full dataset
schema and known data-quality characteristics (the raw file has ~2%
deliberately invalid rows, useful for exercising the validation pipeline).

### 3. Run the pipeline

```powershell
python src\data\ingest.py --config config\config.yaml
```
Validates and cleans the raw data. Expect **18,098 valid rows, 1,902
rejected** (quarantined with reasons in `data/interim/rejected/`, not
dropped silently).

```powershell
python src\features\build_features.py --config config\config.yaml
```
Adds time features, cyclical hour encoding, categorical encodings.
Output: `data/processed/train.parquet`.

```powershell
python src\models\train.py --config config\config.yaml
```
Trains and compares Linear Regression, Ridge, Random Forest, and
Gradient Boosting. Saves the best model to `models/best_model.joblib`
and logs a comparison to `reports/experiments_log.csv` (readable in the
dashboard's Model Comparison tab).

### 4. Serve

**Option A — directly:**
```powershell
uvicorn api.main:app --reload
```

**Option B — Docker:**
```powershell
docker compose -f docker/docker-compose.yml up --build
```
The Docker image uses a slim, serving-only dependency set
(`docker/requirements-api.txt`) — much faster to build than installing
the full training/dev toolchain into the container.

Either way, test it:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -InFile "data/samples/sample_request.json" | ConvertTo-Json
```

or with curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @data/samples/sample_request.json
```

**Sample response:**
```json
{
  "predicted_duration_min": 17.04,
  "model_name": "gradient_boosting",
  "model_version": "2026-08-22T06:35:13.203993+00:00",
  "request_id": "a4115b59-c260-47c5-954c-deafda72d158",
  "predicted_at_utc": "2026-08-25T09:35:14.877301+00:00"
}
```

### 5. Dashboard

```powershell
streamlit run dashboard\app.py
```

Three tabs:
- **Predict ETA** — live predictions through the real API, with a form for all trip fields
- **Model Comparison** — MAE/RMSE/R² across all 4 trained models
- **Monitoring & Drift** — prediction log, a live "Run drift check now" button, and retraining recommendation status

### 6. Monitoring & drift simulation

```powershell
python src\monitoring\simulate_drift.py     # sends normal + deliberately drifted traffic to /predict
python src\monitoring\detect_drift.py --reference data\processed\train.parquet --recent logs\predictions_log.csv --out reports\drift_report.json
```

`simulate_drift.py` sends a mix of normal trips plus a simulated
storm/rush-hour-surge scenario. `detect_drift.py` compares the recent
window against the training distribution using PSI (Population
Stability Index) per feature. On our own run, this correctly flagged
`temperature_c`, `precipitation_mm`, and `weather_condition` as
significantly drifted (PSI > 0.25).

Retraining trigger design (what fires retraining and why):
[`docs/retraining-trigger-design.md`](docs/retraining-trigger-design.md)

### 7. Tests

```powershell
pytest tests/ -v              # data + feature unit tests
pytest api/tests/ -v          # API contract tests (11 tests)
```

### 8. Load/latency test

```powershell
python src\monitoring\load_test.py --n 100
```
Sequential latency test against `/predict`. Our own run: **p50 24ms,
p95 45ms, ~36 req/sec**.

## Project structure

```
eta-prediction/
├── src/
│   ├── data/            # ingest.py, validate_schema.py
│   ├── features/         # build_features.py
│   ├── models/            # train.py, config.py
│   ├── monitoring/         # log_predictions.py, detect_drift.py, simulate_drift.py, load_test.py
│   └── serving/             # schemas.py
├── api/
│   ├── main.py            # FastAPI app
│   ├── model_loader.py     # loads the trained model, builds features from a request
│   └── tests/               # API contract tests
├── dashboard/
│   └── app.py              # Streamlit console (predict / compare / monitor)
├── docker/
│   ├── Dockerfile
│   ├── requirements-api.txt  # slim, serving-only deps
│   └── docker-compose.yml
├── config/config.yaml       # shared pipeline configuration
├── data/                     # raw/interim/processed (DVC-tracked, not in git)
├── models/                    # trained model artifact (DVC-tracked)
├── notebooks/                  # interactive .py scripts (# %% cells), not .ipynb
├── reports/                     # experiments log, drift reports, load test results
├── logs/                         # prediction log (runtime output)
└── docs/                          # architecture doc, data sources, this-and-that design docs
```

## Documentation

- [Architecture & Design](docs/architecture-and-design.docx) — full system design, rationale, weekly milestones
- [Data Sources](docs/data-sources.md) — dataset schema, known data-quality characteristics
- [Retraining Trigger Design](docs/retraining-trigger-design.md) — drift thresholds, retraining policy
- [Repo Structure Guide](docs/repo-structure.md) — annotated folder-by-folder walkthrough

## Known limitations

- Drift detection and retraining are demonstrated and designed, not
  scheduled — no automated job runs `detect_drift.py` periodically or
  triggers retraining on its own.
- The load test measures sequential (not concurrent) request latency —
  a tool like `locust` would be the next step for genuine concurrent
  load testing.
- MLflow's own run-tracking UI has a known registration issue (runs
  aren't queryable via `search_runs()`); worked around via a
  separate experiments log that the dashboard reads directly, which
  serves the same evidence purpose.

## Demo

[Link to 5–7 minute demo video — added on submission]