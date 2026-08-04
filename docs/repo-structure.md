# Repository Structure — Annotated

```
repo-root/
├── .github/workflows/ci.yml   # lint + pytest on every push/PR
├── .gitignore                 # excludes raw/interim/processed data & mlruns (tracked via DVC instead)
├── .env.example                # template for MLFLOW_TRACKING_URI, ports, etc.
├── Makefile                    # make ingest / features / train / serve / test / monitor
├── README.md                   # setup + quickstart
├── config/
│   └── config.yaml             # single source of truth: paths, feature list, hyperparams, thresholds
├── data/
│   ├── raw/                    # immutable source extract (DVC-tracked, git-ignored)
│   ├── interim/                # post-validation, pre-feature-engineering
│   ├── processed/              # model-ready feature table (DVC-tracked)
│   └── samples/                # small JSON payloads for manual/API testing
├── src/
│   ├── data/                   # M2: ingest.py, validate_schema.py
│   ├── features/                # M2: build_features.py
│   ├── models/                  # M3: train.py, evaluate.py, register.py
│   ├── serving/                  # M4: predict.py, schemas.py (used by api/)
│   ├── monitoring/                # M5: log_predictions.py, detect_drift.py, retrain_trigger.py
│   └── utils/                     # shared helpers (io, logging)
├── api/
│   ├── main.py                 # FastAPI app: /health, /predict, /model/info
│   └── tests/test_api.py       # API contract tests
├── docker/
│   ├── Dockerfile               # packages api/ + src/ into a single image
│   └── docker-compose.yml       # api + mlflow services
├── notebooks/                   # exploratory work only — nothing here is a pipeline dependency
├── tests/                       # unit tests mirroring src/ (pytest)
├── reports/
│   ├── metrics.json (generated) # DVC metrics output, tracked for `dvc metrics diff`
│   └── figures/                 # model comparison plots, drift charts
├── mlruns/                      # local MLflow tracking store (git-ignored; use a remote store for teams)
├── dvc.yaml                     # DVC pipeline: ingest -> features -> train, wired to the same config.yaml
└── docs/
    ├── architecture-and-design.docx
    ├── repo-structure.md
    ├── references.md
    └── images/architecture_diagram.png
```

## Design rationale
- **`config/config.yaml` is the single source of truth.** Every stage (ingest,
  features, train, serve, monitor) reads from it, so a run is reproducible
  from the config + DVC data hash + Git commit alone — directly supporting
  the M3 rubric criterion on reproducibility.
- **`src/` vs `api/` split.** `src/serving/predict.py` holds the inference
  logic; `api/main.py` is a thin FastAPI wrapper. This keeps the model logic
  testable without spinning up a server, and lets the same `predict()` be
  reused in batch-scoring scripts later.
- **Data is DVC-tracked, not Git-tracked.** `.gitignore` excludes
  `data/raw|interim|processed`; DVC hashes point to remote storage, keeping
  the Git history small while still giving every commit a resolvable data
  version.
- **`mlruns/` stays local by default** but `docker-compose.yml` runs MLflow
  as its own service so the team can point `MLFLOW_TRACKING_URI` at a shared
  server without code changes.
- **Tests mirror `src/`** one-to-one so coverage gaps are obvious, and CI
  runs them on every push per the "meaningful, incremental commits" and
  academic-integrity expectations in the brief.
