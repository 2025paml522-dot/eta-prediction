# Repo Structure Guide — Ride ETA Prediction

A quick orientation to the scaffold so everyone knows where their work goes.
Read this once, then keep it open as a reference for the first week or two.

## The 30-second version

The repo is organized around the **four course modules (M2–M5)**, which also
match the **four project weeks**. If you know which week's work you're doing,
you mostly know which folder you're in.

| Week | Module | You'll mainly work in |
|---|---|---|
| 1 | M2 — Data | `src/data/`, `src/features/`, `config/` |
| 2 | M3 — Experimentation | `src/models/`, `notebooks/` (exploration only) |
| 3 | M4 — Serving | `src/serving/`, `api/`, `docker/` |
| 4 | M5 — Monitoring | `src/monitoring/` |

Everyone touches `tests/` and `README.md` throughout — not just in week 4.

## Full tree

```
repo-root/
├── .github/workflows/ci.yml    # auto-runs tests on every push
├── .gitignore                  # tells git what NOT to track (data, secrets, etc.)
├── .env.example                # template for local secrets/config — copy to .env
├── Makefile                    # shortcuts: make train, make serve, make test...
├── README.md                   # front door — setup steps + quickstart
├── config/
│   └── config.yaml             # ⭐ all settings live here — paths, features, hyperparams
├── data/
│   ├── raw/                    # untouched source file (goes here, not committed to git)
│   ├── interim/                # after validation, before feature engineering
│   ├── processed/               # model-ready feature table
│   └── samples/                # tiny example request for testing the API
├── src/
│   ├── data/                    # Week 1: ingest.py, validate_schema.py
│   ├── features/                 # Week 1: build_features.py
│   ├── models/                   # Week 2: train.py, evaluate.py, register.py
│   ├── serving/                   # Week 3: predict.py, schemas.py
│   ├── monitoring/                 # Week 4: log_predictions.py, detect_drift.py
│   └── utils/                      # shared helpers anyone can add to
├── api/
│   ├── main.py                  # Week 3: the actual FastAPI app
│   └── tests/                    # tests for the API endpoints
├── docker/                       # Week 3: Dockerfile + docker-compose.yml
├── notebooks/                     # scratch space for exploration — never a dependency
├── tests/                          # unit tests, one file per src/ module
├── reports/figures/                 # charts/plots you generate (drift charts, comparisons)
├── mlruns/                           # local MLflow tracking data (auto-created, not committed)
├── dvc.yaml                          # defines the data pipeline stages for DVC
├── config/config.yaml                # (see above)
└── docs/
    ├── architecture-and-design.docx  # the full design doc
    ├── repo-structure.md              # longer version of this guide
    ├── references.md                  # team names, dataset citation, libraries used
    └── images/architecture_diagram.png
```

## What each top-level folder is *for*

**`src/`** — this is the actual pipeline code, organized by module (M2–M5).
Each subfolder is meant to be run as a standalone script during development
(`python src/data/ingest.py`), but also imported as a module by other stages
(e.g. `api/main.py` imports from `src/serving/`). If you're not sure where a
new function belongs, ask: *which module (M2–M5) does this serve?*

**`api/`** — deliberately separate from `src/serving/`. `src/serving/predict.py`
holds the actual "given a trip, predict ETA" logic; `api/main.py` is just the
thin FastAPI wrapper around it. This split means the prediction logic can be
tested without spinning up a server.

**`config/config.yaml`** — the single place for paths, feature lists,
hyperparameters, and thresholds. Every script reads from this file instead of
hardcoding values, so a run is reproducible just from the config + data
version + git commit. **If you're adding a new tunable value, put it here,
not inline in your script.**

**`data/`** — `raw/`, `interim/`, and `processed/` are **git-ignored on
purpose** (see `.gitignore`) and tracked with DVC instead, so the repo stays
small and everyone works from the same dataset version. Don't try to `git
add` anything in these folders directly.

**`tests/`** — mirrors `src/` one-to-one (`test_ingest.py` tests
`src/data/ingest.py`, etc.). Whoever writes a function in `src/` should add
at least a basic test for it in the matching file here.

**`notebooks/`** — free-for-all scratch space for exploration (EDA, trying
things out). Nothing in the pipeline should ever depend on a notebook —
once something works, move the logic into `src/`.

**`docker/`** and **`mlruns/`** — you generally won't touch these by hand.
`docker/docker-compose.yml` spins up the API + MLflow together; `mlruns/`
is where MLflow writes tracking data automatically when you run training.

**`docs/`** — where the design doc and this guide live. If you make an
architectural decision that isn't already written down, add a line to
`docs/architecture-and-design.docx` or open a short PR note — the brief
requires design decisions to be *justified in writing*, not just implemented.

## A few ground rules worth agreeing on now

- **Don't commit `data/raw|interim|processed`, `mlruns/`, or `.env`** — they're
  already in `.gitignore`, but double-check `git status` before committing if
  you're not sure something got picked up.
- **Work in `config/config.yaml`, not hardcoded values** — makes it much
  easier to compare each other's experiments later.
- **Commit incrementally, not in one big Week-4 dump** — the brief explicitly
  grades commit history, and it's much easier to debug a broken pipeline if
  changes land in small, reviewable steps.
- **If you add a new top-level folder or rename something**, update
  `docs/repo-structure.md` in the same commit so this doc doesn't go stale.

## Suggested ownership (adjust to your team)

| Area | Folder(s) | Suggested owner |
|---|---|---|
| Data & features | `src/data/`, `src/features/`, `config/` | _TBD_ |
| Modeling & tracking | `src/models/`, MLflow setup | _TBD_ |
| Serving & API | `src/serving/`, `api/`, `docker/` | _TBD_ |
| Monitoring & drift | `src/monitoring/` | _TBD_ |
| Docs & demo | `docs/`, `README.md`, final presentation | _TBD_ |

Ownership doesn't mean "only you touch it" — it means that person keeps an
eye on it, reviews PRs there, and is the go-to if something breaks.
