"""
Train and compare candidate models, logging every run to MLflow (M3).

Trains at minimum:
- Baseline: Linear Regression
- Candidate: Gradient Boosting (XGBoost)

For each run, logs: hyperparameters, train/val metrics (MAE, RMSE, R2),
the fitted model artifact, and the exact feature-table version (DVC hash)
used, so any run is reproducible from its logged config alone.

Usage:
    python src/models/train.py --config config/config.yaml
"""
