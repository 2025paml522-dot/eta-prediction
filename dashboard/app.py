"""
Streamlit dashboard for the Delivery / Ride ETA Prediction project.

Run (with the FastAPI service already running):
    streamlit run dashboard/app.py

Tabs:
    1. Predict ETA        - call the live API with trip details
    2. Model Comparison    - M3 experiment tracking results
    3. Monitoring & Drift  - M5 prediction logs, drift report, retrain trigger
"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.config import (
    EXPERIMENTS_LOG, PREDICTIONS_LOG, DRIFT_REPORT_PATH, MODEL_METADATA_PATH,
)

API_URL = os.environ.get("ETA_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ETA Prediction Console", page_icon="🚗", layout="wide")
st.title("🚗 Delivery / Ride ETA Prediction — Ops Console")

tab_predict, tab_models, tab_monitor = st.tabs(
    ["🔮 Predict ETA", "📊 Model Comparison", "📈 Monitoring & Drift"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Predict
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Request a live ETA prediction")

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(
            f"API online — serving **{health.get('model_name')}** "
            f"(trained {health.get('model_trained_at')})"
        )
    except Exception:
        st.error(f"Could not reach API at {API_URL}. Start it with:\n\n`uvicorn api.main:app --reload`")

    col1, col2 = st.columns(2)
    with col1:
        pickup_date = st.date_input("Pickup date", datetime(2024, 6, 7))
        pickup_time = st.time_input("Pickup time", datetime(2024, 6, 7, 18, 30).time())
        distance_km = st.slider("Trip distance (km)", 0.5, 40.0, 4.2)
        passenger_count = st.number_input("Passenger count", 1, 8, 1)
    with col2:
        traffic_level = st.selectbox("Traffic level", ["low", "medium", "high", "severe"], index=1)
        weather_condition = st.selectbox("Weather", ["clear", "cloudy", "rain", "storm", "fog"])
        temperature_c = st.slider("Temperature (°C)", -10.0, 45.0, 20.0)
        precipitation_mm = st.slider("Precipitation (mm)", 0.0, 30.0, 0.0)

    st.markdown("**Pickup / dropoff coordinates**")
    c1, c2, c3, c4 = st.columns(4)
    pickup_lat = c1.number_input("Pickup lat", value=40.75, format="%.5f")
    pickup_lon = c2.number_input("Pickup lon", value=-73.98, format="%.5f")
    dropoff_lat = c3.number_input("Dropoff lat", value=40.77, format="%.5f")
    dropoff_lon = c4.number_input("Dropoff lon", value=-73.96, format="%.5f")

    if st.button("Predict ETA", type="primary"):
        payload = {
            "pickup_datetime": datetime.combine(pickup_date, pickup_time).isoformat(),
            "pickup_lat": pickup_lat, "pickup_lon": pickup_lon,
            "dropoff_lat": dropoff_lat, "dropoff_lon": dropoff_lon,
            "distance_km": distance_km,
            "weather_condition": weather_condition,
            "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm,
            "traffic_level": traffic_level,
            "passenger_count": int(passenger_count),
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            st.metric("Predicted ETA", f"{result['predicted_duration_min']} min")
            st.caption(f"model={result['model_name']} · request_id={result['request_id']}")
            with st.expander("Raw response"):
                st.json(result)
        except Exception as e:
            st.error(f"Prediction request failed: {e}")

# ----------------------------------------------------------------------------
# TAB 2 — Model comparison (M3 experiment tracking)
# ----------------------------------------------------------------------------
with tab_models:
    st.subheader("Experiment tracking — model comparison")

    if os.path.exists(EXPERIMENTS_LOG):
        exp_df = pd.read_csv(EXPERIMENTS_LOG)
        latest_run_time = exp_df["timestamp_utc"].max()
        latest_batch = exp_df[exp_df["timestamp_utc"] == latest_run_time]

        st.caption(f"Most recent training run: {latest_run_time}")
        st.dataframe(
            latest_batch[["model_name", "mae", "rmse", "r2", "train_time_sec"]]
            .sort_values("rmse").reset_index(drop=True),
            use_container_width=True,
        )

        fig = px.bar(latest_batch.sort_values("rmse"), x="model_name", y=["mae", "rmse"],
                     barmode="group", title="MAE / RMSE by model (lower is better)")
        st.plotly_chart(fig, use_container_width=True)

        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH) as f:
                meta = json.load(f)
            st.success(f"🏆 Currently deployed: **{meta['best_model_name']}** "
                       f"(MAE={meta['metrics']['mae']:.2f}, R²={meta['metrics']['r2']:.3f})")

        with st.expander("Full experiment history"):
            st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("No experiments logged yet. Run `python -m src.train`.")

# ----------------------------------------------------------------------------
# TAB 3 — Monitoring & drift (M5)
# ----------------------------------------------------------------------------
with tab_monitor:
    st.subheader("Prediction logs, drift detection & retraining trigger")

    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("🔄 Run drift check now"):
            try:
                from src.monitoring.detect_drift import run_drift_check
                run_drift_check()
                st.success("Drift check complete.")
            except Exception as e:
                st.error(f"Drift check failed: {e}")

    if os.path.exists(DRIFT_REPORT_PATH):
        with open(DRIFT_REPORT_PATH) as f:
            drift_report = json.load(f)

        if drift_report.get("retrain_recommended"):
            st.error(f"🚨 RETRAINING RECOMMENDED — reasons: {drift_report.get('reasons')}")
        elif drift_report.get("status") == "no_predictions_logged_yet":
            st.info("No predictions logged yet — send some via the Predict tab or the simulator script.")
        else:
            st.success("✅ No significant drift detected.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions logged", drift_report.get("n_predictions_total", 0))
        c1.metric("Max feature PSI", drift_report.get("max_psi", 0))
        perf = drift_report.get("performance_drift", {})
        c2.metric("Live MAE", perf.get("live_mae", "—"))
        c2.metric("Training-time test MAE", perf.get("test_mae", "—"))
        c3.metric("Labeled feedback samples", perf.get("n_labeled_samples", 0))
        c3.metric("Degradation ratio", perf.get("degradation_ratio", "—"))

        if drift_report.get("feature_drift_psi"):
            psi_df = pd.DataFrame(
                list(drift_report["feature_drift_psi"].items()), columns=["feature", "psi"]
            ).sort_values("psi", ascending=False)
            fig = px.bar(psi_df, x="feature", y="psi", title="Feature drift (PSI) vs. training distribution")
            fig.add_hline(y=0.1, line_dash="dot", annotation_text="warning (0.10)")
            fig.add_hline(y=0.25, line_dash="dash", annotation_text="critical (0.25)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift report yet. Click 'Run drift check now'.")

    st.markdown("---")
    st.markdown("**Recent prediction log**")
    if os.path.exists(PREDICTIONS_LOG):
        pred_df = pd.read_csv(PREDICTIONS_LOG)
        st.dataframe(pred_df.tail(200), use_container_width=True)

        labeled = pred_df.dropna(subset=["actual_duration_min"])
        if len(labeled) > 5:
            fig2 = px.scatter(
                labeled, x="predicted_duration_min", y="actual_duration_min",
                color="traffic_level", title="Predicted vs. actual duration (labeled feedback)",
            )
            max_val = max(labeled["predicted_duration_min"].max(), labeled["actual_duration_min"].max())
            fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                            line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No predictions logged yet.")
