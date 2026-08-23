"""
FastAPI service exposing the ETA-prediction model (M4).

Endpoints:
    GET  /health          liveness/readiness probe
    POST /predict         single-trip ETA prediction
    GET  /model/info      currently loaded model version + metrics

Handles malformed input (422 via Pydantic validation), tracks basic
request latency, and logs every prediction for the monitoring pipeline.
uvicorn api.main:app --reload
"""
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import model_service
from src.serving.schemas import TripRequest, PredictionResponse, ActualDurationUpdate, HealthResponse
from src.models.config import PREDICTIONS_LOG, LOGS_DIR
from src.monitoring.log_predictions import log_prediction

app = FastAPI(
    title="Delivery / Ride ETA Prediction API",
    description="Predicts trip duration (ETA) from distance, time, weather and traffic features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _log_prediction(request_id: str, trip: TripRequest, prediction: float):
    os.makedirs(LOGS_DIR, exist_ok=True)
    row = trip.model_dump()
    row["pickup_datetime"] = trip.pickup_datetime.isoformat()
    row.update({
        "request_id": request_id,
        "predicted_duration_min": prediction,
        "actual_duration_min": None,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_service.metadata.get("best_model_name"),
    })
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(PREDICTIONS_LOG)
    df_row.to_csv(PREDICTIONS_LOG, mode="a", header=write_header, index=False)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_name=model_service.metadata.get("best_model_name"),
        model_trained_at=model_service.metadata.get("trained_at_utc"),
    )


@app.get("/model/info")
def model_info():
    return model_service.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest):
    try:
        prediction = model_service.predict(trip)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    request_id = str(uuid.uuid4())
    trip_fields = trip.model_dump()
    trip_fields["pickup_datetime"] = trip.pickup_datetime.isoformat()
    log_prediction(
        request_id=request_id,
        trip_fields=trip_fields,
        prediction=prediction,
        model_name=model_service.metadata.get("best_model_name"),
        log_path=PREDICTIONS_LOG,
    )

    return PredictionResponse(
        predicted_duration_min=round(prediction, 2),
        model_name=model_service.metadata.get("best_model_name", "unknown"),
        model_version=model_service.metadata.get("trained_at_utc", "unknown"),
        request_id=request_id,
        predicted_at_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/feedback")
def feedback(update: ActualDurationUpdate):
    """Attach the real observed trip duration to a previous prediction.
    This is what M5's drift monitor compares against."""
    if not os.path.exists(PREDICTIONS_LOG):
        raise HTTPException(status_code=404, detail="No predictions logged yet")

    df = pd.read_csv(PREDICTIONS_LOG)
    mask = df["request_id"] == update.request_id
    if not mask.any():
        raise HTTPException(status_code=404, detail="request_id not found")

    df.loc[mask, "actual_duration_min"] = update.actual_duration_min
    df.to_csv(PREDICTIONS_LOG, index=False)
    return {"status": "updated", "request_id": update.request_id}


@app.get("/")
def root():
    return {
        "service": "Delivery / Ride ETA Prediction API",
        "docs": "/docs",
        "health": "/health",
    }