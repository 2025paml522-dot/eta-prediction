"""
FastAPI service exposing the ETA-prediction model (M4).

Endpoints:
    GET  /health          liveness/readiness probe
    POST /predict         single-trip ETA prediction
    GET  /model/info      currently loaded model version + metrics

Handles malformed input (422 via Pydantic validation), tracks basic
request latency, and logs every prediction for the monitoring pipeline.
"""
from fastapi import FastAPI
from src.serving.schemas import TripRequest, ETAResponse

app = FastAPI(title="ETA Prediction Service", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ETAResponse)
def predict(trip: TripRequest):
    raise NotImplementedError

@app.get("/model/info")
def model_info():
    raise NotImplementedError
