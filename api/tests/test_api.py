"""
API contract tests for the ETA prediction service (M4).

Uses FastAPI's TestClient, which runs the app in-process -- no need for
a separately running uvicorn server. Note: these tests exercise the REAL
model_service (loads models/best_model.joblib), so `models/best_model.joblib`
must exist before running (it does, from Sprint 2 training).

These tests will append rows to logs/predictions_log.csv, same as any
real API call -- that's an accepted side effect, consistent with how
manual testing/the dashboard already behave.

Run with: pytest api/tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_TRIP = {
    "pickup_datetime": "2026-08-25T18:30:00",
    "pickup_lat": 40.7580,
    "pickup_lon": -73.9855,
    "dropoff_lat": 40.7128,
    "dropoff_lon": -74.0060,
    "distance_km": 4.2,
    "weather_condition": "rain",
    "temperature_c": 18.5,
    "precipitation_mm": 3.1,
    "traffic_level": "high",
    "passenger_count": 2,
}


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] is not None


def test_root_returns_service_info():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "service" in body
    assert "docs" in body


def test_predict_valid_request_returns_prediction():
    resp = client.post("/predict", json=VALID_TRIP)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_duration_min"] > 0
    assert "request_id" in body
    assert body["model_name"] is not None


def test_predict_missing_required_field_returns_422():
    bad_trip = dict(VALID_TRIP)
    del bad_trip["distance_km"]
    resp = client.post("/predict", json=bad_trip)
    assert resp.status_code == 422


def test_predict_negative_distance_returns_422():
    # distance_km has gt=0 in the schema -- this must be rejected before
    # ever reaching the model
    bad_trip = dict(VALID_TRIP)
    bad_trip["distance_km"] = -5.0
    resp = client.post("/predict", json=bad_trip)
    assert resp.status_code == 422


def test_predict_invalid_weather_condition_returns_422():
    # weather_condition is a Literal[...] -- anything outside the allowed
    # set should be rejected by Pydantic validation
    bad_trip = dict(VALID_TRIP)
    bad_trip["weather_condition"] = "tornado"
    resp = client.post("/predict", json=bad_trip)
    assert resp.status_code == 422


def test_predict_out_of_range_latitude_returns_422():
    # pickup_lat has ge=-90, le=90
    bad_trip = dict(VALID_TRIP)
    bad_trip["pickup_lat"] = 999.0
    resp = client.post("/predict", json=bad_trip)
    assert resp.status_code == 422


def test_model_info_returns_metadata():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "best_model_name" in body


def test_feedback_unknown_request_id_returns_404():
    resp = client.post("/feedback", json={
        "request_id": "this-id-does-not-exist-12345",
        "actual_duration_min": 20.0,
    })
    assert resp.status_code == 404


def test_predict_then_feedback_roundtrip():
    # Make a real prediction, then attach ground truth to it via /feedback
    predict_resp = client.post("/predict", json=VALID_TRIP)
    assert predict_resp.status_code == 200
    request_id = predict_resp.json()["request_id"]

    feedback_resp = client.post("/feedback", json={
        "request_id": request_id,
        "actual_duration_min": 19.5,
    })
    assert feedback_resp.status_code == 200
    assert feedback_resp.json()["status"] == "updated"


def test_feedback_rejects_non_positive_duration():
    # ActualDurationUpdate requires actual_duration_min > 0 (gt=0)
    predict_resp = client.post("/predict", json=VALID_TRIP)
    request_id = predict_resp.json()["request_id"]

    resp = client.post("/feedback", json={
        "request_id": request_id,
        "actual_duration_min": -1.0,
    })
    assert resp.status_code == 422