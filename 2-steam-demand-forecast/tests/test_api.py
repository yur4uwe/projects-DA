"""Unit tests for FastAPI HTTP forecasting service."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_api_health() -> None:
    with TestClient(app) as tc:
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "hybrid" in data["loaded_models"]
        assert data["catalog_size"] > 0


def test_api_catalog() -> None:
    with TestClient(app) as tc:
        resp = tc.get("/catalog")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 6
        appids = [item["appid"] for item in items]
        assert 367520 in appids  # Hollow Knight


def test_api_forecast_endpoint() -> None:
    with TestClient(app) as tc:
        payload = {
            "appid": 367520,
            "horizon_days": 7,
            "discount_depth": 0.33,
            "sale_duration_days": 4,
            "is_custom_event": True,
            "custom_event_name": "API Test Deal",
            "model_engine": "hybrid",
        }
        resp = tc.post("/forecast", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["appid"] == 367520
        assert data["horizon_days"] == 7
        assert len(data["forecast_series"]) == 7
        assert data["peak_promo_ccu"] > 0
