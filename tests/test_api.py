"""
Integration Tests for Flask Backend REST API Endpoints.
"""

import json
import pytest
from backend.app import create_app
from backend.config import Config


@pytest.fixture
def client():
    app = create_app(Config)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_root_endpoint(client):
    response = client.get("/api/")
    assert response.status_code == 200
    data = response.get_json()
    assert "project" in data
    assert "LSTM + GloVe" in data["model"]
    assert "endpoints" in data


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_validation_empty_request(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_validation_non_json(client):
    response = client.post("/api/predict", data="not json", content_type="text/plain")
    assert response.status_code == 415


def test_predict_validation_blank_review(client):
    response = client.post("/api/predict", json={"review": "   "})
    assert response.status_code == 400
    data = response.get_json()
    assert "review" in data["error"].lower()


def test_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data


def test_static_frontend_serving(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Product Review Sentiment Analysis" in response.data
