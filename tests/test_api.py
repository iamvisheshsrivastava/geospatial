from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_index_serves_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Deforestation Classifier" in response.text


def test_health_includes_model_metadata() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "model_loaded" in payload
    assert "model_path" in payload
    assert "classes" in payload
