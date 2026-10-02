from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


def test_index_serves_ui() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Geospatial ML Platform" in response.text


def test_health_includes_model_metadata() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "classifier_loaded" in payload
    assert "anomaly_detector_loaded" in payload
    assert "model_path" in payload
    assert "classes" in payload


def test_predict_rejects_oversized_upload(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main.settings, "max_upload_mb", 1)
    monkeypatch.setattr(main, "classifier", object())
    client = TestClient(app)
    response = client.post("/predict", files={"file": ("x.png", b"0" * (2 * 1024 * 1024))})
    assert response.status_code == 413


def test_predict_rejects_garbage_bytes(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "classifier", object())
    client = TestClient(app)
    response = client.post("/predict", files={"file": ("x.png", b"not an image")})
    assert response.status_code == 400


def test_predict_503_when_classifier_missing(monkeypatch) -> None:
    """Null-check now happens inside the model lock (issue #31) — still a 503."""
    from src.api import main

    monkeypatch.setattr(main, "classifier", None)
    client = TestClient(app)
    response = client.post("/predict", files={"file": ("x.png", b"not an image")})
    assert response.status_code == 503


def test_anomaly_503_when_autoencoder_missing(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "autoencoder", None)
    client = TestClient(app)
    response = client.post("/anomaly", files={"file": ("x.png", b"not an image")})
    assert response.status_code == 503


def test_rate_limit_returns_429_after_threshold(monkeypatch) -> None:
    """Hammering a rate-limited endpoint past its per-minute cap returns 429."""
    from src.api import main

    monkeypatch.setattr(main.settings, "rate_limit_light_per_minute", 3)
    monkeypatch.setattr(main, "classifier", None)
    main._rate_limit_buckets.clear()
    client = TestClient(app)

    statuses = [
        client.post("/predict", files={"file": ("x.png", b"not an image")}).status_code
        for _ in range(5)
    ]

    assert statuses[:3] == [503, 503, 503]
    assert 429 in statuses[3:]
