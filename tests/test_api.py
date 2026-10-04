from __future__ import annotations

import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


def _png_bytes(size: tuple[int, int] = (16, 16), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeClassifier(torch.nn.Module):
    """Returns a fixed logits row for every item in the batch, regardless of input."""

    def __init__(self, logits_row: list[float]) -> None:
        super().__init__()
        self._logits_row = torch.tensor(logits_row)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits_row.unsqueeze(0).repeat(x.shape[0], 1)


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


# ---------------------------------------------------------------------------
# issue #29 — GET /models metadata endpoint
# ---------------------------------------------------------------------------

def test_models_endpoint_lists_all_three_models() -> None:
    client = TestClient(app)

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    names = {entry["name"] for entry in payload}
    assert names == {"classifier", "anomaly_detector", "segmentation"}
    for entry in payload:
        assert "loaded" in entry
        assert "checkpoint_path" in entry
        assert "metadata" in entry


# ---------------------------------------------------------------------------
# issue #28 — POST /predict/batch
# ---------------------------------------------------------------------------

def test_predict_batch_scores_all_valid_images_in_one_pass(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "classifier", _FakeClassifier([5.0, 1.0, 0.0]))
    monkeypatch.setattr(main, "class_names", ["Forest", "SeaLake", "Highway"])
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes((8, 8), (200, 50, 50)), "image/png")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["failed"] == []
    for item in payload["results"]:
        assert item["predicted_class"] == "Forest"


def test_predict_batch_isolates_one_bad_file_into_failed(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "classifier", _FakeClassifier([5.0, 1.0, 0.0]))
    monkeypatch.setattr(main, "class_names", ["Forest", "SeaLake", "Highway"])
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("good.png", _png_bytes(), "image/png")),
            ("files", ("bad.png", b"not an image", "image/png")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 1
    assert len(payload["failed"]) == 1
    assert payload["failed"][0]["filename"] == "bad.png"


def test_predict_batch_rejects_over_max_batch_size(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main.settings, "max_batch_size", 1)
    monkeypatch.setattr(main, "classifier", _FakeClassifier([1.0, 0.0]))
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes(), "image/png")),
        ],
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# issue #17 — entropy + OOD plausibility gate on /predict
# ---------------------------------------------------------------------------

def test_predict_includes_entropy_and_likely_ood_fields(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "classifier", _FakeClassifier([5.0, 0.0, 0.0]))
    monkeypatch.setattr(main, "class_names", ["Forest", "SeaLake", "Highway"])
    monkeypatch.setattr(main, "autoencoder", None)  # OOD gate falls back to confidence-only
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert "entropy" in payload
    assert 0.0 <= payload["entropy"] <= 1.0
    assert "likely_ood" in payload
    assert payload["likely_ood"] is False  # high confidence, autoencoder not loaded


def test_predict_likely_ood_true_when_autoencoder_flags_anomaly(monkeypatch) -> None:
    from src.api import main

    monkeypatch.setattr(main, "classifier", _FakeClassifier([5.0, 0.0, 0.0]))
    monkeypatch.setattr(main, "class_names", ["Forest", "SeaLake", "Highway"])
    monkeypatch.setattr(main, "autoencoder", object())  # non-None => gate is consulted
    monkeypatch.setattr(
        "src.anomaly.compute_anomaly_score",
        lambda *a, **k: {"anomaly_score": 0.9, "is_anomaly": True, "threshold": 0.05},
    )
    client = TestClient(app)

    response = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})

    assert response.status_code == 200
    payload = response.json()
    # Classifier confidence is high, but the autoencoder plausibility check
    # flags the input as anomalous -> likely_ood must be True.
    assert payload["likely_ood"] is True


# ---------------------------------------------------------------------------
# issue #30 — async job queue for /segment and /pointcloud
# ---------------------------------------------------------------------------

def test_segment_async_job_flow(monkeypatch) -> None:
    from src.api import main

    async def _fake_segment_infer(tmp_path, confidence_threshold):
        return {"num_trees": 2, "detections": [], "masks_shape": [10, 10]}

    monkeypatch.setattr(main, "_segment_infer", _fake_segment_infer)
    client = TestClient(app)

    accepted = client.post("/segment/async", files={"file": ("x.png", _png_bytes())})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "done"
    assert body["result"]["num_trees"] == 2


def test_pointcloud_async_job_flow(monkeypatch) -> None:
    from src.api import main

    async def _fake_pointcloud_infer(tmp_path):
        return {
            "stats": {
                "num_points": 100, "bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1],
                "mean_canopy_height_m": 5.0, "max_canopy_height_m": 10.0,
                "canopy_cover_fraction": 0.5, "stem_density_per_ha": 20.0,
            },
            "trees": [],
            "num_trees_detected": 0,
        }

    monkeypatch.setattr(main, "_pointcloud_infer", _fake_pointcloud_infer)
    client = TestClient(app)

    accepted = client.post("/pointcloud/async", files={"file": ("x.las", b"fake lidar bytes")})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "done"


def test_job_not_found_returns_404() -> None:
    client = TestClient(app)
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404
