from __future__ import annotations

from contextlib import asynccontextmanager
import tempfile
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.data.preprocessing import preprocess_image
from src.models.resnet import build_resnet50_classifier
from src.models.autoencoder import SatelliteAutoencoder
from src.storage.s3 import download_file_from_s3


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class AnomalyResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    heatmap: list[list[float]]  # H×W, values in [0,1]


class ChangeDetectionResponse(BaseModel):
    change_score: float
    is_changed: bool
    threshold: float
    change_map: list[list[float]]  # H×W, values in [0,1]


class SegmentationDetection(BaseModel):
    box: list[float]        # [x1, y1, x2, y2]
    score: float
    mask_area_px: int


class SegmentationResponse(BaseModel):
    num_trees: int
    detections: list[SegmentationDetection]
    masks_shape: list[int]  # [H, W]


class PointCloudTreeStats(BaseModel):
    num_points: int
    bbox_min: list[float]
    bbox_max: list[float]
    mean_canopy_height_m: float
    max_canopy_height_m: float
    canopy_cover_fraction: float
    stem_density_per_ha: float


class PointCloudTree(BaseModel):
    tree_id: int
    centroid_xy: list[float]
    height_m: float
    crown_radius_m: float
    num_points: int


class PointCloudResponse(BaseModel):
    stats: PointCloudTreeStats
    trees: list[PointCloudTree]
    num_trees_detected: int


class HealthResponse(BaseModel):
    ok: bool
    classifier_loaded: bool
    anomaly_detector_loaded: bool
    segmentation_loaded: bool
    model_path: str
    autoencoder_path: str
    classes: list[str]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Satellite Geospatial ML Platform",
    description=(
        "Three capabilities in one API: "
        "(1) land-cover classification, "
        "(2) unsupervised anomaly detection, "
        "(3) temporal change detection."
    ),
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

classifier: torch.nn.Module | None = None
autoencoder: SatelliteAutoencoder | None = None
segmentation_model: torch.nn.Module | None = None
class_names: list[str] = []
autoencoder_image_size: int = settings.image_size
device = torch.device(settings.device)

# Default thresholds — tuned after training; can be overridden via query params
_DEFAULT_ANOMALY_THRESHOLD = 0.6
_DEFAULT_CHANGE_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_file(local_path: Path, s3_key: str | None) -> Path:
    if local_path.exists():
        return local_path
    if settings.s3_bucket and s3_key:
        return download_file_from_s3(settings.s3_bucket, s3_key, local_path, settings.aws_region)
    raise FileNotFoundError(
        f"Checkpoint not found at {local_path}. "
        "Set the path env var or provide S3_BUCKET + key."
    )


def load_classifier() -> None:
    global classifier, class_names
    path = _ensure_file(settings.model_path, settings.s3_model_key)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_resnet50_classifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    classifier = model


def load_anomaly_detector() -> None:
    global autoencoder, autoencoder_image_size
    from src.anomaly import load_autoencoder
    path = _ensure_file(settings.autoencoder_path, settings.s3_autoencoder_key)
    autoencoder, autoencoder_image_size = load_autoencoder(path, device)


def load_segmentation() -> None:
    """Load segmentation model on demand — called lazily to save RAM."""
    global segmentation_model
    from src.models.segmentation import load_segmentation_model
    path = _ensure_file(settings.segmentation_path, settings.s3_segmentation_key)
    segmentation_model = load_segmentation_model(path, device)


def unload_segmentation() -> None:
    """Free Mask R-CNN from memory after inference — it is large (~200 MB)."""
    global segmentation_model
    segmentation_model = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load only the two lightweight models at startup (~140 MB together).
    # Mask R-CNN (~200 MB) is loaded lazily on first /segment request and
    # unloaded afterwards to stay within Railway's 512 MB free-tier limit.
    for loader in (load_classifier, load_anomaly_detector):
        try:
            loader()
        except FileNotFoundError:
            pass  # server starts without models; /health reports status
    yield


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile, content: bytes) -> Path:
    suffix = Path(upload.filename or "upload.tif").suffix or ".tif"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Segmentation is lazy-loaded — report True if checkpoint file exists
    seg_available = settings.segmentation_path.exists() or bool(settings.s3_segmentation_key)
    return HealthResponse(
        ok=True,
        classifier_loaded=classifier is not None,
        anomaly_detector_loaded=autoencoder is not None,
        segmentation_loaded=seg_available,
        model_path=str(settings.model_path),
        autoencoder_path=str(settings.autoencoder_path),
        classes=class_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")
    content = await file.read()
    tmp_path = _save_upload(file, content)
    try:
        tensor = preprocess_image(tmp_path, settings.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        confidence, predicted_index = torch.max(probs, dim=0)
        return PredictionResponse(
            predicted_class=class_names[int(predicted_index)],
            confidence=round(float(confidence), 4),
            probabilities={name: round(float(probs[i]), 4) for i, name in enumerate(class_names)},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/anomaly", response_model=AnomalyResponse)
async def anomaly_detect(
    file: UploadFile = File(...),
    threshold: float = _DEFAULT_ANOMALY_THRESHOLD,
) -> AnomalyResponse:
    """Detect whether a satellite patch is anomalous (e.g. deforested, damaged).

    Uses an unsupervised convolutional autoencoder trained on normal land patches.
    High reconstruction error = the patch deviates from the learned normal distribution.
    """
    if autoencoder is None:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded.")
    content = await file.read()
    tmp_path = _save_upload(file, content)
    try:
        from src.anomaly import compute_anomaly_score
        result = compute_anomaly_score(autoencoder, tmp_path, autoencoder_image_size, device, threshold)
        return AnomalyResponse(
            anomaly_score=result["anomaly_score"],
            is_anomaly=result["is_anomaly"],
            threshold=result["threshold"],
            heatmap=result["heatmap"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Anomaly detection failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/change-detect", response_model=ChangeDetectionResponse)
async def change_detect(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    threshold: float = _DEFAULT_CHANGE_THRESHOLD,
) -> ChangeDetectionResponse:
    """Detect land-cover change between two satellite images of the same area.

    Compares deep features extracted by a pretrained ResNet-50 encoder.
    Feature-space comparison is robust to illumination and sensor differences.
    Returns a per-region change map and a scalar change score in [0, 1].
    """
    before_content = await before.read()
    after_content = await after.read()
    before_path = _save_upload(before, before_content)
    after_path = _save_upload(after, after_content)
    try:
        from src.change_detection import detect_change
        result = detect_change(before_path, after_path, settings.image_size, device, threshold)
        return ChangeDetectionResponse(
            change_score=result["change_score"],
            is_changed=result["is_changed"],
            threshold=result["threshold"],
            change_map=result["change_map"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Change detection failed: {exc}") from exc
    finally:
        before_path.unlink(missing_ok=True)
        after_path.unlink(missing_ok=True)


@app.post("/segment", response_model=SegmentationResponse)
async def segment(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
) -> SegmentationResponse:
    """Detect and segment individual tree crowns in aerial or satellite RGB imagery.

    Uses a Mask R-CNN (ResNet-50 + FPN) model. Returns per-tree bounding boxes,
    confidence scores, and mask areas. Designed for forestry inventory workflows.

    The model is loaded lazily on first call and unloaded after inference to
    conserve memory on constrained deployment environments (Railway 512 MB limit).
    The lightweight classifier and autoencoder are temporarily freed to make room.
    """
    import gc

    # Lazy-load segmentation model on first request.
    # Free the two always-resident models first so Mask R-CNN (~200 MB) fits in RAM.
    if segmentation_model is None:
        global classifier, autoencoder
        classifier = None
        autoencoder = None
        # Also free the change-detection ResNet encoder if it was loaded.
        try:
            from src.change_detection import unload_encoder as _unload_cd
            _unload_cd()
        except Exception:
            pass
        gc.collect()
        try:
            load_segmentation()
        except FileNotFoundError:
            # Reload the lightweight models before surfacing the error.
            try: load_classifier()
            except Exception: pass
            try: load_anomaly_detector()
            except Exception: pass
            raise HTTPException(status_code=503, detail="Segmentation model not loaded.")

    content = await file.read()
    tmp_path = _save_upload(file, content)
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
        pil = Image.open(tmp_path).convert("RGB")
        img_tensor = TF.to_tensor(pil).to(device)  # CHW float [0,1] — MaskRCNN normalises internally

        from src.models.segmentation import run_segmentation
        result = run_segmentation(segmentation_model, img_tensor, confidence_threshold)
        return SegmentationResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
        # Unload Mask R-CNN immediately after inference to free ~200 MB RAM,
        # then reload the lightweight models.
        unload_segmentation()
        gc.collect()
        try: load_classifier()
        except Exception: pass
        try: load_anomaly_detector()
        except Exception: pass


@app.post("/pointcloud", response_model=PointCloudResponse)
async def pointcloud_analyse(file: UploadFile = File(...)) -> PointCloudResponse:
    """Process a LiDAR point cloud (LAS/LAZ) and extract forest structure metrics.

    Returns:
      - Stand-level stats: canopy height, cover fraction, stem density per hectare.
      - Individual tree segments with height and crown radius estimates.

    This endpoint implements the core forest inventory pipeline used by
    airborne LiDAR survey companies.
    """
    content = await file.read()
    suffix = Path(file.filename or "scan.las").suffix or ".las"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        from src.pointcloud import process_las_file
        result = process_las_file(tmp_path)
        return PointCloudResponse(**result)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"LiDAR processing unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Point cloud processing failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
