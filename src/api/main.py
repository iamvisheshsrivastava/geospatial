from __future__ import annotations

from contextlib import asynccontextmanager
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator

# Heroku always injects the DYNO env var (e.g. "web.1").
# On any other host (local Docker, Railway, etc.) it is absent.
# We use this to gate the segmentation endpoint: Mask R-CNN needs ~500 MB
# alongside the other models, which exceeds Heroku's 512 MB dyno limit and
# triggers an OS-level OOM kill that brings down the entire process.
_HEROKU = bool(os.getenv("DYNO"))

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


class ExplainResponse(BaseModel):
    predicted_class: str
    confidence: float
    gradcam_b64: str          # base64 PNG — GradCAM overlay on original image


class SpectralResponse(BaseModel):
    vegetation_b64: str       # base64 PNG — VARI heatmap (RdYlGn)
    water_b64: str            # base64 PNG — ExWI heatmap (Blues)
    urban_b64: str            # base64 PNG — ExUI heatmap (Oranges)
    vegetation_mean: float
    water_mean: float
    urban_mean: float
    interpretation: str


class PovertyContributor(BaseModel):
    class_name: str
    probability: float
    contribution: float
    weight: float


class PovertyResponse(BaseModel):
    wealth_index: float       # [0, 1]
    wealth_label: str         # Very Low … Very High
    wealth_color: str         # hex colour for UI badge
    interpretation: str
    top_contributors: list[dict]
    methodology: str


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

# Default thresholds — overridden at load time by the 95th-percentile value from the checkpoint
_DEFAULT_ANOMALY_THRESHOLD = 0.05
_DEFAULT_CHANGE_THRESHOLD = 0.15

classifier: torch.nn.Module | None = None
autoencoder: SatelliteAutoencoder | None = None
segmentation_model: torch.nn.Module | None = None
class_names: list[str] = []
autoencoder_image_size: int = settings.image_size
autoencoder_threshold: float = _DEFAULT_ANOMALY_THRESHOLD
device = torch.device(settings.device)


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
    global autoencoder, autoencoder_image_size, autoencoder_threshold
    from src.anomaly import load_autoencoder
    path = _ensure_file(settings.autoencoder_path, settings.s3_autoencoder_key)
    autoencoder, autoencoder_image_size, autoencoder_threshold = load_autoencoder(path, device)


def load_segmentation() -> None:
    """Load segmentation model on demand — called lazily to save RAM.

    After loading, the model's internal GeneralizedRCNN transform is constrained
    to a small image size (256 px) so that intermediate feature-map activations
    during inference stay around 15–20 MB instead of the default ~200 MB that
    the standard 800×1333 transform produces.  Detection quality is reduced, but
    the endpoint stays alive on a 512 MB Heroku dyno.
    """
    global segmentation_model
    from src.models.segmentation import load_segmentation_model
    path = _ensure_file(settings.segmentation_path, settings.s3_segmentation_key)
    model = load_segmentation_model(path, device)
    # Constrain internal resize so activations fit in a 512 MB dyno.
    # Default: min_size=800, max_size=1333 → ~200 MB activations.
    # Reduced: min_size=256, max_size=320 → ~15 MB activations.
    if hasattr(model, "transform"):
        model.transform.min_size = (256,)
        model.transform.max_size = 320
    segmentation_model = model


def unload_segmentation() -> None:
    """Free Mask R-CNN from memory after inference — it is large (~200 MB)."""
    global segmentation_model
    segmentation_model = None
    if device.type == "cuda":
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Limit PyTorch to 2 threads — shared CPU environments benefit from
    # capped thread counts to avoid contention that slows inference down.
    torch.set_num_threads(2)

    # Load only the two lightweight models at startup (~140 MB together).
    # Mask R-CNN (~200 MB) is loaded lazily on first /segment request and
    # unloaded afterwards to conserve memory on constrained deployments.
    for loader in (load_classifier, load_anomaly_detector):
        try:
            loader()
        except Exception as e:
            # Missing file or incompatible checkpoint — start anyway, endpoint
            # returns 503 until a compatible checkpoint is deployed.
            print(f"WARNING: {loader.__name__} failed — {e}")

    # Warm up the classifier with a dummy forward pass so the first real
    # request doesn't pay the JIT/kernel-load penalty.
    if classifier is not None:
        try:
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            with torch.inference_mode():
                classifier(dummy)
        except Exception:
            pass

    yield


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

async def _read_upload_capped(upload: UploadFile) -> bytes:
    """Read an UploadFile in bounded chunks, rejecting once it exceeds MAX_UPLOAD_MB.

    Reading in chunks (rather than `await upload.read()`) means an oversized
    upload is rejected as soon as the cap is crossed instead of first being
    buffered into memory in full.
    """
    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _save_upload(upload: UploadFile, content: bytes) -> Path:
    suffix = Path(upload.filename or "upload.tif").suffix or ".tif"
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(content)
        tmp.close()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid upload: {exc}") from exc
    return Path(tmp.name)


def _cgroup_available_mb() -> float | None:
    """Return available memory in MB from the container's cgroup limit, or None.

    Returns None when no cgroup memory-limit file is readable (e.g. not running
    inside a container), so callers can fall back to a host-level check.
    """
    try:
        # cgroup v2 (Docker/Railway/k8s default on modern kernels)
        max_path = Path("/sys/fs/cgroup/memory.max")
        cur_path = Path("/sys/fs/cgroup/memory.current")
        if max_path.exists() and cur_path.exists():
            max_raw = max_path.read_text().strip()
            if max_raw != "max":  # "max" means no limit set — not useful here
                limit = int(max_raw)
                current = int(cur_path.read_text().strip())
                return (limit - current) / (1024 * 1024)
    except Exception:
        pass

    try:
        # cgroup v1 fallback (older Docker/Heroku)
        limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        usage_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        if limit_path.exists() and usage_path.exists():
            limit = int(limit_path.read_text().strip())
            # cgroup v1 reports a huge sentinel (close to 2**63) for "unlimited"
            if limit < (1 << 62):
                usage = int(usage_path.read_text().strip())
                return (limit - usage) / (1024 * 1024)
    except Exception:
        pass

    return None


def _host_available_mb() -> float:
    """Return available memory in MB from /proc/meminfo (host-level, not container-aware)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass  # Not Linux or /proc unavailable
    return 0.0


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
    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        tensor = preprocess_image(tmp_path, settings.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
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
    threshold: float | None = None,
) -> AnomalyResponse:
    """Detect whether a satellite patch is anomalous (e.g. deforested, damaged).

    Uses an unsupervised convolutional autoencoder trained on normal land patches.
    High reconstruction error = the patch deviates from the learned normal distribution.
    """
    if autoencoder is None:
        raise HTTPException(status_code=503, detail="Anomaly detector not loaded.")
    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        from src.anomaly import compute_anomaly_score
        effective_threshold = threshold if threshold is not None else autoencoder_threshold
        result = compute_anomaly_score(autoencoder, tmp_path, autoencoder_image_size, device, effective_threshold)
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
    before_content = await _read_upload_capped(before)
    after_content = await _read_upload_capped(after)
    tmp_paths: list[Path] = []
    try:
        before_path = _save_upload(before, before_content)
        tmp_paths.append(before_path)
        after_path = _save_upload(after, after_content)
        tmp_paths.append(after_path)

        from src.change_detection import detect_change
        result = detect_change(before_path, after_path, settings.image_size, device, threshold)
        return ChangeDetectionResponse(
            change_score=result["change_score"],
            is_changed=result["is_changed"],
            threshold=result["threshold"],
            change_map=result["change_map"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Change detection failed: {exc}") from exc
    finally:
        for _p in tmp_paths:
            _p.unlink(missing_ok=True)


@app.post("/segment", response_model=SegmentationResponse)
async def segment(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
) -> SegmentationResponse:
    """Detect and segment individual tree crowns in aerial or satellite RGB imagery.

    Uses a Mask R-CNN (ResNet-50 + FPN) model. Returns per-tree bounding boxes,
    confidence scores, and mask areas. Designed for forestry inventory workflows.

    The model is loaded lazily on first call and unloaded after inference to
    conserve memory on constrained deployment environments.
    The lightweight classifier and autoencoder are temporarily freed to make room.
    """
    import gc

    # ── Heroku guard ─────────────────────────────────────────────────────────
    # Mask R-CNN needs ~500 MB alongside Python/torch/other models.
    # Heroku Eco dynos have 512 MB total — attempting to load causes the OS
    # OOM-killer to SIGKILL the entire process, taking down every other
    # endpoint with it.  Return a fast JSON 503 before touching any model.
    if _HEROKU:
        raise HTTPException(
            status_code=503,
            detail=(
                "Tree crown segmentation requires ~500 MB RAM for Mask R-CNN. "
                "The free Heroku dyno (512 MB total) cannot run all models at once — "
                "loading it would crash the entire application. "
                "Clone the repo and run `docker compose up` locally for the full demo."
            ),
        )
    # ─────────────────────────────────────────────────────────────────────────

    # Lazy-load segmentation model on first request.
    if segmentation_model is None:
        # Free the lightweight models first to make room for Mask R-CNN.
        global classifier, autoencoder
        classifier = None
        autoencoder = None
        try:
            from src.change_detection import unload_encoder as _unload_cd
            _unload_cd()
        except Exception:
            pass
        gc.collect()

        # ── Pre-flight memory guard ──────────────────────────────────────────
        # Loading Mask R-CNN needs ~180 MB peak (mmap-assisted) + ~15 MB for
        # inference activations at the reduced 256 px transform.
        # If available memory is below our safety threshold, return a graceful
        # 503 rather than letting the OS OOM-kill the container. Available
        # memory is computed from the container's cgroup limit (Docker/
        # Railway/Heroku/k8s all use cgroups) since /proc/meminfo reports the
        # *host's* memory, not the container's, and never reflects the
        # cgroup cap under any of these runtimes.
        _avail_mb: float | None = _cgroup_available_mb()
        if _avail_mb is None:
            _avail_mb = _host_available_mb()  # not containerized — fall back to host check

        if 0 < _avail_mb < 200:
            try: load_classifier()
            except Exception: pass
            try: load_anomaly_detector()
            except Exception: pass
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Not enough RAM for Mask R-CNN ({_avail_mb:.0f} MB available, "
                    "~200 MB needed). Free-tier limit reached. "
                    "Run locally with `docker compose up` for full segmentation."
                ),
            )
        # ────────────────────────────────────────────────────────────────────

        load_ok = False
        load_err: Exception | None = None
        try:
            load_segmentation()
            load_ok = True
        except FileNotFoundError as exc:
            load_err = exc
        except (MemoryError, RuntimeError) as exc:
            try: load_classifier()
            except Exception: pass
            try: load_anomaly_detector()
            except Exception: pass
            raise HTTPException(
                status_code=503,
                detail=(
                    "Segmentation requires ~200 MB RAM. Not enough memory available. "
                    "Run locally with `docker compose up` for full functionality."
                )
            ) from exc

        if not load_ok:
            # Restore lightweight models before returning error
            try: load_classifier()
            except Exception: pass
            try: load_anomaly_detector()
            except Exception: pass
            raise HTTPException(
                status_code=503,
                detail="Segmentation model checkpoint not found. Train the model first (see README)."
            )

    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
        pil = Image.open(tmp_path).convert("RGB")
        img_tensor = TF.to_tensor(pil).to(device)

        from src.models.segmentation import run_segmentation
        with torch.inference_mode():
            result = run_segmentation(segmentation_model, img_tensor, confidence_threshold)
        return SegmentationResponse(**result)
    except (MemoryError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Out of memory during inference. Upgrade to a plan with ≥1 GB RAM."
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
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
    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        from src.pointcloud import process_las_file
        result = process_las_file(tmp_path, max_points=settings.max_pointcloud_points)
        return PointCloudResponse(**result)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"LiDAR processing unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Point cloud processing failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# XAI — GradCAM explanation
# ---------------------------------------------------------------------------

@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)) -> ExplainResponse:
    """Run GradCAM on the ResNet-50 classifier and return a saliency overlay.

    Highlights which image regions most influenced the predicted land-cover class.
    Implements the XAI pipeline from notebooks/04_gradcam_xai.ipynb.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")

    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        # First run classification to get predicted class
        tensor = preprocess_image(tmp_path, settings.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = classifier(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()
        confidence, predicted_index = torch.max(probs, dim=0)
        predicted_class = class_names[int(predicted_index)]

        # Run GradCAM (requires gradients — handled inside gradcam_explain)
        from src.xai import gradcam_explain
        gradcam_b64 = gradcam_explain(
            model=classifier,
            image_path=tmp_path,
            image_size=settings.image_size,
            device=device,
            target_class_idx=int(predicted_index),
        )
        return ExplainResponse(
            predicted_class=predicted_class,
            confidence=round(float(confidence), 4),
            gradcam_b64=gradcam_b64,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GradCAM failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Spectral analysis — RGB-based vegetation / water / urban indices
# ---------------------------------------------------------------------------

@app.post("/spectral", response_model=SpectralResponse)
async def spectral_analysis(file: UploadFile = File(...)) -> SpectralResponse:
    """Compute VARI, ExWI and ExUI spectral indices from an RGB satellite image.

    Returns three colour-coded heatmaps (vegetation, water, urban) and scalar
    mean values — inspired by notebooks/06_multispectral_features.ipynb.
    """
    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        from src.spectral import compute_spectral_indices
        result = compute_spectral_indices(tmp_path, output_size=64)
        return SpectralResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Spectral analysis failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Poverty proxy — wealth index from land-cover probabilities
# ---------------------------------------------------------------------------

@app.post("/poverty-proxy", response_model=PovertyResponse)
async def poverty_proxy(file: UploadFile = File(...)) -> PovertyResponse:
    """Estimate an economic wealth index from satellite land-cover classification.

    Combines ResNet-50 land-cover probabilities with class-level wealth weights
    calibrated to the Jean et al. (2016) and Yeh et al. (2020) NTL-based
    poverty-estimation methodology — see notebooks/05_poverty_proxy_nightlights.ipynb.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")

    content = await _read_upload_capped(file)
    tmp_path = _save_upload(file, content)
    try:
        tensor = preprocess_image(tmp_path, settings.image_size).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = classifier(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu()

        class_probs = {name: round(float(probs[i]), 6) for i, name in enumerate(class_names)}

        from src.poverty import compute_poverty_proxy
        result = compute_poverty_proxy(class_probs)
        return PovertyResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Poverty proxy failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
