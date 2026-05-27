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
from src.storage.s3 import download_file_from_s3


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    model_path: str
    classes: list[str]


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Satellite Deforestation Classifier",
    description="Classifies uploaded satellite imagery into EuroSAT land-cover classes.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model: torch.nn.Module | None = None
class_names: list[str] = []
device = torch.device(settings.device)


def ensure_checkpoint_available() -> Path:
    if settings.model_path.exists():
        return settings.model_path

    if settings.s3_bucket and settings.s3_model_key:
        return download_file_from_s3(
            settings.s3_bucket,
            settings.s3_model_key,
            settings.model_path,
            settings.aws_region,
        )

    raise FileNotFoundError(
        f"Model checkpoint not found at {settings.model_path}. "
        "Set MODEL_PATH or provide S3_BUCKET and S3_MODEL_KEY."
    )


def load_model() -> None:
    global model, class_names

    checkpoint_path = ensure_checkpoint_available()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    loaded_model = build_resnet50_classifier(num_classes=len(class_names), pretrained=False)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()
    model = loaded_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        load_model()
    except FileNotFoundError:
        # The container can still start for health checks before a model is mounted.
        pass
    yield


app.router.lifespan_context = lifespan


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        model_loaded=model is not None,
        model_path=str(settings.model_path),
        classes=class_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded.")

    suffix = Path(file.filename or "upload.tif").suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        tensor = preprocess_image(tmp_path, image_size=settings.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        confidence, predicted_index = torch.max(probs, dim=0)
        probabilities = {name: float(probs[idx]) for idx, name in enumerate(class_names)}
        return PredictionResponse(
            predicted_class=class_names[int(predicted_index)],
            confidence=float(confidence),
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
