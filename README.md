# Satellite & LiDAR Geospatial ML Platform

A production-ready Python platform for geospatial machine learning, covering the full workflow from raw satellite and LiDAR data through training, evaluation, and cloud deployment.

**Five capabilities in one API:**

| Endpoint | Task | Model |
|---|---|---|
| `POST /predict` | Land-cover classification | ResNet-50 fine-tuned on EuroSAT |
| `POST /anomaly` | Unsupervised anomaly detection | Convolutional Autoencoder |
| `POST /change-detect` | Temporal change detection | ResNet-50 feature-space comparison |
| `POST /segment` | Tree crown instance segmentation | Mask R-CNN (ResNet-50 + FPN) |
| `POST /pointcloud` | LiDAR forest inventory | CHM + ITS watershed segmentation |

---

## Architecture

```
satellite image / LiDAR point cloud
        │
        ▼
  rasterio / laspy          ← geospatial I/O
        │
        ▼
  PyTorch models            ← ResNet-50 · Autoencoder · Mask R-CNN
        │
        ▼
  FastAPI (Docker)          ← inference API, /docs UI
        │
   ┌────┴────┐
   │   AWS   │            ← S3 checkpoints · ECS Fargate · CloudWatch
   └─────────┘
        │
  Weights & Biases         ← experiment tracking, metrics, confusion matrix
```

---

## Results Benchmark (EuroSAT RGB)

| Config | Backbone | Image size | Epochs | Val macro F1 | Val AUC OVR |
|---|---|---:|---:|---:|---:|
| Frozen backbone | ResNet-50 ImageNet | 224 | 5 | 0.91 | 0.98 |
| Full fine-tune | ResNet-50 ImageNet | 224 | 10 | **0.96** | **0.99** |

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1 — Download EuroSAT dataset

```bash
python scripts/download_eurosat.py --output-dir data/eurosat
```

### 2 — Train the classifier

```bash
python -m src.train \
  --data-root data/eurosat \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --wandb-project satellite-geospatial \
  --wandb-mode disabled          # or: online (needs wandb login)
```

### 3 — Train the anomaly detector

```bash
# Train on Forest class only (normal = healthy forest patches)
python -m src.anomaly \
  --data-root data/eurosat \
  --normal-classes Forest \
  --epochs 30 \
  --wandb-mode disabled
```

### 4 — Evaluate

```bash
python -m src.evaluate \
  --data-root data/eurosat \
  --checkpoint checkpoints/best_model.pt \
  --output-dir reports
```

### 5 — Run the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the UI, `http://localhost:8000/docs` for the full API.

```bash
# Classification
curl -X POST http://localhost:8000/predict -F "file=@sample.jpg"

# Anomaly detection
curl -X POST http://localhost:8000/anomaly -F "file=@sample.jpg"

# Change detection (two images, same area, different times)
curl -X POST http://localhost:8000/change-detect -F "before=@t1.jpg" -F "after=@t2.jpg"

# Tree crown segmentation
curl -X POST http://localhost:8000/segment -F "file=@aerial.jpg"

# LiDAR forest inventory
curl -X POST http://localhost:8000/pointcloud -F "file=@scan.las"
```

---

## Docker

```bash
docker compose up --build
```

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_PATH` | Classifier checkpoint | `checkpoints/best_model.pt` |
| `AUTOENCODER_PATH` | Anomaly detector checkpoint | `checkpoints/autoencoder_best.pt` |
| `SEGMENTATION_PATH` | Mask R-CNN checkpoint | `checkpoints/segmentation_best.pt` |
| `S3_BUCKET` | S3 bucket for model artifacts | unset |
| `S3_MODEL_KEY` | S3 key for classifier | unset |
| `S3_AUTOENCODER_KEY` | S3 key for autoencoder | unset |
| `S3_SEGMENTATION_KEY` | S3 key for segmentation model | unset |
| `AWS_REGION` | AWS region | `eu-central-1` |
| `DEVICE` | `cpu` or `cuda` | `cpu` |

---

## AWS Deployment (ECS Fargate)

```bash
export AWS_ACCOUNT_ID=123456789012
export AWS_REGION=eu-central-1
bash deploy/deploy.sh
```

The script builds and pushes the Docker image to ECR, registers a new ECS task definition,
and triggers a rolling deployment. Model checkpoints are read from S3 at container startup.

---

## Project Structure

```
src/
  api/
    main.py              FastAPI app — 5 endpoints + health
    static/              Browser UI (HTML/CSS/JS)
  data/
    preprocessing.py     rasterio GeoTIFF → normalised tensor
    dataset.py           EuroSAT discovery, stratified splits, augmentation
  models/
    resnet.py            ResNet-50 classifier
    autoencoder.py       Convolutional autoencoder (encoder + decoder)
    segmentation.py      Mask R-CNN tree crown instance segmentation
  anomaly.py             Unsupervised anomaly detection — train + infer
  change_detection.py    Feature-space temporal change detection
  pointcloud.py          LiDAR CHM, individual tree segmentation (ITS)
  train.py               Classifier training loop (W&B + S3)
  evaluate.py            F1, AUC, confusion matrix, JSON report
  config.py              Pydantic settings (env vars / .env)
  metrics.py             sklearn metric helpers
  storage/s3.py          boto3 upload / download helpers
scripts/
  download_eurosat.py    One-command EuroSAT dataset download
deploy/
  task-definition.json   ECS Fargate task definition
  deploy.sh              ECR + ECS deployment script
tests/                   pytest suite (CI via GitHub Actions)
```

---

## CI

GitHub Actions runs `pytest` on every push (`.github/workflows/ci.yml`).

---

## Key libraries

`torch` · `torchvision` · `rasterio` · `laspy` · `open3d` · `fastapi` · `boto3` · `wandb` · `scikit-learn`
