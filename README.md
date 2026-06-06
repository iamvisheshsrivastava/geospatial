# Satellite & LiDAR Geospatial ML Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ed?logo=docker)
![Heroku](https://img.shields.io/badge/Deployed-Heroku-430098?logo=heroku)
![CI](https://github.com/iamvisheshsrivastava/geospatial/actions/workflows/ci.yml/badge.svg)

> **Live demo:** [geospatial-ml-374ed002e5df.herokuapp.com](https://geospatial-ml-374ed002e5df.herokuapp.com) · [Interactive API docs](https://geospatial-ml-374ed002e5df.herokuapp.com/docs)

A production-ready Python platform for geospatial machine learning on **Sentinel-2 satellite imagery and LiDAR point clouds** — covering the full workflow from raw data through model training, explainability analysis, and cloud deployment.

---

## Research Foundation

### Anomaly Detection — IEEE IJCNN 2024

The unsupervised anomaly-detection module (`POST /anomaly`) is a direct application of published research:

> **V. Srivastava**, "Autoencoder Optimization for Anomaly Detection," *IEEE International Joint Conference on Neural Networks (IJCNN)*, 2024.

The implementation follows every methodological decision from the paper:
- **Binary Cross-Entropy loss** over MSE for feature-rich coloured satellite images
- **[0, 1] normalisation** (no ImageNet mean/std) to preserve radiometric content
- **Three convolutional autoencoder architectures** (CAE-2Conv, CAE-3Conv, CAE-VariedFilter) trained in parallel; winner selected by **AUC-ROC** on a held-out normal/anomaly split
- **Flip-only data augmentation** (horizontal + vertical) — rotation is avoided because orientation carries semantic meaning in satellite imagery
- **Early stopping** with patience = 5 on validation BCE
- **Anomaly threshold = 95th percentile** of normal-class reconstruction errors, computed automatically during training and stored in the checkpoint

---

## Connection to Poverty Estimation and SDG Research

This platform directly implements the three pillars of satellite-based poverty estimation research, aligned with the methodology of the [AI and Global Development Lab](https://www.aidevlab.org) at Chalmers University:

### Pillar 1 — Deep learning on Sentinel-2 for poverty estimation (SDG-1)

The EuroSAT dataset used throughout this repo is derived from **Sentinel-2 satellite imagery** — the same sensor used in leading poverty-estimation pipelines. Notebook 07 implements the full **Jean et al. (2016, *Science*)** two-stage transfer learning pipeline:

1. Train a CNN on daytime Sentinel-2 imagery to predict **VIIRS nighttime light intensity** (free global proxy for ground-truth wealth)
2. Extract CNN penultimate-layer features → **Ridge regression** → DHS survey wealth index

This produces poverty maps at continental scale, including locations with **no ground-truth survey data** — the core technical contribution of the Jean et al. and Yeh et al. (2020, *Nature Communications*) pipelines.

### Pillar 2 — Multi-sensor comparison readiness

The preprocessing pipeline (`src/data/preprocessing.py`) uses `rasterio` for resolution-agnostic raster loading, making it straightforward to swap imagery sources:

| Sensor | Resolution | Access |
|---|---|---|
| Pléiades | 2 m | Commercial |
| **Sentinel-2** | **10 m** | **Free — used here** |
| Landsat | 30 m | Free |

Comparing poverty-estimate quality and computational cost across these sensors — quantifying the precision/cost trade-off — is a direct extension of this codebase.

### Pillar 3 — XAI for policy-relevant model interpretation (SDG-16)

The `/explain` API endpoint and [Notebook 04](notebooks/04_gradcam_xai.ipynb) implement **GradCAM** saliency maps on the ResNet-50 classifier. Applied to a poverty-estimation CNN, these maps answer the policy-critical question:

> *Does the model attend to controllable features (roof materials, road access) or structural geography (distance to cities)?*

This distinction determines whether model predictions can inform **targeted interventions** — a requirement for responsible deployment in SDG policy contexts.

**Relevant references:** Jean et al. (2016, *Science*) · Yeh et al. (2020, *Nature Comms*) · Henderson et al. (2012, *AER*) · Engstrom et al. (2017, *World Bank*)

---

## Five API Capabilities

| Endpoint | Task | Model |
|---|---|---|
| `POST /predict` | Land-cover classification | ResNet-50 fine-tuned on EuroSAT |
| `POST /anomaly` | Unsupervised anomaly detection | Convolutional Autoencoder |
| `POST /change-detect` | Temporal change detection | ResNet-50 feature-space comparison |
| `POST /segment` | Tree crown instance segmentation | Mask R-CNN (ResNet-50 + FPN) |
| `POST /pointcloud` | LiDAR forest inventory | CHM + ITS watershed segmentation |

---

## Notebooks

| Notebook | Description |
|---|---|
| [`train_on_colab.ipynb`](train_on_colab.ipynb) | End-to-end training on Google Colab free GPU — downloads EuroSAT, trains all models, saves to Drive |
| [`notebooks/03_lidar_open3d_forest.ipynb`](notebooks/03_lidar_open3d_forest.ipynb) | **LiDAR forest inventory with Open3D** — load LAS/LAZ, height normalisation, CHM rasterisation, individual tree segmentation (ITS), per-tree metrics; interactive 3D visualisation via Open3D; production AWS/Airflow pipeline sketch for 44moles-style forestry applications |
| [`notebooks/04_gradcam_xai.ipynb`](notebooks/04_gradcam_xai.ipynb) | GradCAM + SHAP explainability on ResNet-50 — spatial heatmaps showing which pixels drive land-cover predictions; poverty-estimation interpretation |
| [`notebooks/05_poverty_proxy_nightlights.ipynb`](notebooks/05_poverty_proxy_nightlights.ipynb) | VIIRS nighttime lights as poverty proxy — wealth index combining NTL intensity + urban LC fraction; connection to Jean et al. 2016 pipeline |
| [`notebooks/06_multispectral_features.ipynb`](notebooks/06_multispectral_features.ipynb) | 13-band Sentinel-2 feature extraction — NDVI, NDBI, NDWI computation; spectral profile visualisation; logistic regression AUC vs RGB baseline |
| [`notebooks/07_africa_poverty_transfer_learning.ipynb`](notebooks/07_africa_poverty_transfer_learning.ipynb) | **Jean et al. (2016) pipeline implemented** — two-stage transfer learning: CNN→NTL prediction, Ridge regression→DHS wealth index, dense poverty map over Nigeria; connects to PhD Objectives 1–3 |

---

## Architecture

```
satellite image / LiDAR point cloud
        │
        ▼
  rasterio / laspy          ← geospatial I/O (GeoTIFF, LAS/LAZ)
        │
        ▼
  PyTorch models            ← ResNet-50 · Autoencoder · Mask R-CNN
        │
        ▼
  GradCAM / SHAP            ← explainability layer (XAI)
        │
        ▼
  FastAPI (Docker)          ← inference API + browser UI
        │
        ▼
  Heroku (cloud)            ← containerised deployment, live public URL
        │
  Weights & Biases          ← experiment tracking, metrics, confusion matrix
```

---

## Results Benchmark (EuroSAT RGB)

| Config | Backbone | Image size | Epochs | Val macro F1 | Val AUC OVR |
|---|---|---:|---:|---:|---:|
| Frozen backbone | ResNet-50 ImageNet | 224 | 5 | 0.91 | 0.98 |
| Full fine-tune | ResNet-50 ImageNet | 224 | 10 | **0.99** | **0.99** |

---

## Train on Google Colab (recommended — free GPU)

If you don't have a local GPU, use the provided notebook to train on a free T4 GPU in the cloud:

1. Open [train_on_colab.ipynb](train_on_colab.ipynb) in [Google Colab](https://colab.research.google.com)
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells (~25 minutes)

The notebook downloads EuroSAT, trains all 3 models, and saves checkpoints to your Google Drive.

---

## Quick Start (local)

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
  --wandb-mode disabled
```

### 3 — Train the anomaly detector

```bash
# Trains all 3 CAE architectures (paper methodology), picks best by AUC-ROC
python -m src.anomaly \
  --data-root data/eurosat \
  --normal-classes Forest \
  --epochs 50 \
  --batch-size 256 \
  --image-size 64 \
  --patience 5 \
  --threshold-percentile 95 \
  --wandb-mode disabled
```

### 4 — Evaluate

```bash
python -m src.evaluate \
  --data-root data/eurosat \
  --checkpoint checkpoints/best_model.pt \
  --output-dir reports
```

### 5 — Run the API locally

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
| `PORT` | Port the server listens on | `8000` |
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

## Deployment (Heroku)

The live API is deployed on [Heroku](https://heroku.com) using Docker (Standard-2X dyno, 1 GB RAM).
Model checkpoints are downloaded automatically from Google Drive at container build time via
`model_config.json` — updated automatically after each Colab training run, no manual upload needed.

```bash
# CI/CD is fully automated via GitHub Actions
# Just push to main — tests run, then Docker image builds and deploys to Heroku
git push origin main
```

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
notebooks/
  04_gradcam_xai.ipynb           GradCAM + SHAP explainability
  05_poverty_proxy_nightlights.ipynb  Nighttime lights wealth index proxy
  06_multispectral_features.ipynb     13-band Sentinel-2 spectral indices
scripts/
  download_eurosat.py    One-command EuroSAT dataset download
figures/                 Output figures from notebooks
deploy/
  task-definition.json   ECS Fargate task definition (optional AWS path)
  deploy.sh              ECR + ECS deployment script (optional AWS path)
tests/                   pytest suite (CI via GitHub Actions)
```

---

## CI

GitHub Actions runs `pytest` on every push (`.github/workflows/ci.yml`).

---

## Key libraries

`torch` · `torchvision` · `rasterio` · `laspy` · `open3d` · `fastapi` · `boto3` · `wandb` · `scikit-learn` · `pytorch-grad-cam` · `shap` · `geopandas`
