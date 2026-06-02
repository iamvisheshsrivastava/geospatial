# Satellite & LiDAR Geospatial ML Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ed?logo=docker)
![Railway](https://img.shields.io/badge/Deployed-Railway-6441a5)
![CI](https://github.com/iamvisheshsrivastava/geospatial/actions/workflows/ci.yml/badge.svg)

> **Live demo:** [api-production-3378.up.railway.app](https://api-production-3378.up.railway.app) · [Interactive API docs](https://api-production-3378.up.railway.app/docs)

A production-ready Python platform for geospatial machine learning on **Sentinel-2 satellite imagery and LiDAR point clouds** — covering the full workflow from raw data through model training, explainability analysis, and cloud deployment.

---

## Research Relevance

This platform is built on **Sentinel-2 imagery** (via the EuroSAT benchmark), the same satellite data used in leading poverty estimation research:

- **Jean et al. (2016, *Science*)** demonstrated that CNNs trained on daytime satellite imagery can predict consumption poverty across African villages with r² ≈ 0.64 — outperforming traditional survey extrapolation.
- **Yeh et al. (2020, *Nature Communications*)** extended this to Sentinel-2, achieving state-of-the-art poverty maps at 2.4 km resolution across Africa.

Beyond classification, this repo includes **XAI (GradCAM + SHAP)** notebooks that reveal *which visual features* drive model decisions — a critical requirement for deploying satellite-based socioeconomic models in policy contexts. Identifying whether a poverty model attends to roof materials, road density, or vegetation patterns directly informs its trustworthiness and fairness.

**Earth observation for socioeconomic analysis** is one of the most impactful applications of geospatial ML, directly advancing UN Sustainable Development Goals SDG-1 (No Poverty) and SDG-11 (Sustainable Cities).

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
| [`notebooks/04_gradcam_xai.ipynb`](notebooks/04_gradcam_xai.ipynb) | GradCAM + SHAP explainability on ResNet-50 — spatial heatmaps showing which pixels drive land-cover predictions; poverty-estimation interpretation |
| [`notebooks/05_poverty_proxy_nightlights.ipynb`](notebooks/05_poverty_proxy_nightlights.ipynb) | VIIRS nighttime lights as poverty proxy — wealth index combining NTL intensity + urban LC fraction; connection to Jean et al. 2016 pipeline |
| [`notebooks/06_multispectral_features.ipynb`](notebooks/06_multispectral_features.ipynb) | 13-band Sentinel-2 feature extraction — NDVI, NDBI, NDWI computation; spectral profile visualisation; logistic regression AUC vs RGB baseline |

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
  Railway (cloud)           ← containerised deployment, live public URL
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

## Deployment (Railway)

The live API is deployed on [Railway](https://railway.app) using Docker. Model checkpoints are
downloaded automatically from [GitHub Releases v1.0.0](https://github.com/iamvisheshsrivastava/geospatial/releases/tag/v1.0.0)
at container startup — no S3 or external storage required.

```bash
# railway.toml already configured — just push to main
git push origin main
# Railway auto-deploys on every push
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
