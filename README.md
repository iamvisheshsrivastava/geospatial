# Geospatial ML Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ed?logo=docker)
![Heroku](https://img.shields.io/badge/Deployed-Heroku-430098?logo=heroku)
![CI](https://github.com/iamvisheshsrivastava/geospatial/actions/workflows/ci.yml/badge.svg)

**Live demo:** [geospatial-ml-374ed002e5df.herokuapp.com](https://geospatial-ml-374ed002e5df.herokuapp.com) · [API docs](https://geospatial-ml-374ed002e5df.herokuapp.com/docs)

PyTorch + FastAPI platform for analysing satellite imagery and LiDAR point clouds. Eight endpoints covering land-cover classification, anomaly detection, change detection, tree crown segmentation, LiDAR forest inventory, GradCAM explainability, spectral indices, and a poverty proxy estimator.

---

## What it does

| Endpoint | Task | Model |
|---|---|---|
| `POST /predict` | 10-class land-cover classification | ResNet-50 fine-tuned on EuroSAT |
| `POST /anomaly` | Unsupervised anomaly detection | Convolutional Autoencoder (IEEE IJCNN 2024) |
| `POST /change-detect` | Temporal change detection (before/after) | ResNet-50 feature-space comparison |
| `POST /segment` | Tree crown instance segmentation | Mask R-CNN (ResNet-50 + FPN, COCO pretrained) |
| `POST /pointcloud` | LiDAR forest inventory | CHM + ITS watershed segmentation |
| `POST /explain` | GradCAM saliency map on any prediction | ResNet-50 with gradient hooks |
| `POST /spectral` | VARI / ExWI / ExUI index heatmaps | Pixel-level RGB spectral arithmetic |
| `POST /poverty-proxy` | Wealth index estimate from land cover | Classifier probs × class-level NTL weights |

The classifier was trained on [EuroSAT](https://github.com/phelber/EuroSAT) (27,000 Sentinel-2 patches, 10 land-cover classes) and hit **val macro F1 = 0.9905, AUC-OVR = 0.9998** in 30 epochs.

---

## Research basis

The `/anomaly` endpoint implements the methodology from my published paper:

> **V. Srivastava**, "Autoencoder Optimization for Anomaly Detection," *IEEE IJCNN 2024*

Specific choices from the paper that are wired into the training code:
- BCE loss instead of MSE (better gradient signal on coloured imagery)
- [0, 1] normalisation, not ImageNet stats
- Three architectures trained (CAE-2Conv, CAE-3Conv, CAE-VariedFilter), winner picked by AUC-ROC
- H+V flip augmentation only — rotation avoided because orientation is semantically meaningful in satellite images
- Threshold = 95th percentile of normal-class reconstruction errors

The `/poverty-proxy` endpoint draws from the Jean et al. (2016, *Science*) two-stage transfer learning pipeline — land-cover probabilities from the ResNet-50 are combined with class-level nighttime-light weights to produce a coarse wealth index.

---

## Results

Trained on Kaggle (Tesla P100, ~55 minutes total):

| Stage | Val macro F1 | Val AUC OVR |
|---|---:|---:|
| Classifier (ResNet-50, 30 epochs) | **0.9905** | **0.9998** |

The autoencoder was trained with Forest as the normal class. An Industrial patch scores ~0.72 anomaly score vs ~0.69 for a healthy forest patch — enough separation to flag deforestation.

---

## Training

Training runs on Kaggle (free P100 GPU). The notebook `train_models.ipynb` handles everything — EuroSAT download, all three models, checkpoint upload to a Kaggle Dataset, and a CI trigger that redeploys to Heroku automatically.

To retrain:
1. Open `train_models.ipynb` in Kaggle, enable GPU
2. Run all cells (~55 min)
3. Step 8 uploads checkpoints → GitHub Actions picks them up → Heroku redeploys

Or run locally:

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download EuroSAT (~90 MB)
python scripts/download_eurosat.py --output-dir data/eurosat

# train classifier
python -m src.train \
  --data-root data/eurosat \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --num-workers 0 \
  --checkpoint-dir checkpoints \
  --wandb-mode disabled

# train anomaly detector
python -m src.anomaly \
  --data-root data/eurosat \
  --normal-classes Forest \
  --epochs 50 \
  --batch-size 64 \
  --image-size 224 \
  --patience 5 \
  --threshold-percentile 95 \
  --wandb-mode disabled
```

---

## Run locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

`http://localhost:8000` — browser UI  
`http://localhost:8000/docs` — Swagger

```bash
# classify a patch
curl -X POST http://localhost:8000/predict -F "file=@patch.jpg"

# anomaly check
curl -X POST http://localhost:8000/anomaly -F "file=@patch.jpg"

# change detection
curl -X POST http://localhost:8000/change-detect \
  -F "before=@t1.jpg" -F "after=@t2.jpg"

# tree crown segmentation
curl -X POST http://localhost:8000/segment -F "file=@aerial.jpg"

# LiDAR forest inventory
curl -X POST http://localhost:8000/pointcloud -F "file=@scan.las"

# GradCAM explanation
curl -X POST http://localhost:8000/explain -F "file=@patch.jpg"
```

Or Docker:

```bash
docker compose up --build
```

---

## Deployment

GitHub Actions (`ci.yml`) runs on every push to `main`:
1. Runs `pytest`
2. Downloads model checkpoints from the Kaggle Dataset `visheshsrivastava/geospatial-checkpoints`
3. Builds a Docker image with checkpoints baked in
4. Pushes to Heroku Container Registry and releases

No manual steps needed after pushing — just push and the live URL updates in ~8 minutes.

---

## Project layout

```
src/
  api/
    main.py             FastAPI app (8 endpoints + /health)
    static/             Browser UI — index.html, app.js, samples/
  data/
    preprocessing.py    rasterio GeoTIFF → normalised tensor
    dataset.py          EuroSAT loader, stratified split, augmentation
  models/
    resnet.py           ResNet-50 classifier
    autoencoder.py      CAE encoder + decoder
    segmentation.py     Mask R-CNN wrapper
  anomaly.py            Autoencoder train loop + inference (IEEE IJCNN 2024)
  change_detection.py   ResNet feature-space difference map
  pointcloud.py         CHM rasterisation, ITS watershed, per-tree metrics
  xai.py                GradCAM with gradient hooks
  spectral.py           VARI / ExWI / ExUI heatmaps
  poverty.py            NTL-weighted wealth index from class probabilities
  train.py              Classifier training loop
  evaluate.py           F1, AUC, confusion matrix, JSON report
  config.py             Pydantic settings (env vars / .env)
notebooks/
  04_gradcam_xai.ipynb                 GradCAM + SHAP explainability
  05_poverty_proxy_nightlights.ipynb   Nighttime lights wealth proxy
  06_multispectral_features.ipynb      13-band Sentinel-2 spectral indices
  07_africa_poverty_transfer_learning.ipynb  Jean et al. 2016 pipeline
train_models.ipynb                     Kaggle training notebook (all 3 models)
scripts/
  download_eurosat.py   EuroSAT dataset download
tests/                  pytest suite
```

---

## Environment variables

| Variable | Default |
|---|---|
| `PORT` | `8000` |
| `MODEL_PATH` | `checkpoints/best_model.pt` |
| `AUTOENCODER_PATH` | `checkpoints/autoencoder_best.pt` |
| `SEGMENTATION_PATH` | `checkpoints/segmentation_best.pt` |
| `DEVICE` | `cpu` |
| `S3_BUCKET` | unset |

---

## Key dependencies

`torch` · `torchvision` · `fastapi` · `rasterio` · `laspy` · `scikit-learn` · `wandb` · `pytorch-grad-cam` · `pillow`
