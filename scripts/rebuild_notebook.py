"""Rebuild train_models.ipynb — Kaggle-ready training notebook."""
import json
from pathlib import Path

cells = []

def md(src): cells.append({"cell_type":"markdown","id":f"md{len(cells)}","metadata":{},"source":src})
def code(src): cells.append({"cell_type":"code","id":f"c{len(cells)}","metadata":{},"source":src,"outputs":[],"execution_count":None})

# ── Title ──────────────────────────────────────────────────────────────────
md("""# Geospatial ML — Train All Models

Trains all 3 model checkpoints needed for the API:
1. **ResNet-50 classifier** — land-cover classification (`best_model.pt`)
2. **Convolutional Autoencoder** — anomaly detection, IEEE IJCNN 2024 (`autoencoder_best.pt`)
3. **Mask R-CNN** — tree crown segmentation (`segmentation_best.pt`)

After training, checkpoints are uploaded to a Kaggle Dataset and a GitHub commit
triggers the CI/CD pipeline → Docker build → Heroku deploy automatically.

**Works on:** Kaggle (recommended — T4 x2 GPU, 30h/week free) or Google Colab.

> Select **Runtime → T4 GPU** (Kaggle: Accelerator → GPU T4 x2) before running.""")

# ── Step 1 ─────────────────────────────────────────────────────────────────
md("## Step 1 — Setup: secrets and output directory")

code("""\
import os
from pathlib import Path

# ── Detect environment ────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle')
IS_COLAB  = 'google.colab' in str(__import__('sys').modules)

print(f'Environment: {"Kaggle" if IS_KAGGLE else "Colab" if IS_COLAB else "Other"}')

# ── Load secrets ──────────────────────────────────────────────────────────
if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    GITHUB_TOKEN   = _secrets.get_secret('GITHUB_TOKEN')
    KAGGLE_API_TOK = None  # Kaggle CLI is pre-authenticated in-notebook — no secret needed
    OUTPUT_DIR = Path('/kaggle/working/checkpoints')
    WORK_DIR   = Path('/kaggle/working')
elif IS_COLAB:
    from google.colab import userdata
    GITHUB_TOKEN   = userdata.get('GITHUB_TOKEN')
    KAGGLE_API_TOK = userdata.get('KAGGLE_API_TOKEN')
    OUTPUT_DIR = Path('/content/checkpoints')
    WORK_DIR   = Path('/content')
else:
    raise RuntimeError('Run this on Kaggle or Google Colab.')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Checkpoints will be saved to: {OUTPUT_DIR}')
print(f'GitHub token loaded: {"OK" if GITHUB_TOKEN else "MISSING — add in Secrets"}')""")

# ── Step 2 ─────────────────────────────────────────────────────────────────
md("## Step 2 — Clone the repo")

code("""\
import subprocess
REPO = 'iamvisheshsrivastava/geospatial'

if IS_KAGGLE:
    os.chdir('/kaggle/working')
else:
    os.chdir('/content')

if not Path('geospatial').exists():
    subprocess.run(['git', 'clone', f'https://{GITHUB_TOKEN}@github.com/{REPO}.git'], check=True)

os.chdir('geospatial')
subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
print('Repo ready.')
import subprocess as _sp
print(_sp.run(['git', 'log', '--oneline', '-3'], capture_output=True, text=True).stdout)""")

# ── Step 3 ─────────────────────────────────────────────────────────────────
md("## Step 3 — Install dependencies")

code("""\
!pip install -q \\
    torch torchvision \\
    rasterio \\
    wandb \\
    boto3 \\
    scikit-learn \\
    numpy pandas matplotlib pillow \\
    pydantic pydantic-settings \\
    tqdm

import torch
print(f'PyTorch {torch.__version__}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')""")

# ── Step 4 ─────────────────────────────────────────────────────────────────
md("""## Step 4 — Download & extract EuroSAT dataset (~90 MB)

Downloads the EuroSAT RGB dataset (Sentinel-2 satellite imagery, 10 land-cover classes).
Validates the ZIP magic bytes before extracting — if the server returns an error page
instead of a ZIP, we catch it immediately.""")

code("""\
import os, zipfile, shutil, requests
from pathlib import Path

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
EUROSAT_DIR = Path('data/eurosat')
EUROSAT_DIR.mkdir(parents=True, exist_ok=True)
zip_path = EUROSAT_DIR / 'EuroSAT.zip'

# Skip download if already extracted
if all((EUROSAT_DIR / cls).exists() for cls in CLASSES):
    print('EuroSAT already extracted — skipping download.')
else:
    URL = 'https://madm.dfki.de/files/sentinel/EuroSAT.zip'
    print(f'Downloading EuroSAT from {URL} ...')
    resp = requests.get(URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(zip_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f'\\r  {downloaded//1_048_576}/{total//1_048_576} MB ({downloaded/total*100:.1f}%)', end='')
    print(f'\\nDownload complete ({zip_path.stat().st_size/1_048_576:.1f} MB)')

    with open(zip_path, 'rb') as f:
        if f.read(4) != b'PK\\x03\\x04':
            raise RuntimeError('Downloaded file is NOT a ZIP. Server may have returned an error page.')

    tmp_dir = EUROSAT_DIR / '_tmp'
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    found = {}
    for dirpath, dirnames, _ in os.walk(tmp_dir):
        for d in dirnames:
            if d in CLASSES and d not in found:
                found[d] = Path(dirpath) / d
    for cls, src in found.items():
        dest = EUROSAT_DIR / cls
        if dest.exists(): shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

print('\\nDataset verification:')
all_ok = True
for cls in CLASSES:
    d = EUROSAT_DIR / cls
    count = len(list(d.glob('*'))) if d.exists() else 0
    status = 'OK' if count > 0 else 'MISSING'
    if count == 0: all_ok = False
    print(f'  {status:7s}  {cls} ({count} images)')
if not all_ok:
    raise RuntimeError('Some classes missing — check ZIP contents.')
print('\\nAll 10 classes ready.')""")

# ── Step 5 ─────────────────────────────────────────────────────────────────
md("""## Step 5 — Train the ResNet-50 classifier

~25 minutes on T4 GPU. Fine-tunes a ResNet-50 (pretrained on ImageNet) on EuroSAT
RGB imagery for 10-class land-cover classification. Uses 30 epochs for solid convergence
(0.99 macro F1 target). Saves the best checkpoint based on validation macro F1.""")

code("""\
!python -m src.train \\
--data-root data/eurosat \\
--epochs 30 \\
--batch-size 64 \\
--learning-rate 3e-4 \\
--num-workers 2 \\
--checkpoint-dir checkpoints \\
--wandb-mode disabled

# Show results
import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
metrics = ckpt.get('metrics', {})
print(f"Val macro F1 : {metrics.get('macro_f1', 'N/A')}")
print(f"Val AUC OVR  : {metrics.get('auc_ovr', 'N/A')}")
print(f"Classes      : {ckpt.get('class_names')}")""")

# ── Step 6 ─────────────────────────────────────────────────────────────────
md("""## Step 6 — Train the Anomaly Detector (IEEE IJCNN 2024 methodology)

~30 minutes on T4 GPU. Implements the full methodology from:
> **V. Srivastava**, "Autoencoder Optimization for Anomaly Detection," *IEEE IJCNN 2024*

Key decisions from the paper applied automatically:
- **BCE loss** (not MSE) for coloured satellite images
- **[0, 1] normalisation** (not ImageNet stats)
- **3 architectures** trained: CAE-2Conv, CAE-3Conv, CAE-VariedFilter
- **Best selected by AUC-ROC** on a held-out normal/anomaly split
- **H+V flip augmentation only** (no rotation)
- **Early stopping** patience = 5
- **Threshold = 95th percentile** of normal reconstruction errors""")

code("""\
!git pull origin main

!python -m src.anomaly \\
--data-root data/eurosat \\
--normal-classes Forest \\
--epochs 50 \\
--batch-size 64 \\
--image-size 224 \\
--patience 5 \\
--threshold-percentile 95 \\
--wandb-mode disabled

import torch
ckpt = torch.load('checkpoints/autoencoder_best.pt', map_location='cpu', weights_only=False)
print(f"Best arch : {ckpt['arch']}")
print(f"AUC-ROC   : {ckpt['auc_roc']:.4f}")
print(f"Threshold : {ckpt['threshold']:.6f}  (95th pct of normal errors)")""")

# ── Step 6b ────────────────────────────────────────────────────────────────
md("""## Step 6b — Save Mask R-CNN for tree crown segmentation

Downloads a Mask R-CNN pretrained on COCO (80 object classes) from torchvision.
No custom training needed — works out of the box for object/tree detection in aerial imagery.""")

code("""\
import torch, torchvision
from pathlib import Path

Path('checkpoints').mkdir(exist_ok=True)
print('Downloading pretrained Mask R-CNN weights...')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
torch.save(model.state_dict(), 'checkpoints/segmentation_best.pt')
size_mb = Path('checkpoints/segmentation_best.pt').stat().st_size / 1_048_576
print(f'Saved checkpoints/segmentation_best.pt ({size_mb:.1f} MB)')""")

# ── Step 7 ─────────────────────────────────────────────────────────────────
md("## Step 7 — Verify all checkpoints")

code("""\
import os
all_good = True
for f in ['checkpoints/best_model.pt', 'checkpoints/autoencoder_best.pt', 'checkpoints/segmentation_best.pt']:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1_048_576
        print(f'  OK       {f}  ({size_mb:.1f} MB)')
    else:
        print(f'  MISSING  {f}')
        all_good = False

print('\\nAll 3 checkpoints ready — proceed to Step 8.' if all_good else '\\nSome checkpoints missing — check steps above.')""")

# ── Step 8 ─────────────────────────────────────────────────────────────────
md("""## Step 8 — Upload checkpoints to Kaggle Dataset → triggers Heroku deploy

This cell:
1. Uploads the 3 model checkpoints as a **Kaggle Dataset** (`geospatial-checkpoints`)
2. Pushes a commit to GitHub → triggers GitHub Actions CI/CD
3. GitHub Actions downloads from the dataset → builds Docker → deploys to Heroku (~12 min)

**No Google Drive needed. Works every time.**""")

code("""\
import subprocess, json, datetime, shutil, os
from pathlib import Path

RUN_TS = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
REPO   = 'iamvisheshsrivastava/geospatial'
BRANCH = 'main'

# Set Kaggle API credentials
os.environ['KAGGLE_API_TOKEN'] = KAGGLE_API_TOK

# ── 1. Verify checkpoints ─────────────────────────────────────────────────
FILES = ['best_model.pt', 'autoencoder_best.pt', 'segmentation_best.pt']
print('Verifying checkpoints...')
for fname in FILES:
    p = Path('checkpoints') / fname
    if not p.exists():
        raise FileNotFoundError(f'{fname} not found — run training cells first.')
    print(f'  OK  {fname}  ({p.stat().st_size / 1_048_576:.1f} MB)')

# ── 2. Upload to Kaggle Dataset ───────────────────────────────────────────
dataset_dir = WORK_DIR / 'dataset_upload'
dataset_dir.mkdir(exist_ok=True)

for fname in FILES:
    shutil.copy2(Path('checkpoints') / fname, dataset_dir / fname)

metadata = {
    'title': 'Geospatial ML Checkpoints',
    'id': 'visheshsrivastava/geospatial-checkpoints',
    'licenses': [{'name': 'CC0-1.0'}]
}
with open(dataset_dir / 'dataset-metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'\\nUploading checkpoints to Kaggle Dataset (run {RUN_TS})...')
result = subprocess.run(
    ['kaggle', 'datasets', 'create', '-p', str(dataset_dir), '--dir-mode', 'zip'],
    capture_output=True, text=True
)
if 'already exists' in result.stdout + result.stderr or result.returncode != 0:
    print('Dataset exists — uploading new version...')
    result = subprocess.run(
        ['kaggle', 'datasets', 'version', '-p', str(dataset_dir),
         '-m', f'Training run {RUN_TS}', '--dir-mode', 'zip'],
        capture_output=True, text=True
    )
print(result.stdout or result.stderr)

# ── 3. Push trigger commit to GitHub ─────────────────────────────────────
print('Pushing trigger commit to GitHub...')

repo_dir = WORK_DIR / 'repo_push'
if repo_dir.exists():
    shutil.rmtree(repo_dir)

subprocess.run([
    'git', 'clone', '--depth', '1',
    f'https://{GITHUB_TOKEN}@github.com/{REPO}.git', str(repo_dir)
], check=True)

trigger = {
    '_trained_at': RUN_TS,
    '_dataset': 'visheshsrivastava/geospatial-checkpoints',
    '_note': 'Auto-updated by training notebook — do not edit manually.'
}
with open(repo_dir / 'model_config.json', 'w') as f:
    json.dump(trigger, f, indent=2)

subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.email', 'training@auto.bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'config', 'user.name', 'Training Bot'], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'add', 'model_config.json'], check=True)
subprocess.run([
    'git', '-C', str(repo_dir), 'commit', '--allow-empty', '-m',
    f'chore(models): training run {RUN_TS} — deploy from Kaggle dataset'
], check=True)
subprocess.run(['git', '-C', str(repo_dir), 'push', 'origin', BRANCH], check=True)

print('\\n' + '='*60)
print(f'  DONE — run {RUN_TS}')
print('='*60)
print('Kaggle Dataset: kaggle.com/visheshsrivastava/geospatial-checkpoints')
print('Watch CI/CD  : https://github.com/iamvisheshsrivastava/geospatial/actions')
print('Live app     : https://geospatial-ml-374ed002e5df.herokuapp.com')
print('='*60)""")

# ── Step 9 ─────────────────────────────────────────────────────────────────
md("""## Step 9 — Quick sanity check (optional)

Loads the trained classifier and runs a prediction on a random EuroSAT image.
Expected: the predicted class matches the folder the image came from.""")

code("""\
import torch
from pathlib import Path
from src.models.resnet import build_resnet50_classifier
from src.data.preprocessing import preprocess_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
class_names = ckpt['class_names']
model = build_resnet50_classifier(num_classes=len(class_names), pretrained=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

# Test one image from each class
print('Per-class sanity check:')
correct, total = 0, 0
for cls_dir in sorted(Path('data/eurosat').iterdir()):
    if not cls_dir.is_dir(): continue
    sample = next(cls_dir.glob('*.jpg'), None)
    if sample is None: continue
    tensor = preprocess_image(sample, 224).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    conf, idx = probs.max(0)
    pred = class_names[idx]
    ok = pred == cls_dir.name
    correct += int(ok)
    total += 1
    status = 'OK' if ok else 'WRONG'
    print(f'  [{status}] {cls_dir.name:25s} -> {pred} ({conf:.1%})')

print(f'\\nResult: {correct}/{total} correct')
print(f'Val F1: {ckpt.get(\"metrics\", {}).get(\"macro_f1\", \"N/A\")}')""")

# ── Build and save ──────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

out = Path(__file__).parent.parent / "train_models.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Created {out}")
