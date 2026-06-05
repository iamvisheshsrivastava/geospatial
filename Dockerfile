FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/
COPY entrypoint.sh .
COPY README.md .

# Verify src/data module is present at build time
RUN python -c "import src.data.preprocessing; print('src.data OK')"

# ── Model checkpoint download ──────────────────────────────────────────────
# Priority 1: Google Drive (set GDRIVE_*_ID build args in Railway dashboard
#             after each Colab training run — no manual upload needed).
# Priority 2: GitHub Releases v1.0.0 (fallback if Drive IDs are not set).
#
# To use Google Drive:
#   In Railway → your service → Settings → Build → Add build variable:
#     GDRIVE_CLASSIFIER_ID   = <file-id from Google Drive share link>
#     GDRIVE_AUTOENCODER_ID  = <file-id from Google Drive share link>
#     GDRIVE_SEGMENTATION_ID = <file-id from Google Drive share link>
#   Then trigger a redeploy — no git push required.
# ───────────────────────────────────────────────────────────────────────────
ARG GDRIVE_CLASSIFIER_ID=""
ARG GDRIVE_AUTOENCODER_ID=""
ARG GDRIVE_SEGMENTATION_ID=""

RUN python - <<'EOF'
import os, pathlib, sys, requests

pathlib.Path("checkpoints").mkdir(exist_ok=True)

GDRIVE_IDS = {
    "best_model.pt":         os.getenv("GDRIVE_CLASSIFIER_ID", ""),
    "autoencoder_best.pt":   os.getenv("GDRIVE_AUTOENCODER_ID", ""),
    "segmentation_best.pt":  os.getenv("GDRIVE_SEGMENTATION_ID", ""),
}
GITHUB_RELEASE = "https://github.com/iamvisheshsrivastava/geospatial/releases/download/v1.0.0"

def download_gdrive(file_id: str, dest: pathlib.Path) -> None:
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False, fuzzy=True)

def download_github(fname: str, dest: pathlib.Path) -> None:
    url = f"{GITHUB_RELEASE}/{fname}"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)

for fname, gdrive_id in GDRIVE_IDS.items():
    dest = pathlib.Path("checkpoints") / fname
    if gdrive_id:
        print(f"[Google Drive] Downloading {fname} (id={gdrive_id}) ...", flush=True)
        download_gdrive(gdrive_id, dest)
    else:
        print(f"[GitHub Release] Downloading {fname} ...", flush=True)
        download_github(fname, dest)
    mb = dest.stat().st_size / 1_048_576
    print(f"  ✓ {fname}: {mb:.1f} MB", flush=True)

print("All checkpoints ready.", flush=True)
EOF

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
