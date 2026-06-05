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
COPY model_config.json .

# Verify src/data module is present at build time
RUN python -c "import src.data.preprocessing; print('src.data OK')"

# ── Model checkpoint download ──────────────────────────────────────────────
# Reads model_config.json (auto-committed by train_on_colab.ipynb after each
# training run) to get Google Drive file IDs.
#
# Flow:
#   1. Colab trains models → saves to Google Drive → commits model_config.json
#      → pushes to GitHub → Railway auto-deploys → downloads fresh models here
#
# Fallback: if a Drive ID is empty, falls back to GitHub Releases v1.0.0
# ───────────────────────────────────────────────────────────────────────────
RUN python - <<'EOF'
import json, pathlib, requests, sys, time

pathlib.Path("checkpoints").mkdir(exist_ok=True)

with open("model_config.json") as f:
    config = json.load(f)

FILES = {
    "best_model.pt":        config.get("best_model", ""),
    "autoencoder_best.pt":  config.get("autoencoder", ""),
    "segmentation_best.pt": config.get("segmentation", ""),
}

GITHUB_RELEASE = "https://github.com/iamvisheshsrivastava/geospatial/releases/download/v1.0.0"
MIN_SIZE_BYTES = 1_000_000  # files must be at least 1 MB or something went wrong


def download_gdrive(file_id: str, dest: pathlib.Path) -> bool:
    """Try multiple gdown strategies for large files. Returns True on success."""
    import gdown

    # Strategy 1: download by ID (recommended for large files)
    try:
        out = gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
        if out and dest.exists() and dest.stat().st_size >= MIN_SIZE_BYTES:
            return True
        print(f"  Strategy 1 result too small or None, trying strategy 2...", flush=True)
    except Exception as e:
        print(f"  Strategy 1 failed: {e}", flush=True)

    # Strategy 2: use uc export URL
    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
        out = gdown.download(url, str(dest), quiet=False, fuzzy=True)
        if out and dest.exists() and dest.stat().st_size >= MIN_SIZE_BYTES:
            return True
        print(f"  Strategy 2 result too small or None, trying strategy 3...", flush=True)
    except Exception as e:
        print(f"  Strategy 2 failed: {e}", flush=True)

    # Strategy 3: requests with session cookie workaround
    try:
        session = requests.Session()
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        resp = session.get(url, stream=True, timeout=60)
        # Handle large-file confirmation token
        token = None
        for key, val in resp.cookies.items():
            if key.startswith("download_warning"):
                token = val
                break
        if token:
            url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
            resp = session.get(url, stream=True, timeout=300)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if dest.exists() and dest.stat().st_size >= MIN_SIZE_BYTES:
            return True
    except Exception as e:
        print(f"  Strategy 3 failed: {e}", flush=True)

    return False


def download_github(fname: str, dest: pathlib.Path) -> None:
    url = f"{GITHUB_RELEASE}/{fname}"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


for fname, file_id in FILES.items():
    dest = pathlib.Path("checkpoints") / fname
    if file_id:
        print(f"[Google Drive] Downloading {fname} (id={file_id}) ...", flush=True)
        success = download_gdrive(file_id, dest)
        if not success:
            print(f"  All Drive strategies failed — falling back to GitHub Release", flush=True)
            download_github(fname, dest)
    else:
        print(f"[GitHub Release] Downloading {fname} ...", flush=True)
        download_github(fname, dest)

    if not dest.exists() or dest.stat().st_size < MIN_SIZE_BYTES:
        print(f"ERROR: {fname} download failed or file too small", flush=True)
        sys.exit(1)

    mb = dest.stat().st_size / 1_048_576
    print(f"  OK  {fname}  {mb:.1f} MB", flush=True)

print("All checkpoints ready.", flush=True)
EOF

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
