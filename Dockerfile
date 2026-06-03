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

# Download model checkpoints at BUILD TIME using Python requests (already
# installed above). More reliable than wget in Docker — handles redirects,
# streams large files, and fails the build loudly if a download fails.
RUN python - <<'EOF'
import requests, pathlib, sys

RELEASE = "https://github.com/iamvisheshsrivastava/geospatial/releases/download/v1.0.0"
FILES   = ["best_model.pt", "autoencoder_best.pt", "segmentation_best.pt"]

pathlib.Path("checkpoints").mkdir(exist_ok=True)
for fname in FILES:
    url  = f"{RELEASE}/{fname}"
    dest = pathlib.Path("checkpoints") / fname
    print(f"Downloading {fname} ...", flush=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    mb = dest.stat().st_size / 1_048_576
    print(f"  {fname}: {mb:.1f} MB", flush=True)

print("All checkpoints downloaded.", flush=True)
EOF

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
