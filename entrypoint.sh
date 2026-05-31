#!/bin/sh
# Download model checkpoints from GitHub Releases if not already present.
# This runs once on every container start. If the files are already there
# (e.g. mounted volume or cached layer), the download is skipped.

RELEASE_BASE="https://github.com/iamvisheshsrivastava/geospatial/releases/download/v1.0.0"
CHECKPOINT_DIR="checkpoints"

mkdir -p "$CHECKPOINT_DIR"

download_if_missing() {
    local filename="$1"
    local dest="$CHECKPOINT_DIR/$filename"
    if [ ! -f "$dest" ]; then
        echo "Downloading $filename ..."
        wget -q --show-progress -O "$dest" "$RELEASE_BASE/$filename" \
            || { echo "ERROR: failed to download $filename"; exit 1; }
        echo "$filename ready."
    else
        echo "$filename already present, skipping download."
    fi
}

download_if_missing "best_model.pt"
download_if_missing "autoencoder_best.pt"
download_if_missing "segmentation_best.pt"

echo "All checkpoints ready. Starting API..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
