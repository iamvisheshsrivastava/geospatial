FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget \
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

# Download model checkpoints at BUILD TIME so they are baked into the image.
# This is more reliable than downloading at container startup — no runtime
# network issues, no shell script problems, no startup timeouts.
RUN RELEASE="https://github.com/iamvisheshsrivastava/geospatial/releases/download/v1.0.0" \
    && mkdir -p checkpoints \
    && echo "Downloading best_model.pt ..." \
    && wget -q --show-progress -O checkpoints/best_model.pt         "$RELEASE/best_model.pt" \
    && echo "Downloading autoencoder_best.pt ..." \
    && wget -q --show-progress -O checkpoints/autoencoder_best.pt   "$RELEASE/autoencoder_best.pt" \
    && echo "Downloading segmentation_best.pt ..." \
    && wget -q --show-progress -O checkpoints/segmentation_best.pt  "$RELEASE/segmentation_best.pt" \
    && echo "All checkpoints downloaded." \
    && ls -lh checkpoints/

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
