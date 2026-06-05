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
COPY entrypoint.sh .
COPY README.md .
COPY model_config.json .

# Model checkpoints are downloaded by GitHub Actions before docker build
# and placed in checkpoints/ — they are NOT committed to git.
# See .github/workflows/ci.yml → "Download model checkpoints" step.
COPY checkpoints/ ./checkpoints/

RUN python -c "import src.data.preprocessing; print('src OK')"
RUN chmod +x entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
