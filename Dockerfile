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

# cache-bust: 2026-05-29-v2
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/
COPY README.md .

# Verify src/data module is present
RUN python -c "import src.data.preprocessing; print('src.data.preprocessing OK')" \
 && python -c "import src.data.dataset; print('src.data.dataset OK')"

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
