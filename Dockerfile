FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 wget dos2unix \
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

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
