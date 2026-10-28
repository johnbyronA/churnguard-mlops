# ---------- build stage: wheels for the minimal serving runtime ----------
FROM python:3.13-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt

# ---------- runtime stage ----------
FROM python:3.13-slim
LABEL org.opencontainers.image.title="churnguard-api" \
      org.opencontainers.image.description="Real-time churn scoring API (FastAPI + LightGBM)" \
      org.opencontainers.image.source="https://github.com/johnbyronA/churnguard-mlops"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CHURNGUARD_ROOT=/app MODEL_DIR=/app/models/champion PREDICTIONS_LOG=/app/logs/predictions.jsonl

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 churnguard

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY params.yaml pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && mkdir -p logs && chown -R churnguard /app
COPY --chown=churnguard models/champion ./models/champion

USER churnguard
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "churnguard.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
