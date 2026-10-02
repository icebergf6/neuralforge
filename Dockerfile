FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ cmake make curl wget \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ───────────────────────────────────────────────────────────────
COPY . .

# ── Create models directory ────────────────────────────────────────────────────
RUN mkdir -p models

# ── Download model at build time (optional — can also mount a volume) ──────────
# Uncomment to bake model into image (~60MB):
# RUN python tools/download_model.py

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8088/v1/models || exit 1

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8088

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["python", "tools/serve.py"]
