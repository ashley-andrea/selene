# ──────────────────────────────────────────────────────────────────────────────
#  Selene ML APIs — Hugging Face Spaces Dockerfile
#
#  Serves BOTH the cluster (GMM) and simulator (HistGBM) models in a single
#  FastAPI process on port 7860 (required by HF Spaces).
#
#  Build & push via push_to_hf.sh — do NOT build from repo root directly.
#  The Dockerfile expects to run with build context = this hf_space/ directory.
#
#  Local test:
#    docker build -t selene-ml-apis .
#    docker run -p 7860:7860 selene-ml-apis
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Model artifacts ───────────────────────────────────────────────────────────
COPY artifacts/ ./artifacts/

# ── Pill reference database ───────────────────────────────────────────────────
COPY drugs/ ./drugs/

# ── Inference server ──────────────────────────────────────────────────────────
COPY serve.py ./serve.py

# ── HF Spaces runs as a non-root user by default — ensure writable cache ──────
RUN chmod -R 777 /app

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

ENV PORT=7860

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]
