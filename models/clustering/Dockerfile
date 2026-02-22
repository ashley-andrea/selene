# ──────────────────────────────────────────────────────────────────────────────
#  Cluster Model Inference Server — Dockerfile
#
#  Context: build from the REPO ROOT (2-and-1-2-hackers/), not from this dir.
#
#  docker build -f models/clustering/Dockerfile -t cluster-model:latest .
#  docker run -p 8000:8000 cluster-model:latest
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY models/clustering/requirements-serve.txt ./requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

# ── Model artifacts ───────────────────────────────────────────────────────────
# Only copy what the inference server actually needs
COPY models/clustering/artifacts/gmm_model.pkl      ./artifacts/gmm_model.pkl
COPY models/clustering/artifacts/scaler.pkl         ./artifacts/scaler.pkl
COPY models/clustering/artifacts/imputer.pkl        ./artifacts/imputer.pkl
COPY models/clustering/artifacts/profile_rules.json ./artifacts/profile_rules.json

# ── Inference server ──────────────────────────────────────────────────────────
COPY models/clustering/serve.py ./serve.py

# ── Non-root user (OpenShift requires this — arbitrary UID support) ───────────
RUN chown -R 1001:0 /app && chmod -R g=u /app
USER 1001

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

ENV PORT=8000
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info"]
