# ──────────────────────────────────────────────────────────────────────────────
#  Simulator Model Inference Server — Dockerfile
#
#  Context: build from the REPO ROOT (2-and-1-2-hackers/), not from this dir.
#
#  docker build -f models/simulation/Dockerfile -t simulator-model:latest .
#  docker run -p 8001:8001 simulator-model:latest
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY models/simulation/requirements-serve.txt ./requirements-serve.txt
RUN pip install --no-cache-dir -r requirements-serve.txt

# ── Model artifacts ───────────────────────────────────────────────────────────
# Only copy what the inference server actually needs
COPY models/simulation/artifacts/model_symptoms.pkl      ./artifacts/model_symptoms.pkl
COPY models/simulation/artifacts/model_satisfaction.pkl  ./artifacts/model_satisfaction.pkl
COPY models/simulation/artifacts/feature_meta.json       ./artifacts/feature_meta.json

# ── Pill reference database (used to build the pill features lookup) ──────────
COPY drugs/output/pill_reference_db.csv ./drugs/output/pill_reference_db.csv

# ── Inference server ──────────────────────────────────────────────────────────
COPY models/simulation/serve.py ./serve.py

# ── Non-root user (OpenShift requires this — arbitrary UID support) ───────────
RUN chown -R 1001:0 /app && chmod -R g=u /app
USER 1001

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8001

ENV PORT=8001
CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT} --workers 2 --log-level info"]
