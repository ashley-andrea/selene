"""
Lightweight mock server mimicking the Cluster Model and Simulator Model APIs.

Runs locally on port 8001. The agent hits this server via environment variables:
    CLUSTER_API_URL=http://localhost:8001/cluster/predict
    SIMULATOR_API_URL=http://localhost:8001/simulator/simulate

This server is a development and testing tool ONLY.
It is never deployed and must not be included in the production build.

REMOVE WHEN: Real models are deployed on Red Hat OpenShift.
"""

import random

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BC Agent Mock Model Server", version="0.1.0-mock")


# ── Request schemas matching what the agent sends ───────────────────────────


class ClusterRequest(BaseModel):
    patient: dict


class SimulatorRequest(BaseModel):
    candidate_pill: dict  # Full pill record from pills.csv
    patient: dict


# ── Cluster Model endpoint ──────────────────────────────────────────────────


@app.post("/cluster/predict")
def cluster_predict(body: ClusterRequest):
    """
    Mimics the Cluster Model API response.
    Assigns cluster based on patient age and condition count to produce
    varied, somewhat realistic cluster assignments.
    """
    patient = body.patient
    age = patient.get("age", 25)
    
    # Count number of conditions (cond_* fields set to 1)
    condition_count = sum(
        1 for key, value in patient.items()
        if key.startswith("cond_") and value == 1
    )
    
    # Check if smoker
    is_smoker = patient.get("obs_smoker", 0) == 1

    if age < 25 and condition_count == 0:
        profile, confidence = "cluster_0", 0.93
    elif age < 35:
        profile, confidence = "cluster_2", round(random.uniform(0.78, 0.92), 2)
    elif age < 45:
        profile, confidence = "cluster_3", round(random.uniform(0.68, 0.82), 2)
    else:
        # Older patients with more conditions → lower confidence
        profile = "cluster_4"
        confidence = round(max(0.50, 0.80 - (condition_count * 0.05)), 2)

    return {"cluster_profile": profile, "cluster_confidence": confidence}


# ── Simulator Model endpoint ────────────────────────────────────────────────

# Deterministic per-pill results so the optimizer always produces the same ranking
PILL_PROFILES = {
    "pill_levonorgestrel_30": (0.10, 0.004, 0.18, 0.98),
    "pill_levonorgestrel_20": (0.12, 0.005, 0.20, 0.97),
    "pill_norgestimate_35": (0.15, 0.008, 0.28, 0.97),
    "pill_desogestrel_30": (0.14, 0.009, 0.30, 0.96),
    "pill_desogestrel_20": (0.16, 0.010, 0.32, 0.96),
    "pill_drospirenone_30": (0.18, 0.011, 0.35, 0.97),
    "pill_drospirenone_20": (0.17, 0.010, 0.33, 0.97),
    "pill_gestodene_30": (0.13, 0.009, 0.27, 0.97),
    "pill_gestodene_20": (0.14, 0.008, 0.25, 0.97),
    "pill_cyproterone_35": (0.20, 0.018, 0.40, 0.96),
    "pill_norethisterone_35": (0.16, 0.006, 0.24, 0.96),
    "pill_norelgestromin_20": (0.15, 0.009, 0.29, 0.97),
    "pill_desogestrel_mini": (0.22, 0.003, 0.35, 0.95),
    "pill_norethisterone_mini": (0.25, 0.002, 0.38, 0.94),
    "pill_drospirenone_mini": (0.20, 0.003, 0.32, 0.95),
    # Test-only pill IDs (used in unit tests)
    "pill_a": (0.10, 0.005, 0.20, 0.98),
    "pill_b": (0.18, 0.009, 0.35, 0.97),
    "pill_c": (0.27, 0.014, 0.55, 0.95),
    "pill_d": (0.33, 0.011, 0.45, 0.96),
    "pill_e": (0.22, 0.007, 0.30, 0.97),
}


@app.post("/simulator/simulate")
def simulator_simulate(body: SimulatorRequest):
    """
    Mimics the Simulator Model API response.
    Returns deterministic results for known pills, seeded random for unknown.
    """
    # Extract pill identifier - could be set_id, pill_id, or derived from brand_name
    pill_record = body.candidate_pill
    pill = pill_record.get("set_id") or pill_record.get("pill_id") or pill_record.get("brand_name", "unknown")

    if pill in PILL_PROFILES:
        disc, severe, mild, effectiveness = PILL_PROFILES[pill]
    else:
        # Deterministic fallback based on pill identifier
        seed = sum(ord(c) for c in str(pill))
        rng = random.Random(seed)
        disc = round(rng.uniform(0.05, 0.40), 3)
        severe = round(rng.uniform(0.001, 0.025), 4)
        mild = round(rng.uniform(0.10, 0.80), 2)
        effectiveness = round(rng.uniform(0.90, 0.99), 3)

    return {
        "discontinuation_probability": disc,
        "severe_event_probability": severe,
        "mild_side_effect_score": mild,
        "contraceptive_effectiveness": effectiveness,
    }


# ── Health check ────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "mode": "MOCK — not for production"}


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("tests.mock_server:app", host="0.0.0.0", port=8001, reload=True)
