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
    Assigns one of the 12 GMM profiles (cluster_0 through cluster_11) based on
    patient conditions — approximating the behaviour of the trained GMM.
    Returns format matching ML_models_API.md: {"cluster_profile": "cluster_N", ...}
    """
    patient = body.patient
    age = patient.get("age", 25)

    # Check Cat-4 absolute contraindications
    has_breast_cancer = patient.get("cond_breast_cancer", 0) == 1
    has_liver_disease = patient.get("cond_liver_disease", 0) == 1
    has_thrombophilia = patient.get("cond_thrombophilia", 0) == 1
    has_thrombophilia_and_pcos = has_thrombophilia and patient.get("cond_pcos", 0) == 1
    has_epilepsy = patient.get("cond_epilepsy", 0) == 1
    has_migraine_aura = patient.get("cond_migraine_with_aura", 0) == 1
    has_hypertension = patient.get("cond_hypertension", 0) == 1
    has_diabetes = patient.get("cond_diabetes", 0) == 1
    has_pcos = patient.get("cond_pcos", 0) == 1
    has_migraine = patient.get("cond_migraine", 0) == 1
    has_depression = patient.get("cond_depression", 0) == 1

    # Route to approximate profile — mirrors GMM blocking logic
    if has_breast_cancer or has_liver_disease:
        # Profile 2: Diabetes + Breast Cancer (all 9 blocked)
        profile, confidence = "cluster_2", 0.91
    elif has_thrombophilia_and_pcos or (has_thrombophilia and has_epilepsy):
        # Profile 6: PCOS + Thrombophilia (all 9 blocked)
        profile, confidence = "cluster_6", 0.87
    elif has_migraine_aura and has_hypertension:
        # Profile 10: Hypertension + Migraine+Aura (combined blocked)
        profile, confidence = "cluster_10", 0.84
    elif has_hypertension and has_diabetes:
        # Profile 3: Hypertension + Diabetes (combined blocked)
        profile, confidence = "cluster_3", 0.82
    elif has_thrombophilia:
        # Profile 4: Thrombophilia + Endometriosis (combined blocked)
        profile, confidence = "cluster_4", 0.85
    elif has_epilepsy:
        # Profile 7: Epilepsy (combined blocked)
        profile, confidence = "cluster_7", 0.88
    elif has_hypertension and has_pcos:
        # Profile 8: PCOS + Hypertension (combined blocked)
        profile, confidence = "cluster_8", 0.80
    elif has_hypertension:
        # Profile 11: Hypertension + Thrombophilia (combined blocked)
        profile, confidence = "cluster_11", 0.78
    elif has_pcos and not has_migraine:
        # Profile 9: PCOS (no blocking)
        profile, confidence = "cluster_9", 0.90
    elif has_migraine or has_depression:
        # Profile 1: Migraine + Depression (no blocking)
        profile, confidence = "cluster_1", 0.86
    else:
        # Profile 5: Baseline / Low-Risk
        profile, confidence = "cluster_5", 0.95

    return {"cluster_profile": profile, "cluster_confidence": confidence}


# ── Simulator Model endpoint ────────────────────────────────────────────────

# Deterministic per-pill results so the optimizer always produces the same ranking.
# Keys are combo_ids from pill_reference_db.csv.
# Tuple: (disc_prob, severe_prob, mild_score, effectiveness)
# Lower disc/severe/mild = better; higher effectiveness = better.
PILL_PROFILES = {
    # ── Real combo_ids from pill_reference_db.csv ───────────────────────────
    # Combined OCPs (8) — ordered roughly by VTE risk class / progestin generation
    "EE20_LNG90":         (0.11, 0.004, 0.18, 0.98),  # Low-dose EE + levonorgestrel (gen2, low VTE)
    "EE30_LNG150":        (0.12, 0.005, 0.20, 0.98),  # Std-dose EE + levonorgestrel
    "EE35_NET500_1000":   (0.13, 0.005, 0.22, 0.97),  # Higher-dose EE + norethindrone
    "EE20_NET1000":       (0.12, 0.005, 0.21, 0.97),  # Low-dose EE + norethindrone
    "EE25_35_NGM":        (0.14, 0.007, 0.25, 0.97),  # Triphasic norgestimate
    "EE30_DSG150":        (0.15, 0.009, 0.28, 0.97),  # Desogestrel (gen3, higher VTE)
    "EE30_DRSP3":         (0.16, 0.010, 0.30, 0.97),  # Drospirenone (gen4)
    "EE20_DRSP3":         (0.17, 0.010, 0.32, 0.97),  # Low-dose EE + drospirenone
    # Progestin-only (minipill) — higher disc, near-zero severe
    "NET_PO_350":         (0.22, 0.002, 0.35, 0.95),  # Norethindrone 0.35 mg
    # ── Test-only pill IDs (used in unit/integration tests) ─────────────────
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
    # Extract pill identifier from combo_id (new format) or fall back to legacy fields
    pill_record = body.candidate_pill
    pill = (
        pill_record.get("combo_id")
        or pill_record.get("set_id")
        or pill_record.get("pill_id")
        or pill_record.get("brand_name", "unknown")
    )

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
