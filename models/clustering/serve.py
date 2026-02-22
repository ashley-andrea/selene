#!/usr/bin/env python3
"""
serve.py — FastAPI inference server for the trained GMM Clustering Model.

Exposes the API contract defined in _docs/ML_models_API.md:

    POST /api/v1/cluster/predict
    GET  /api/v1/health

Artifacts loaded at startup from models/clustering/artifacts/:
    gmm_model.pkl     — trained GaussianMixture (k=12, covariance_type=diag)
    scaler.pkl        — StandardScaler fitted on training continuous features
    imputer.pkl       — dict of {SimpleImputer for cont., SimpleImputer for bin., feature lists}
    profile_rules.json — blocking rules and feature order

Run locally:
    uvicorn models.clustering.serve:app --host 0.0.0.0 --port 8000

Or from this directory:
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = _DIR / "artifacts"

# ──────────────────────────────────────────────────────────────────────────────
#  Load artifacts once at startup
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl(name: str):
    with open(ARTIFACTS_DIR / name, "rb") as f:
        return pickle.load(f)

logger.info("Loading clustering model artifacts from %s …", ARTIFACTS_DIR)
GMM       = _load_pkl("gmm_model.pkl")
SCALER    = _load_pkl("scaler.pkl")
IMP_DICT  = _load_pkl("imputer.pkl")
RULES     = json.loads((ARTIFACTS_DIR / "profile_rules.json").read_text())

FEATURE_ORDER = RULES["feature_order"]
AVAIL_CONT    = IMP_DICT["avail_cont"]
AVAIL_BIN     = IMP_DICT["avail_bin"]
IMP_CONT      = IMP_DICT["continuous"]
IMP_BIN       = IMP_DICT["binary"]
logger.info("Artifacts loaded — k=%d profiles, %d features", RULES["k"], len(FEATURE_ORDER))

# ──────────────────────────────────────────────────────────────────────────────
#  Feature preparation (mirrors evaluate_profiles.prepare_X)
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_features(patient: dict) -> np.ndarray:
    """
    Build a single-row feature matrix from the patient dict.
    Applies the same imputation + scaling pipeline used during training.
    Missing fields default to NaN (continuous) or 0 (binary).
    """
    row = {}
    for feat in FEATURE_ORDER:
        val = patient.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            row[feat] = np.nan
        else:
            row[feat] = float(val)

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    # ── Impute ────────────────────────────────────────────────────────────
    cont_arr = IMP_CONT.transform(df[AVAIL_CONT])
    bin_arr  = IMP_BIN.transform(df[AVAIL_BIN])

    df_cont = pd.DataFrame(cont_arr, columns=AVAIL_CONT)
    df_bin  = pd.DataFrame(bin_arr,  columns=AVAIL_BIN)
    df_full = pd.concat([df_cont, df_bin], axis=1)[FEATURE_ORDER]

    # ── Scale continuous features ─────────────────────────────────────────
    df_full[AVAIL_CONT] = SCALER.transform(df_full[AVAIL_CONT])

    return df_full.values  # shape (1, n_features)


# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Birth-Control Cluster Model API",
    version="1.0.0",
    description=(
        "GMM-based patient clustering for oral contraceptive recommendation. "
        "Assigns one of 12 WHO-MEC-informed risk profiles to each patient."
    ),
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """Full patient feature set — all cond_* and obs_* fields from ML_models_API.md."""
    age: float = Field(..., ge=15, le=60, description="Patient age in years")

    # WHO MEC Cat 4 conditions (binary 0/1)
    cond_migraine_with_aura:  int = Field(0, ge=0, le=1)
    cond_stroke:              int = Field(0, ge=0, le=1)
    cond_mi:                  int = Field(0, ge=0, le=1)
    cond_dvt:                 int = Field(0, ge=0, le=1)
    cond_breast_cancer:       int = Field(0, ge=0, le=1)
    cond_lupus:               int = Field(0, ge=0, le=1)
    cond_thrombophilia:       int = Field(0, ge=0, le=1)
    cond_atrial_fibrillation: int = Field(0, ge=0, le=1)
    cond_liver_disease:       int = Field(0, ge=0, le=1)

    # WHO MEC Cat 3 conditions
    cond_hypertension:           int = Field(0, ge=0, le=1)
    cond_migraine:               int = Field(0, ge=0, le=1)
    cond_gallstones:             int = Field(0, ge=0, le=1)
    cond_diabetes:               int = Field(0, ge=0, le=1)
    cond_prediabetes:            int = Field(0, ge=0, le=1)
    cond_epilepsy:               int = Field(0, ge=0, le=1)
    cond_chronic_kidney_disease: int = Field(0, ge=0, le=1)
    cond_sleep_apnea:            int = Field(0, ge=0, le=1)

    # Indications / comorbidities
    cond_pcos:                  int = Field(0, ge=0, le=1)
    cond_endometriosis:         int = Field(0, ge=0, le=1)
    cond_depression:            int = Field(0, ge=0, le=1)
    cond_hypothyroidism:        int = Field(0, ge=0, le=1)
    cond_rheumatoid_arthritis:  int = Field(0, ge=0, le=1)
    cond_fibromyalgia:          int = Field(0, ge=0, le=1)
    cond_osteoporosis:          int = Field(0, ge=0, le=1)
    cond_asthma:                int = Field(0, ge=0, le=1)

    # Observations — None is allowed; imputer fills missing values with training-set mean
    obs_bmi:          Optional[float] = Field(None, ge=10.0, le=60.0)
    obs_systolic_bp:  Optional[float] = Field(None, ge=70.0, le=220.0)
    obs_diastolic_bp: Optional[float] = Field(None, ge=40.0, le=140.0)
    obs_phq9_score:   Optional[float] = Field(None, ge=0.0, le=27.0)
    obs_testosterone: Optional[float] = Field(None, ge=0.0, le=300.0)
    obs_smoker:       int              = Field(0, ge=0, le=1)

    # Additional fields from API contract (not used by GMM but accepted)
    obs_pain_score:                        float = Field(0.0, ge=0.0, le=10.0)
    med_ever_ocp:                          int   = Field(0, ge=0, le=1)
    med_current_combined_ocp:              int   = Field(0, ge=0, le=1)
    med_current_minipill:                  int   = Field(0, ge=0, le=1)
    has_absolute_contraindication_combined_oc: int = Field(0, ge=0, le=1)

    class Config:
        extra = "allow"  # Accept (and ignore) any additional fields


class ClusterRequest(BaseModel):
    patient: PatientFeatures


class ClusterResponse(BaseModel):
    cluster_profile:    str   = Field(..., description="e.g. 'cluster_5'")
    cluster_confidence: float = Field(..., ge=0.0, le=1.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/cluster/predict",
    response_model=ClusterResponse,
    summary="Assign patient to a risk profile",
)
def cluster_predict(body: ClusterRequest) -> ClusterResponse:
    """
    Runs the GMM model on the patient feature vector and returns:
    - cluster_profile: "cluster_N" (N = 0 … 11)
    - cluster_confidence: highest probability across all profiles (0–1)
    """
    patient_dict = body.patient.model_dump()

    try:
        X = _prepare_features(patient_dict)
    except Exception as exc:
        logger.error("Feature preparation failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"UNPROCESSABLE_INPUT: {exc}")

    try:
        probs       = GMM.predict_proba(X)[0]       # shape (k,)
        hard_label  = int(np.argmax(probs))
        confidence  = float(probs[hard_label])
    except Exception as exc:
        logger.error("GMM inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="MODEL_ERROR: inference failed")

    profile_name = f"cluster_{hard_label}"
    logger.info(
        "Assigned %s  confidence=%.3f  probs=%s",
        profile_name,
        confidence,
        ", ".join(f"P{i}={p:.3f}" for i, p in enumerate(probs)),
    )
    return ClusterResponse(cluster_profile=profile_name, cluster_confidence=confidence)


@app.get("/api/v1/health", summary="Health check")
def health():
    """Returns OK when the model is loaded and ready to serve."""
    return {"status": "ok", "model": "cluster", "version": "1.0.0"}


# Legacy health path (some OpenShift probes use /health)
@app.get("/health")
def health_short():
    return {"status": "ok", "model": "cluster", "version": "1.0.0"}


# ──────────────────────────────────────────────────────────────────────────────
#  Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
