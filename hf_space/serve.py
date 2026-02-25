#!/usr/bin/env python3
"""
serve.py — Combined FastAPI inference server for Hugging Face Spaces.

Exposes both model APIs on a single port (7860) as required by HF Spaces:

    POST /api/v1/cluster/predict      ← GMM clustering model
    POST /api/v1/simulator/simulate   ← HistGBM simulation model
    GET  /api/v1/health               ← combined health check
    GET  /                            ← root info

Artifact layout (relative to this file):
    artifacts/clustering/
        gmm_model.pkl
        scaler.pkl
        imputer.pkl
        profile_rules.json
    artifacts/simulation/
        model_symptoms.pkl
        model_satisfaction.pkl
        feature_meta.json
    drugs/output/pill_reference_db.csv
"""

import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER MODEL — artifact loading
# ══════════════════════════════════════════════════════════════════════════════

_CLUSTER_DIR = _DIR / "artifacts" / "clustering"

def _load_pkl_cluster(name: str):
    with open(_CLUSTER_DIR / name, "rb") as f:
        return pickle.load(f)

logger.info("Loading clustering artifacts from %s …", _CLUSTER_DIR)
GMM      = _load_pkl_cluster("gmm_model.pkl")
SCALER   = _load_pkl_cluster("scaler.pkl")
IMP_DICT = _load_pkl_cluster("imputer.pkl")
RULES    = json.loads((_CLUSTER_DIR / "profile_rules.json").read_text())

FEATURE_ORDER = RULES["feature_order"]
AVAIL_CONT    = IMP_DICT["avail_cont"]
AVAIL_BIN     = IMP_DICT["avail_bin"]
IMP_CONT      = IMP_DICT["continuous"]
IMP_BIN       = IMP_DICT["binary"]
logger.info("Clustering artifacts loaded — k=%d profiles, %d features", RULES["k"], len(FEATURE_ORDER))


def _prepare_cluster_features(patient: dict) -> np.ndarray:
    row = {}
    for feat in FEATURE_ORDER:
        val = patient.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            row[feat] = np.nan
        else:
            row[feat] = float(val)

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    cont_arr = IMP_CONT.transform(df[AVAIL_CONT])
    bin_arr  = IMP_BIN.transform(df[AVAIL_BIN])

    df_cont = pd.DataFrame(cont_arr, columns=AVAIL_CONT)
    df_bin  = pd.DataFrame(bin_arr,  columns=AVAIL_BIN)
    df_full = pd.concat([df_cont, df_bin], axis=1)[FEATURE_ORDER]

    df_full[AVAIL_CONT] = SCALER.transform(df_full[AVAIL_CONT])
    return df_full.values  # shape (1, n_features)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR MODEL — artifact loading
# ══════════════════════════════════════════════════════════════════════════════

_SIM_DIR  = _DIR / "artifacts" / "simulation"
_PILLS_CSV = _DIR / "drugs" / "output" / "pill_reference_db.csv"

logger.info("Loading simulation artifacts from %s …", _SIM_DIR)
SYMPTOMS_MODEL     = joblib.load(_SIM_DIR / "model_symptoms.pkl")
SATISFACTION_MODEL = joblib.load(_SIM_DIR / "model_satisfaction.pkl")

with open(_SIM_DIR / "feature_meta.json") as _f:
    META = json.load(_f)

FEATURE_NAMES    = META["feature_names"]
PATIENT_FEATURES = META["patient_features"]
PILL_FEATURES    = META["pill_features"]
BINARY_TARGETS   = META["binary_targets"]
ANDROGENIC_MAP   = META["androgenic_map"]
VTE_MAP          = META["vte_map"]

logger.info(
    "Simulation artifacts loaded — %d features, %d binary targets",
    len(FEATURE_NAMES), len(BINARY_TARGETS),
)


def _build_pill_lookup() -> Dict[str, Dict[str, float]]:
    try:
        pills_df = pd.read_csv(_PILLS_CSV)
    except FileNotFoundError:
        logger.warning("pill_reference_db.csv not found at %s — empty pill lookup", _PILLS_CSV)
        return {}

    p = pills_df.copy()
    p["pill_type_binary"]    = p["pill_type"].apply(lambda x: 0 if "progestin_only" in str(x) else 1)
    p["androgenic_score"]    = p["androgenic_activity"].str.lower().map(ANDROGENIC_MAP).astype(float)
    p["vte_risk_numeric"]    = p["vte_risk_class"].str.lower().map(VTE_MAP).astype(float)
    p["progestin_generation"] = pd.to_numeric(p["progestin_generation"], errors="coerce")
    p["progestin_dose_mg"]   = pd.to_numeric(p["progestin_dose_mg"],    errors="coerce")
    p["estrogen_dose_mcg"]   = pd.to_numeric(p["estrogen_dose_mcg"],    errors="coerce")
    return p.set_index("combo_id")[PILL_FEATURES].to_dict(orient="index")


PILLS_LOOKUP: Dict[str, Dict[str, float]] = _build_pill_lookup()
logger.info("Pill lookup ready — %d pills", len(PILLS_LOOKUP))


def _predict_trajectory(
    patient_row: Dict[str, Any],
    combo_id: str,
    pill_feats_override: Optional[Dict[str, float]] = None,
    n_months: int = 12,
) -> Dict[str, Any]:
    pill_feats = pill_feats_override or PILLS_LOOKUP.get(combo_id, {})
    if not pill_feats:
        logger.warning("combo_id '%s' not found in pill lookup — pill features will be NaN", combo_id)

    rows = []
    for m in range(1, n_months + 1):
        row: Dict[str, Any] = {}
        for feat in FEATURE_NAMES:
            if feat == "month":
                row[feat] = float(m)
            elif feat in PILL_FEATURES:
                val = pill_feats.get(feat)
                row[feat] = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else np.nan
            else:
                val = patient_row.get(feat)
                row[feat] = float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else np.nan
        rows.append(row)

    X_traj = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float).values

    proba_list = SYMPTOMS_MODEL.predict_proba(X_traj)
    sym_probs  = np.column_stack([p[:, 1] for p in proba_list])
    sat_preds  = SATISFACTION_MODEL.predict(X_traj)

    return {
        "combo_id":      combo_id,
        "n_months":      n_months,
        "months":        list(range(1, n_months + 1)),
        "symptom_probs": {tgt: sym_probs[:, i].tolist() for i, tgt in enumerate(BINARY_TARGETS)},
        "satisfaction":  sat_preds.tolist(),
    }


_ADVERSE_SYMS = [
    "sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting",
    "sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
    "sym_libido_decreased", "sym_weight_gain", "sym_acne_worsened", "sym_hair_loss",
]
_SERIOUS_EVTS = ["evt_dvt", "evt_pe", "evt_stroke"]


def _derive_summary_metrics(symptom_probs: Dict[str, List[float]], satisfaction: List[float]) -> Dict[str, float]:
    still_taking = symptom_probs.get("still_taking", [])
    disc_prob = float(1.0 - still_taking[-1]) if still_taking else 0.5

    severe_probs = [max(symptom_probs[e]) for e in _SERIOUS_EVTS if symptom_probs.get(e)]
    severe_prob  = float(np.mean(severe_probs)) if severe_probs else 0.0

    adverse_means = [float(np.mean(symptom_probs[s])) for s in _ADVERSE_SYMS if symptom_probs.get(s)]
    mild_score    = float(np.mean(adverse_means)) if adverse_means else 0.0

    effectiveness = float(np.mean(satisfaction) / 10.0) if satisfaction else 0.5

    return {
        "discontinuation_probability": round(disc_prob,  4),
        "severe_event_probability":    round(severe_prob, 6),
        "mild_side_effect_score":      round(mild_score,  4),
        "contraceptive_effectiveness": round(effectiveness, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class PatientFeatures(BaseModel):
    age: float = Field(..., ge=15, le=60)

    # WHO MEC Cat 4
    cond_migraine_with_aura:  int = Field(0, ge=0, le=1)
    cond_stroke:              int = Field(0, ge=0, le=1)
    cond_mi:                  int = Field(0, ge=0, le=1)
    cond_dvt:                 int = Field(0, ge=0, le=1)
    cond_breast_cancer:       int = Field(0, ge=0, le=1)
    cond_lupus:               int = Field(0, ge=0, le=1)
    cond_thrombophilia:       int = Field(0, ge=0, le=1)
    cond_atrial_fibrillation: int = Field(0, ge=0, le=1)
    cond_liver_disease:       int = Field(0, ge=0, le=1)

    # WHO MEC Cat 3
    cond_hypertension:           int = Field(0, ge=0, le=1)
    cond_migraine:               int = Field(0, ge=0, le=1)
    cond_gallstones:             int = Field(0, ge=0, le=1)
    cond_diabetes:               int = Field(0, ge=0, le=1)
    cond_prediabetes:            int = Field(0, ge=0, le=1)
    cond_epilepsy:               int = Field(0, ge=0, le=1)
    cond_chronic_kidney_disease: int = Field(0, ge=0, le=1)
    cond_sleep_apnea:            int = Field(0, ge=0, le=1)

    # Indications / comorbidities
    cond_pcos:                 int = Field(0, ge=0, le=1)
    cond_endometriosis:        int = Field(0, ge=0, le=1)
    cond_depression:           int = Field(0, ge=0, le=1)
    cond_hypothyroidism:       int = Field(0, ge=0, le=1)
    cond_rheumatoid_arthritis: int = Field(0, ge=0, le=1)
    cond_fibromyalgia:         int = Field(0, ge=0, le=1)
    cond_osteoporosis:         int = Field(0, ge=0, le=1)
    cond_asthma:               int = Field(0, ge=0, le=1)

    # Observations
    obs_bmi:          Optional[float] = Field(None, ge=10.0, le=60.0)
    obs_systolic_bp:  Optional[float] = Field(None, ge=70.0, le=220.0)
    obs_diastolic_bp: Optional[float] = Field(None, ge=40.0, le=140.0)
    obs_phq9_score:   Optional[float] = Field(None, ge=0.0,  le=27.0)
    obs_testosterone: Optional[float] = Field(None, ge=0.0,  le=300.0)
    obs_smoker:       int              = Field(0, ge=0, le=1)
    obs_pain_score:   float            = Field(0.0, ge=0.0, le=10.0)

    # Medication history
    med_ever_ocp:                          int = Field(0, ge=0, le=1)
    med_current_combined_ocp:              int = Field(0, ge=0, le=1)
    med_current_minipill:                  int = Field(0, ge=0, le=1)
    has_absolute_contraindication_combined_oc: int = Field(0, ge=0, le=1)

    class Config:
        extra = "allow"


class ClusterRequest(BaseModel):
    patient: PatientFeatures


class ClusterResponse(BaseModel):
    cluster_profile:    str   = Field(..., description="e.g. 'cluster_5'")
    cluster_confidence: float = Field(..., ge=0.0, le=1.0)


class SimulatorRequest(BaseModel):
    candidate_pill: Dict[str, Any] = Field(..., description="Pill record with at least 'combo_id'")
    patient:        Dict[str, Any] = Field(..., description="Patient feature dict")
    n_months:       int             = Field(12, ge=1, le=12)

    class Config:
        extra = "allow"


class SimulatorResponse(BaseModel):
    combo_id:      str
    n_months:      int
    months:        List[int]
    symptom_probs: Dict[str, List[float]]
    satisfaction:  List[float]

    discontinuation_probability: float
    severe_event_probability:    float
    mild_side_effect_score:      float
    contraceptive_effectiveness: float


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Selene ML APIs",
    version="1.0.0",
    description=(
        "Combined inference server for the Selene birth-control recommendation system. "
        "Cluster model (GMM) + Simulator model (HistGBM) — hosted on Hugging Face Spaces."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Cluster endpoints ──────────────────────────────────────────────────────────

@app.post("/api/v1/cluster/predict", response_model=ClusterResponse)
def cluster_predict(body: ClusterRequest) -> ClusterResponse:
    patient_dict = body.patient.model_dump()
    try:
        X = _prepare_cluster_features(patient_dict)
    except Exception as exc:
        logger.error("Cluster feature prep failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"UNPROCESSABLE_INPUT: {exc}")

    try:
        probs      = GMM.predict_proba(X)[0]
        hard_label = int(np.argmax(probs))
        confidence = float(probs[hard_label])
    except Exception as exc:
        logger.error("GMM inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="MODEL_ERROR: inference failed")

    profile_name = f"cluster_{hard_label}"
    logger.info("Cluster → %s (conf=%.3f)", profile_name, confidence)
    return ClusterResponse(cluster_profile=profile_name, cluster_confidence=confidence)


# ── Simulator endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/simulator/simulate", response_model=SimulatorResponse)
def simulator_simulate(body: SimulatorRequest) -> SimulatorResponse:
    pill     = body.candidate_pill
    patient  = body.patient
    n_months = body.n_months

    combo_id = (
        pill.get("combo_id")
        or pill.get("set_id")
        or pill.get("pill_id")
        or pill.get("brand_name", "unknown")
    )

    pill_feats_override: Optional[Dict[str, float]] = None
    if all(f in pill for f in PILL_FEATURES):
        try:
            pill_feats_override = {
                f: float(pill[f]) for f in PILL_FEATURES
                if pill[f] is not None and not (isinstance(pill[f], float) and math.isnan(pill[f]))
            }
        except (TypeError, ValueError):
            pill_feats_override = None

    try:
        trajectory = _predict_trajectory(patient, combo_id, pill_feats_override, n_months)
    except Exception as exc:
        logger.error("Trajectory inference failed for %s: %s", combo_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"MODEL_ERROR: {exc}")

    summary = _derive_summary_metrics(trajectory["symptom_probs"], trajectory["satisfaction"])
    logger.info(
        "Simulated %s n_months=%d  disc=%.3f severe=%.5f mild=%.3f eff=%.3f",
        combo_id, n_months,
        summary["discontinuation_probability"],
        summary["severe_event_probability"],
        summary["mild_side_effect_score"],
        summary["contraceptive_effectiveness"],
    )

    return SimulatorResponse(
        combo_id=trajectory["combo_id"],
        n_months=trajectory["n_months"],
        months=trajectory["months"],
        symptom_probs=trajectory["symptom_probs"],
        satisfaction=trajectory["satisfaction"],
        **summary,
    )


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    return {
        "status":           "ok",
        "models":           ["cluster", "simulator"],
        "version":          "1.0.0",
        "n_cluster_profiles": RULES["k"],
        "n_pills":          len(PILLS_LOOKUP),
        "n_binary_targets": len(BINARY_TARGETS),
    }


@app.get("/health")
def health_short():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "Selene ML APIs",
        "docs":    "/docs",
        "health":  "/api/v1/health",
        "endpoints": [
            "POST /api/v1/cluster/predict",
            "POST /api/v1/simulator/simulate",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
