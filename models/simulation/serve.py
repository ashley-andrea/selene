#!/usr/bin/env python3
"""
serve.py — FastAPI inference server for the trained Simulator Model.

Exposes:
    POST /api/v1/simulator/simulate
    GET  /api/v1/health

Artifacts loaded at startup from models/simulation/artifacts/:
    model_symptoms.pkl     — MultiOutputClassifier(HistGBM), 18 binary targets
    model_satisfaction.pkl — HistGBMRegressor, satisfaction_score 1–10
    feature_meta.json      — feature / target names, pill encoding maps

Run locally:
    uvicorn models.simulation.serve:app --host 0.0.0.0 --port 8001

Or from this directory:
    uvicorn serve:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
_DIR          = Path(__file__).resolve().parent
ARTIFACTS_DIR = _DIR / "artifacts"
# pill_reference_db.csv lives at different relative paths:
#   Docker image  → /app/drugs/output/...   (serve.py is at /app/serve.py)
#   Local dev     → ../../drugs/output/...  (serve.py is at models/simulation/serve.py)
_PILLS_DOCKER = _DIR / "drugs" / "output" / "pill_reference_db.csv"
_PILLS_LOCAL  = _DIR.parent.parent / "drugs" / "output" / "pill_reference_db.csv"
PILLS_CSV = _PILLS_DOCKER if _PILLS_DOCKER.exists() else _PILLS_LOCAL

# ──────────────────────────────────────────────────────────────────────────────
#  Load model artifacts once at startup
# ──────────────────────────────────────────────────────────────────────────────
logger.info("Loading simulation model artifacts from %s …", ARTIFACTS_DIR)

SYMPTOMS_MODEL     = joblib.load(ARTIFACTS_DIR / "model_symptoms.pkl")
SATISFACTION_MODEL = joblib.load(ARTIFACTS_DIR / "model_satisfaction.pkl")

with open(ARTIFACTS_DIR / "feature_meta.json") as _f:
    META = json.load(_f)

FEATURE_NAMES    = META["feature_names"]       # ordered list used during training
PATIENT_FEATURES = META["patient_features"]
PILL_FEATURES    = META["pill_features"]
BINARY_TARGETS   = META["binary_targets"]
ANDROGENIC_MAP   = META["androgenic_map"]      # str → int
VTE_MAP          = META["vte_map"]             # str → int

logger.info(
    "Simulation models loaded — %d features, %d binary targets",
    len(FEATURE_NAMES), len(BINARY_TARGETS),
)

# ──────────────────────────────────────────────────────────────────────────────
#  Pill pharmacological encoding lookup
#  Built once from pill_reference_db.csv; matches encode_pills() in train_simulator.py
# ──────────────────────────────────────────────────────────────────────────────

def _build_pill_lookup() -> Dict[str, Dict[str, float]]:
    """Return {combo_id: {pill_feature: value}} for all pills in the DB."""
    try:
        pills_df = pd.read_csv(PILLS_CSV)
    except FileNotFoundError:
        logger.warning(
            "pill_reference_db.csv not found at %s — serving with empty pill lookup. "
            "Pill features must be supplied inline in the request.",
            PILLS_CSV,
        )
        return {}

    p = pills_df.copy()
    p["pill_type_binary"] = p["pill_type"].apply(
        lambda x: 0 if "progestin_only" in str(x) else 1
    )
    p["androgenic_score"] = (
        p["androgenic_activity"].str.lower().map(ANDROGENIC_MAP).astype(float)
    )
    p["vte_risk_numeric"] = (
        p["vte_risk_class"].str.lower().map(VTE_MAP).astype(float)
    )
    p["progestin_generation"] = pd.to_numeric(p["progestin_generation"], errors="coerce")
    p["progestin_dose_mg"]    = pd.to_numeric(p["progestin_dose_mg"],    errors="coerce")
    p["estrogen_dose_mcg"]    = pd.to_numeric(p["estrogen_dose_mcg"],    errors="coerce")

    return p.set_index("combo_id")[PILL_FEATURES].to_dict(orient="index")


PILLS_LOOKUP: Dict[str, Dict[str, float]] = _build_pill_lookup()
logger.info("Pill lookup ready — %d pills", len(PILLS_LOOKUP))

# ──────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def _predict_trajectory(
    patient_row: Dict[str, Any],
    combo_id: str,
    pill_feats_override: Optional[Dict[str, float]] = None,
    n_months: int = 12,
) -> Dict[str, Any]:
    """
    Build a month-by-month feature matrix and run both models.

    pill_feats_override: if provided, these pharmacological values are used
    instead of the lookup (useful when the pill is not yet in the DB).
    """
    pill_feats = pill_feats_override or PILLS_LOOKUP.get(combo_id, {})
    if not pill_feats:
        logger.warning(
            "combo_id '%s' not found in pill lookup and no override supplied — "
            "pill features will be NaN (model may produce degraded output).",
            combo_id,
        )

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

    X_traj = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float).values  # (n_months, n_features)

    # Binary probabilities: list of (n_months, 2) arrays — one per target
    proba_list = SYMPTOMS_MODEL.predict_proba(X_traj)
    sym_probs = np.column_stack([p[:, 1] for p in proba_list])  # (n_months, n_targets)

    sat_preds = SATISFACTION_MODEL.predict(X_traj)  # (n_months,)

    symptom_probs: Dict[str, List[float]] = {
        tgt: sym_probs[:, i].tolist()
        for i, tgt in enumerate(BINARY_TARGETS)
    }

    return {
        "combo_id":      combo_id,
        "n_months":      n_months,
        "months":        list(range(1, n_months + 1)),
        "symptom_probs": symptom_probs,
        "satisfaction":  sat_preds.tolist(),
    }


# Adverse symptom columns used to compute mild side-effect score
_ADVERSE_SYMS = [
    "sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting",
    "sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
    "sym_libido_decreased", "sym_weight_gain", "sym_acne_worsened", "sym_hair_loss",
]
_SERIOUS_EVTS = ["evt_dvt", "evt_pe", "evt_stroke"]


def _derive_summary_metrics(symptom_probs: Dict[str, List[float]], satisfaction: List[float]) -> Dict[str, float]:
    """
    Derive the 4 backward-compatible summary metrics from the trajectory output.

    These are the fields consumed by the agent's utility node:
        discontinuation_probability — P(patient stops by end of horizon)
        severe_event_probability    — peak per-month probability of any serious event
        mild_side_effect_score      — mean across adverse symptoms and months
        contraceptive_effectiveness — mean satisfaction normalised to [0, 1]

    Formula rationale
    -----------------
    • still_taking[-1] gives the model's estimate of adherence at month N.
      1 − that = probability of having discontinued by month N.
    • We take max(evt_*) across months and then average the three events so
      that a very high single-month DVT spike is reflected.
    • Mild score is a simple mean over the 11 adverse symptom columns and time;
      it already lives in [0, 1] (sigmoid outputs from the classifier).
    • Satisfaction is 1–10; dividing by 10 gives a [0, 1] proxy for perceived
      effectiveness / tolerability (closely correlated with adherence).
    """
    # --- discontinuation probability ---
    still_taking = symptom_probs.get("still_taking", [])
    if still_taking:
        disc_prob = float(1.0 - still_taking[-1])
    else:
        disc_prob = 0.5  # conservative default if target absent

    # --- severe event probability ---
    severe_probs = []
    for evt in _SERIOUS_EVTS:
        probs = symptom_probs.get(evt, [])
        if probs:
            severe_probs.append(max(probs))
    severe_prob = float(np.mean(severe_probs)) if severe_probs else 0.0

    # --- mild side-effect score ---
    adverse_means = []
    for sym in _ADVERSE_SYMS:
        probs = symptom_probs.get(sym, [])
        if probs:
            adverse_means.append(float(np.mean(probs)))
    mild_score = float(np.mean(adverse_means)) if adverse_means else 0.0

    # --- contraceptive effectiveness (satisfaction proxy) ---
    effectiveness = float(np.mean(satisfaction) / 10.0) if satisfaction else 0.5

    return {
        "discontinuation_probability": round(disc_prob, 4),
        "severe_event_probability":    round(severe_prob, 6),
        "mild_side_effect_score":      round(mild_score, 4),
        "contraceptive_effectiveness": round(effectiveness, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Birth-Control Simulator Model API",
    version="1.0.0",
    description=(
        "Predicts per-patient, per-pill monthly symptom probability trajectories "
        "and satisfaction scores using HistGradientBoosting models trained on "
        "synthetic patient diaries."
    ),
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class SimulatorRequest(BaseModel):
    """
    Accepts the same payload shape that the agent's simulator_api.py sends.

    candidate_pill: full pill record dict from pill_reference_db.csv.
                    Must at minimum contain 'combo_id'.
                    If pharmacological features (pill_type_binary, etc.) are
                    present they are used directly, otherwise the DB lookup
                    is used to fill them.
    patient:        Patient feature dict — all cond_*, obs_*, med_* fields
                    as specified in ML_models_API.md. Unknown fields are silently
                    ignored; missing features default to NaN (handled natively
                    by HistGBM).
    n_months:       Simulation horizon 1–12 (default 12).
    """
    candidate_pill: Dict[str, Any] = Field(
        ..., description="Pill record containing at minimum 'combo_id'."
    )
    patient: Dict[str, Any] = Field(
        ..., description="Patient feature dict (cond_*, obs_*, med_* fields)."
    )
    n_months: int = Field(
        12, ge=1, le=12,
        description="Number of months to simulate (default 12).",
    )

    class Config:
        extra = "allow"


class SimulatorResponse(BaseModel):
    """
    Full trajectory response PLUS backward-compatible 4-metric summary.

    The summary metrics (discontinuation_probability, severe_event_probability,
    mild_side_effect_score, contraceptive_effectiveness) are derived from the
    trajectory and match exactly the schema the agent's utility node expects.
    """
    combo_id:     str
    n_months:     int
    months:       List[int]
    symptom_probs: Dict[str, List[float]]
    satisfaction: List[float]

    # Backward-compatible summary metrics for the agent utility node
    discontinuation_probability: float
    severe_event_probability:    float
    mild_side_effect_score:      float
    contraceptive_effectiveness: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/simulator/simulate",
    response_model=SimulatorResponse,
    summary="Predict symptom trajectory for one patient × one pill",
)
def simulator_simulate(body: SimulatorRequest) -> SimulatorResponse:
    """
    Runs both HistGBM models and returns:
    - Monthly symptom probability trajectories (all 18 binary targets)
    - Monthly satisfaction scores (1–10)
    - Derived 4-metric summary (for backward compatibility with utility node)
    """
    pill   = body.candidate_pill
    patient = body.patient
    n_months = body.n_months

    # Extract combo_id
    combo_id = (
        pill.get("combo_id")
        or pill.get("set_id")
        or pill.get("pill_id")
        or pill.get("brand_name", "unknown")
    )

    # If the request already includes encoded pill features, use them directly
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


@app.get("/api/v1/health", summary="Health check")
def health():
    """Returns OK when both models are loaded and the pill lookup is ready."""
    return {
        "status":  "ok",
        "model":   "simulator",
        "version": "1.0.0",
        "n_pills": len(PILLS_LOOKUP),
        "n_binary_targets": len(BINARY_TARGETS),
    }


# Legacy health path (some OpenShift probes use /health)
@app.get("/health")
def health_short():
    return {"status": "ok", "model": "simulator", "version": "1.0.0"}


# ──────────────────────────────────────────────────────────────────────────────
#  Local dev entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8001, reload=False)
