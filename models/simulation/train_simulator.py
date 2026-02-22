#!/usr/bin/env python3
"""
train_simulator.py
==================
Trains the Simulator Model — the core of the recommendation pipeline.

At inference the agent passes (patient_features, combo_id) and the model
returns a monthly trajectory of predicted symptom probabilities and a
satisfaction score for the full simulation horizon (default: 12 months).

Architecture
------------
Two models, both backed by sklearn HistGradientBoosting:

  1. symptoms_model  — MultiOutputClassifier(HistGBM)
     Input  : patient features + pill pharmacological encoding + month
     Output : predict_proba for 18 binary targets
              14 x sym_*  (nausea, headache, spotting, mood …)
               3 x evt_*  (dvt, pe, stroke — included but very rare)
               1 x still_taking

  2. satisfaction_model — HistGBMRegressor
     Same input → satisfaction_score (continuous, 1-10)

Why HistGBM
-----------
  • Handles NaN natively (no separate imputer needed; PHQ-9 is 64% missing,
    testosterone 92% missing in the patient dataset)
  • Learns non-linear temporal curves from month as a plain numeric feature
  • predict_proba produces well-calibrated risk probabilities
  • Fast training on 400k rows; compact serialised artifacts

Feature set
-----------
  Patient  : age, obs_bmi, obs_systolic_bp, obs_diastolic_bp, obs_phq9_score,
             obs_testosterone, obs_smoker, all 23 cond_* flags, med_ever_ocp
  Pill     : pill_type_binary, estrogen_dose_mcg, progestin_dose_mg,
             progestin_generation, androgenic_score, vte_risk_numeric
  Temporal : month  (1 … N, where N = simulation horizon)

Train / test split
------------------
  Follows the patient-wise 80/20 split created by train_profiles.py
  (data/splits/train_ids.txt and test_ids.txt). Only train IDs are used here.

Usage
-----
    python train_simulator.py
    python train_simulator.py --max-iter 100 --seed 42

Outputs
-------
    models/simulation/artifacts/model_symptoms.pkl
    models/simulation/artifacts/model_satisfaction.pkl
    models/simulation/artifacts/feature_meta.json
"""

import os
import json
import pickle
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PATIENTS_CSV  = os.path.join(PROJECT_ROOT, "data/output/patients_flat.csv")
DIARIES_CSV   = os.path.join(PROJECT_ROOT, "data/output/symptom_diaries.csv")
PILLS_CSV     = os.path.join(PROJECT_ROOT, "drugs/output/pill_reference_db.csv")
TRAIN_IDS_TXT = os.path.join(PROJECT_ROOT, "data/splits/train_ids.txt")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")


# ──────────────────────────────────────────────────────────────────────────────
#  Feature definitions
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_FEATURES = [
    # Continuous vitals (NaN handled natively by HistGBM)
    "age", "obs_bmi", "obs_systolic_bp", "obs_diastolic_bp",
    "obs_phq9_score", "obs_testosterone",
    # Binary vitals
    "obs_smoker",
    # WHO MEC Cat 4 conditions
    "cond_migraine_with_aura", "cond_stroke", "cond_mi", "cond_dvt",
    "cond_breast_cancer", "cond_lupus", "cond_thrombophilia",
    "cond_atrial_fibrillation", "cond_liver_disease",
    # WHO MEC Cat 3 conditions
    "cond_hypertension", "cond_migraine", "cond_gallstones",
    "cond_diabetes", "cond_prediabetes", "cond_epilepsy",
    "cond_chronic_kidney_disease", "cond_sleep_apnea",
    # Indications
    "cond_pcos", "cond_endometriosis",
    # Comorbidities affecting side-effect profile
    "cond_depression", "cond_hypothyroidism", "cond_rheumatoid_arthritis",
    "cond_fibromyalgia", "cond_osteoporosis", "cond_asthma",
    # OCP history (relevant for tolerability)
    "med_ever_ocp",
]

PILL_FEATURES = [
    "pill_type_binary",       # 1=combined, 0=progestin-only
    "estrogen_dose_mcg",      # 0, 20, 25, 30, 35
    "progestin_dose_mg",
    "progestin_generation",   # 1–4 ordinal
    "androgenic_score",       # anti=-1, low=1, moderate=2, high=3
    "vte_risk_numeric",       # very_low=1 … high=5
]

TEMPORAL_FEATURES = ["month"]   # 1–N_months

ALL_FEATURES = PATIENT_FEATURES + PILL_FEATURES + TEMPORAL_FEATURES

# Binary prediction targets
BINARY_TARGETS = [
    # Tolerability
    "sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting",
    # Mood / mental health
    "sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
    "sym_libido_decreased",
    # Weight / cosmetic
    "sym_weight_gain", "sym_acne_improved", "sym_acne_worsened",
    "sym_hair_loss",
    # Positive effects
    "sym_cramps_relieved", "sym_pcos_improvement",
    # Serious events (very rare — included for completeness;
    # primary serious-event gate is the WHO MEC profile layer)
    "evt_dvt", "evt_pe", "evt_stroke",
    # Continuation
    "still_taking",
]

CONTINUOUS_TARGETS = ["satisfaction_score"]

# ──────────────────────────────────────────────────────────────────────────────
#  Pill pharmacological encoding maps
# ──────────────────────────────────────────────────────────────────────────────
ANDROGENIC_MAP = {
    "anti_androgenic": -1,
    "low":              1,
    "moderate":         2,
    "high":             3,
}

VTE_MAP = {
    "very_low":    1,
    "low":         2,
    "low_moderate": 3,
    "moderate":    4,
    "high":        5,
}


def encode_pills(pills_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with engineered numeric pill features keyed by combo_id."""
    p = pills_df.copy()

    p["pill_type_binary"] = p["pill_type"].apply(
        lambda x: 0 if "progestin_only" in str(x) else 1
    )
    p["androgenic_score"] = p["androgenic_activity"].str.lower().map(ANDROGENIC_MAP)
    p["vte_risk_numeric"]  = p["vte_risk_class"].str.lower().map(VTE_MAP)
    p["progestin_generation"] = pd.to_numeric(p["progestin_generation"], errors="coerce")
    p["progestin_dose_mg"]    = pd.to_numeric(p["progestin_dose_mg"],    errors="coerce")
    p["estrogen_dose_mcg"]    = pd.to_numeric(p["estrogen_dose_mcg"],    errors="coerce")

    return p.set_index("combo_id")[PILL_FEATURES]


# ──────────────────────────────────────────────────────────────────────────────
#  Data loader
# ──────────────────────────────────────────────────────────────────────────────
def load_training_data(
    patients_csv: str,
    diaries_csv:  str,
    pills_csv:    str,
    train_ids_txt: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    X_train  : feature matrix  (n_train, n_features)
    Y_binary : binary targets  (n_train, n_binary_targets)
    Y_cont   : continuous targets (n_train, 1)
    """
    print(f"  Loading patients …")
    patients = pd.read_csv(patients_csv)
    train_ids = set(open(train_ids_txt).read().split())
    patients = patients[patients["patient_id"].astype(str).isin(train_ids)].copy()
    patients["patient_id"] = patients["patient_id"].astype(str)
    print(f"  Train patients : {len(patients):,}")

    print(f"  Loading pill encodings …")
    pills_df    = pd.read_csv(pills_csv)
    pill_feats  = encode_pills(pills_df)

    print(f"  Loading symptom diaries (~70 MB) …")
    t0 = time.time()
    diaries = pd.read_csv(diaries_csv, dtype={"patient_id": str})
    diaries = diaries[diaries["patient_id"].isin(train_ids)].copy()
    print(f"  Diaries loaded in {time.time()-t0:.1f}s  →  {len(diaries):,} rows")

    # ── Merge patient features ────────────────────────────────────────────
    print("  Merging features …")
    patient_cols = ["patient_id"] + [f for f in PATIENT_FEATURES if f in patients.columns]
    merged = diaries.merge(patients[patient_cols], on="patient_id", how="left")

    # ── Merge pill features ───────────────────────────────────────────────
    for feat in PILL_FEATURES:
        merged[feat] = merged["combo_id"].map(pill_feats[feat])

    # ── Build final matrices ──────────────────────────────────────────────
    avail_features = [f for f in ALL_FEATURES if f in merged.columns]
    missing        = [f for f in ALL_FEATURES if f not in merged.columns]
    if missing:
        print(f"  [WARN] Missing features (will be absent): {missing}")

    X      = merged[avail_features].astype(float)
    Y_bin  = merged[[t for t in BINARY_TARGETS  if t in merged.columns]].astype(float)
    Y_cont = merged[[t for t in CONTINUOUS_TARGETS if t in merged.columns]].astype(float)

    return X, Y_bin, Y_cont, avail_features


# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────
def train(
    max_iter:       int   = 80,
    max_leaf_nodes: int   = 31,
    learning_rate:  float = 0.05,
    seed:           int   = 42,
    n_jobs:         int   = -1,
) -> None:

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"\n{'─'*62}")
    print("  STEP 1 — Load training data")
    print(f"{'─'*62}")
    X, Y_bin, Y_cont, feature_names = load_training_data(
        PATIENTS_CSV, DIARIES_CSV, PILLS_CSV, TRAIN_IDS_TXT
    )
    print(f"\n  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Binary targets : {Y_bin.shape[1]}")
    print(f"  Regression targets : {Y_cont.shape[1]}")

    # Quick NaN report (HistGBM handles these natively — just informational)
    nan_pct = X.isnull().mean() * 100
    high_nan = nan_pct[nan_pct > 10]
    if len(high_nan):
        print(f"\n  Features with >10% NaN (handled natively by HistGBM):")
        for f, pct in high_nan.items():
            print(f"    {f:<30}  {pct:.1f}%")

    # ── Target base rates ─────────────────────────────────────────────────
    print(f"\n  Binary target base rates:")
    for col in Y_bin.columns:
        rate = Y_bin[col].mean()
        bar  = "█" * int(rate * 200)
        print(f"    {col:<30}  {rate:.4f}  {bar}")

    print(f"\n  satisfaction_score  mean={Y_cont.iloc[:,0].mean():.2f}  "
          f"std={Y_cont.iloc[:,0].std():.2f}")

    # ── Build base estimator ──────────────────────────────────────────────
    base_clf = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=0.1,
        min_samples_leaf=20,
        random_state=seed,
    )

    # ── Train symptoms model ──────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  STEP 2 — Train symptoms model")
    print(f"  MultiOutputClassifier(HistGBM, max_iter={max_iter})")
    print(f"{'─'*62}")
    t0 = time.time()
    symptoms_model = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
    symptoms_model.fit(X.values, Y_bin.values)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Train satisfaction model ──────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 3 — Train satisfaction model  (HistGBMRegressor)")
    print(f"{'─'*62}")
    t0 = time.time()
    satisfaction_model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=0.1,
        min_samples_leaf=20,
        random_state=seed,
    )
    sat_mask = Y_cont.iloc[:, 0].notna()
    n_dropped = (~sat_mask).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped:,} rows with NaN satisfaction_score")
    satisfaction_model.fit(X.values[sat_mask], Y_cont.values[sat_mask].ravel())
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Save artifacts ────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 4 — Save artifacts")
    print(f"{'─'*62}")

    def _save(obj, name):
        import joblib
        path = os.path.join(ARTIFACTS_DIR, name)
        joblib.dump(obj, path, compress=3)
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓ {name:<40} ({size_kb:.0f} KB)")

    _save(symptoms_model,     "model_symptoms.pkl")
    _save(satisfaction_model, "model_satisfaction.pkl")

    meta = {
        "feature_names":      feature_names,
        "patient_features":   [f for f in PATIENT_FEATURES if f in feature_names],
        "pill_features":      PILL_FEATURES,
        "temporal_features":  TEMPORAL_FEATURES,
        "binary_targets":     list(Y_bin.columns),
        "continuous_targets": list(Y_cont.columns),
        "androgenic_map":     ANDROGENIC_MAP,
        "vte_map":            VTE_MAP,
        "hyperparams": {
            "max_iter":       max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "learning_rate":  learning_rate,
            "seed":           seed,
        },
        "training_rows": int(X.shape[0]),
    }
    meta_path = os.path.join(ARTIFACTS_DIR, "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ feature_meta.json")

    print(f"\n{'═'*62}")
    print(f"  TRAINING COMPLETE")
    print(f"  Artifacts → {ARTIFACTS_DIR}")
    print(f"{'═'*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Inference helper (used by evaluate_simulator.py and the agent)
# ──────────────────────────────────────────────────────────────────────────────
def load_models():
    """Load trained models and metadata from artifacts directory."""
    import joblib
    symptoms_model     = joblib.load(os.path.join(ARTIFACTS_DIR, "model_symptoms.pkl"))
    satisfaction_model = joblib.load(os.path.join(ARTIFACTS_DIR, "model_satisfaction.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "feature_meta.json")) as f:
        meta = json.load(f)
    return symptoms_model, satisfaction_model, meta


def build_pill_features_lookup(pills_csv: str = PILLS_CSV) -> dict:
    """Return dict: combo_id → {pill_feature: value}."""
    pills_df   = pd.read_csv(pills_csv)
    pill_feats = encode_pills(pills_df)
    return pill_feats.to_dict(orient="index")


def predict_trajectory(
    patient_row:   dict,
    combo_id:      str,
    pills_lookup:  dict,
    symptoms_model,
    satisfaction_model,
    meta:          dict,
    n_months:      int = 12,
) -> dict:
    """
    Predict monthly symptom probability trajectories for one patient × one pill.

    Parameters
    ----------
    patient_row   : dict of patient features (raw values, NaN for unknowns)
    combo_id      : pill identifier, e.g. "EE30_DRSP3"
    pills_lookup  : output of build_pill_features_lookup()
    n_months      : simulation horizon (default 12)

    Returns
    -------
    dict with keys:
        "combo_id"     : str
        "n_months"     : int
        "months"       : [1, 2, ..., n_months]
        "symptom_probs": {symptom: [p_month1, p_month2, ...]}
        "satisfaction" : [score_month1, ...]
    """
    feat_names   = meta["feature_names"]
    pill_feats   = pills_lookup.get(combo_id, {})
    binary_targets = meta["binary_targets"]

    rows = []
    for m in range(1, n_months + 1):
        row = {}
        for f in feat_names:
            if f == "month":
                row[f] = m
            elif f in PILL_FEATURES:
                row[f] = pill_feats.get(f, np.nan)
            else:
                row[f] = patient_row.get(f, np.nan)
        rows.append(row)

    X_traj = pd.DataFrame(rows, columns=feat_names).astype(float).values

    # Binary probabilities: shape (n_months, n_targets)
    proba_list = symptoms_model.predict_proba(X_traj)
    # MultiOutputClassifier.predict_proba returns a list of (n_months, 2) arrays
    # We want P(class=1) for each target
    sym_probs = np.column_stack([p[:, 1] for p in proba_list])  # (n_months, n_targets)

    sat_preds = satisfaction_model.predict(X_traj)  # (n_months,)

    symptom_probs = {
        tgt: sym_probs[:, i].tolist()
        for i, tgt in enumerate(binary_targets)
    }

    return {
        "combo_id":      combo_id,
        "n_months":      n_months,
        "months":        list(range(1, n_months + 1)),
        "symptom_probs": symptom_probs,
        "satisfaction":  sat_preds.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train the OCP symptom simulation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-iter",       type=int,   default=80)
    parser.add_argument("--max-leaf-nodes", type=int,   default=31)
    parser.add_argument("--learning-rate",  type=float, default=0.05)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--n-jobs",         type=int,   default=-1,
                        help="Parallel jobs for MultiOutputClassifier (-1 = all cores)")
    args = parser.parse_args()

    train(
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        learning_rate=args.learning_rate,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
