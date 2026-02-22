#!/usr/bin/env python3
"""
train_profiles.py
=================
Trains a Gaussian Mixture Model (GMM) on the patient feature matrix to cluster
patients into fuzzy risk profiles. For each profile the script derives a set of
WHO MEC–based safety blocking rules (Cat 3 AND Cat 4 both hard-blocked per
conservative policy). All artefacts needed for inference are serialised to
models/clustering/artifacts/.

Design rationale
----------------
  • GMM over k-means: predict_proba() gives direct P(profile_k | patient)
    without hacking centroid distances into probabilities.
  • BIC for k selection: more principled than the elbow, avoids ambiguity.
  • covariance_type="diag": appropriate for mixed binary/continuous features,
    avoids overfitting on ~4k training samples.
  • Blocking rules are PROFILE-LEVEL (centroid-based). At inference, a second
    pass applies per-patient WHO MEC rules directly on top.

Blocking thresholds
-------------------
  _C4 = 0.15  → if ≥15% of a cluster has a Cat 4 condition, block for whole profile
  _C3 = 0.25  → if ≥25% of a cluster has a Cat 3 condition, block for whole profile

Inference threshold
-------------------
  Default θ = 0.40 → if P(profile_k | patient) > 0.40, apply that profile's
  blocking rules. The union of all triggered profiles applies.

Usage
-----
    python train_profiles.py
    python train_profiles.py --threshold 0.40 --k-min 3 --k-max 12 --seed 42

Outputs
-------
    models/clustering/artifacts/gmm_model.pkl
    models/clustering/artifacts/scaler.pkl
    models/clustering/artifacts/imputer.pkl
    models/clustering/artifacts/feature_order.json
    models/clustering/artifacts/profile_rules.json
    models/clustering/artifacts/bic_curve.csv
    models/clustering/artifacts/bic_curve.png   (requires matplotlib)
    data/splits/train_ids.txt
    data/splits/test_ids.txt
"""

import os
import sys
import json
import pickle
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ──────────────────────────────────────────────────────────────────────────────
#  Paths  (resolved relative to this script's location)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PATIENTS_CSV  = os.path.join(PROJECT_ROOT, "data/output/patients_flat.csv")
SPLITS_DIR    = os.path.join(PROJECT_ROOT, "data/splits")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")


# ──────────────────────────────────────────────────────────────────────────────
#  Pill combo IDs  (from pill_reference_db.csv)
# ──────────────────────────────────────────────────────────────────────────────
ALL_COMBINED = [
    "EE20_LNG90", "EE30_LNG150", "EE35_NET500_1000", "EE20_NET1000",
    "EE25_35_NGM", "EE30_DSG150", "EE30_DRSP3", "EE20_DRSP3",
]
ALL_PILLS = ALL_COMBINED + ["NET_PO_350"]


# ──────────────────────────────────────────────────────────────────────────────
#  Feature definitions
#  Note: med_* and has_absolute_contraindication_* are intentionally excluded —
#  they are derived flags / prior decisions, not intrinsic patient characteristics.
#  obs_pain_score is excluded (too non-specific, not pharmacologically informative).
# ──────────────────────────────────────────────────────────────────────────────
CONTINUOUS_FEATURES = [
    "age",
    "obs_bmi",
    "obs_systolic_bp",
    "obs_diastolic_bp",
    "obs_phq9_score",
    "obs_testosterone",
]

BINARY_FEATURES = [
    "obs_smoker",
    # ── WHO MEC Cat 4: absolute contraindications for combined OCPs ────────
    "cond_migraine_with_aura",
    "cond_stroke",
    "cond_mi",
    "cond_dvt",
    "cond_breast_cancer",
    "cond_lupus",
    "cond_thrombophilia",
    "cond_atrial_fibrillation",
    "cond_liver_disease",
    # ── WHO MEC Cat 3: relative contraindications ──────────────────────────
    "cond_hypertension",
    "cond_migraine",
    "cond_gallstones",
    "cond_diabetes",
    "cond_prediabetes",
    "cond_epilepsy",
    "cond_chronic_kidney_disease",
    "cond_sleep_apnea",
    # ── OCP indications (positive use-case signals) ────────────────────────
    "cond_pcos",
    "cond_endometriosis",
    # ── Comorbidities relevant to side-effect profile ──────────────────────
    "cond_depression",
    "cond_hypothyroidism",
    "cond_rheumatoid_arthritis",
    "cond_fibromyalgia",
    "cond_osteoporosis",
    "cond_asthma",
]

ALL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
#  WHO MEC blocking rules
#  Tuple: (feature, centroid_threshold, mec_category, reason_string, blocked_ids)
#
#  centroid_threshold: fraction of the cluster with this condition that triggers
#  a hard block for the ENTIRE profile (conservative policy: Cat 3 + Cat 4).
# ──────────────────────────────────────────────────────────────────────────────
_C4 = 0.15   # Cat 4 trigger: ≥15% of cluster has absolute contraindication
_C3 = 0.25   # Cat 3 trigger: ≥25% of cluster has relative contraindication

WHO_MEC_RULES = [
    # ── Cat 4: absolute contraindications ─────────────────────────────────
    ("cond_migraine_with_aura",
     _C4, 4,
     "Migraine with aura — estrogen ↑ ischemic stroke risk (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_stroke",
     _C4, 4,
     "History of stroke — no estrogen (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_mi",
     _C4, 4,
     "History of myocardial infarction — no estrogen (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_dvt",
     _C4, 4,
     "History of DVT/VTE — estrogen ↑ recurrence risk (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_breast_cancer",
     _C4, 4,
     "Breast cancer — all hormonal OCPs contraindicated (WHO MEC Cat 4)",
     ALL_PILLS),

    ("cond_lupus",
     _C4, 4,
     "SLE — estrogen increases APS and VTE risk (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_thrombophilia",
     _C4, 4,
     "Hereditary thrombophilia — VTE risk ×8.4 with estrogen "
     "(Bloemenkamp Lancet 1995; WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_atrial_fibrillation",
     _C4, 4,
     "Atrial fibrillation — thromboembolism risk with estrogen (WHO MEC Cat 4)",
     ALL_COMBINED),

    ("cond_liver_disease",
     _C4, 4,
     "Liver disease — impaired hormone metabolism, hepatotoxicity risk (WHO MEC Cat 4)",
     ALL_PILLS),

    # ── Cat 3: blocked per conservative policy ────────────────────────────
    ("cond_hypertension",
     _C3, 3,
     "Hypertension — elevated CVD risk with estrogen (WHO MEC Cat 3)",
     ALL_COMBINED),

    ("cond_gallstones",
     _C3, 3,
     "Gallstone disease — estrogen ↑ biliary cholesterol saturation (WHO MEC Cat 3)",
     ALL_COMBINED),

    ("cond_diabetes",
     _C3, 3,
     "Diabetes — vascular risk compounded by estrogen (WHO MEC Cat 3)",
     ALL_COMBINED),

    ("cond_chronic_kidney_disease",
     _C3, 3,
     "CKD — fluid retention and hypertension risk with estrogen (WHO MEC Cat 3)",
     ALL_COMBINED),

    ("cond_epilepsy",
     _C3, 3,
     "Epilepsy / AED interaction — enzyme-inducing AEDs reduce OCP efficacy "
     "(WHO MEC Cat 3 interaction)",
     ALL_COMBINED),
]


# ──────────────────────────────────────────────────────────────────────────────
#  Human-readable labels for auto-profile naming
#  Ordered by clinical severity (highest-priority names come first)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_LABEL_MAP = OrderedDict([
    ("cond_breast_cancer",          "Breast Cancer"),
    ("cond_thrombophilia",          "Thrombophilia"),
    ("cond_dvt",                    "DVT/VTE History"),
    ("cond_stroke",                 "Stroke History"),
    ("cond_mi",                     "MI History"),
    ("cond_migraine_with_aura",     "Migraine+Aura"),
    ("cond_lupus",                  "Lupus/SLE"),
    ("cond_liver_disease",          "Liver Disease"),
    ("cond_atrial_fibrillation",    "Afib"),
    ("cond_hypertension",           "Hypertension"),
    ("cond_diabetes",               "Diabetes"),
    ("cond_epilepsy",               "Epilepsy"),
    ("cond_chronic_kidney_disease", "CKD"),
    ("cond_gallstones",             "Gallstones"),
    ("cond_pcos",                   "PCOS"),
    ("cond_endometriosis",          "Endometriosis"),
    ("cond_depression",             "Depression"),
    ("cond_migraine",               "Migraine"),
    ("cond_sleep_apnea",            "Sleep Apnea"),
    ("cond_hypothyroidism",         "Hypothyroidism"),
    ("cond_rheumatoid_arthritis",   "Rheum. Arthritis"),
    ("cond_fibromyalgia",           "Fibromyalgia"),
    ("cond_osteoporosis",           "Osteoporosis"),
    ("obs_smoker",                  "Smoker"),
])


# ──────────────────────────────────────────────────────────────────────────────
#  derive_blocking_rules
# ──────────────────────────────────────────────────────────────────────────────
def derive_blocking_rules(centroid: dict) -> dict:
    """
    Apply WHO MEC rules to an unscaled cluster centroid.

    Returns
    -------
    dict : {combo_id: [reason_string, ...]}
        All pills that should be blocked for this profile, with reasons.
    """
    blocked: dict = {}

    # ── Standard single-feature WHO MEC rules ─────────────────────────────
    for feat, thresh, mec_cat, reason, combos in WHO_MEC_RULES:
        if feat in centroid and centroid[feat] >= thresh:
            for combo in combos:
                blocked.setdefault(combo, [])
                blocked[combo].append(
                    f"WHO MEC Cat {mec_cat}: {reason} "
                    f"[profile prevalence: {centroid[feat]:.1%}]"
                )

    # ── Compound rule: smoker + mean age ≥35 → Cat 3/4 ───────────────────
    smoker = centroid.get("obs_smoker", 0.0)
    age    = centroid.get("age", 0.0)
    if smoker >= _C3 and age >= 35.0:
        for combo in ALL_COMBINED:
            blocked.setdefault(combo, [])
            blocked[combo].append(
                f"WHO MEC Cat 3-4: Smoking + age ≥35 — thrombotic risk "
                f"[smoker rate: {smoker:.1%}, mean age: {age:.1f}y]"
            )

    # ── Compound rule: mean systolic BP ≥140 mmHg → Cat 3 ────────────────
    sbp = centroid.get("obs_systolic_bp", 0.0)
    if sbp >= 140.0:
        for combo in ALL_COMBINED:
            blocked.setdefault(combo, [])
            blocked[combo].append(
                f"WHO MEC Cat 3: Elevated cluster blood pressure "
                f"[mean systolic: {sbp:.0f} mmHg]"
            )

    return blocked


# ──────────────────────────────────────────────────────────────────────────────
#  auto_label
# ──────────────────────────────────────────────────────────────────────────────
def auto_label(cluster_id: int, centroid: dict, pop_means: dict, n_pts: int) -> str:
    """
    Generate a readable profile label by finding binary features that are
    elevated ≥1.5× the population mean (and ≥ 0.10 absolute).
    Top 2 most-elevated features are joined into the label.
    Continuous features contribute to a "Low-Risk / Healthy" fallback.
    """
    elevated = []
    for feat, readable in FEATURE_LABEL_MAP.items():
        if feat not in centroid:
            continue
        val = centroid[feat]
        pop = pop_means.get(feat, 0.0)
        if val >= max(pop * 1.5, 0.10):
            elevated.append((readable, val - pop))

    elevated.sort(key=lambda x: -x[1])
    tags = [r for r, _ in elevated[:2]]

    if not tags:
        return f"Profile {cluster_id}: Baseline / Low-Risk"
    return f"Profile {cluster_id}: {' + '.join(tags)}"


# ──────────────────────────────────────────────────────────────────────────────
#  Main training function
# ──────────────────────────────────────────────────────────────────────────────
def train(
    patients_csv: str = PATIENTS_CSV,
    k_min:        int   = 3,
    k_max:        int   = 12,
    threshold:    float = 0.40,
    seed:         int   = 42,
    plot_bic:     bool  = True,
) -> tuple:

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR,    exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 1 — Load patient data")
    print(f"{'─'*62}")
    df = pd.read_csv(patients_csv)
    print(f"  {len(df):,} patients × {len(df.columns)} columns")

    # ── 2. Stratified 80/20 train/test split ─────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 2 — Stratified 80/20 train/test split")
    print(f"{'─'*62}")
    strat_col = "has_absolute_contraindication_combined_ocp"
    strat = (
        df[strat_col].fillna(0).astype(int)
        if strat_col in df.columns
        else pd.Series(0, index=df.index)
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(sss.split(df, strat))
    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    np.savetxt(
        os.path.join(SPLITS_DIR, "train_ids.txt"),
        train_df["patient_id"].values, fmt="%s"
    )
    np.savetxt(
        os.path.join(SPLITS_DIR, "test_ids.txt"),
        test_df["patient_id"].values, fmt="%s"
    )

    train_abs_rate = train_df[strat_col].mean() if strat_col in train_df.columns else 0
    test_abs_rate  = test_df[strat_col].mean()  if strat_col in test_df.columns  else 0
    print(f"  Train : {len(train_df):,}  |  Test : {len(test_df):,}")
    print(f"  Abs. contraindication — Train: {train_abs_rate:.1%}  Test: {test_abs_rate:.1%}")
    print(f"  Saved → data/splits/train_ids.txt  /  test_ids.txt")

    # ── 3. Build feature matrix ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 3 — Build feature matrix")
    print(f"{'─'*62}")
    avail       = [f for f in ALL_FEATURES if f in train_df.columns]
    missing     = [f for f in ALL_FEATURES if f not in train_df.columns]
    avail_cont  = [f for f in CONTINUOUS_FEATURES if f in avail]
    avail_bin   = [f for f in BINARY_FEATURES     if f in avail]

    if missing:
        print(f"  [WARN] Not found in dataset (will skip): {missing}")
    print(f"  Features used: {len(avail)}  "
          f"({len(avail_cont)} continuous, {len(avail_bin)} binary)")

    # ── 4. Impute missing values ──────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 4 — Impute missing values")
    print(f"{'─'*62}")
    X_raw     = train_df[avail].copy()
    miss_pct  = X_raw.isnull().mean() * 100
    if miss_pct.max() > 0:
        for feat, pct in miss_pct[miss_pct > 0].items():
            print(f"    {feat:<35} {pct:.1f}% missing")
    else:
        print("  No missing values in training set.")

    # Continuous → median imputation (training-set medians preserved for inference)
    imp_cont = SimpleImputer(strategy="median")
    X_cont   = pd.DataFrame(
        imp_cont.fit_transform(X_raw[avail_cont]),
        columns=avail_cont
    )

    # Binary → 0 (conservative: absence of reported condition = not present)
    imp_bin  = SimpleImputer(strategy="constant", fill_value=0)
    X_bin    = pd.DataFrame(
        imp_bin.fit_transform(X_raw[avail_bin]),
        columns=avail_bin
    )

    X_train = pd.concat([X_cont, X_bin], axis=1)[avail]

    # ── 5. Scale continuous features ──────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 5 — Scale continuous features (StandardScaler)")
    print(f"{'─'*62}")
    scaler  = StandardScaler()
    X_scaled = X_train.copy()
    X_scaled[avail_cont] = scaler.fit_transform(X_train[avail_cont])
    X_matrix = X_scaled.values

    # Population means (unscaled) needed for profile labelling
    pop_means = X_train.mean().to_dict()

    print("  Continuous feature stats (training set):")
    for feat in avail_cont:
        mn = X_train[feat].mean()
        sd = X_train[feat].std()
        print(f"    {feat:<25}  mean={mn:7.2f}  std={sd:6.2f}")

    # ── 6. BIC model selection ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  STEP 6 — BIC model selection  (k = {k_min} … {k_max})")
    print(f"{'─'*62}")
    bic_scores = []
    aic_scores = []
    k_range    = list(range(k_min, k_max + 1))

    for k in k_range:
        gmm_k = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=5,
            max_iter=300,
        )
        gmm_k.fit(X_matrix)
        bic = gmm_k.bic(X_matrix)
        aic = gmm_k.aic(X_matrix)
        bic_scores.append(bic)
        aic_scores.append(aic)
        print(f"  k={k:2d}   BIC={bic:>14,.1f}   AIC={aic:>14,.1f}")

    k_best = k_range[int(np.argmin(bic_scores))]
    print(f"\n  ✓ Best k by BIC: {k_best}")

    # Save BIC table
    bic_df = pd.DataFrame({"k": k_range, "bic": bic_scores, "aic": aic_scores})
    bic_df.to_csv(os.path.join(ARTIFACTS_DIR, "bic_curve.csv"), index=False)

    # Plot
    if plot_bic and HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(bic_df["k"], bic_df["bic"], "bo-", label="BIC")
        ax.plot(bic_df["k"], bic_df["aic"], "rs--", label="AIC", alpha=0.7)
        ax.axvline(k_best, color="green", linestyle=":", linewidth=1.8,
                   label=f"Best k = {k_best}")
        ax.set_xlabel("Number of profiles (k)")
        ax.set_ylabel("Score  (lower = better)")
        ax.set_title("GMM Profile Selection — BIC & AIC")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(ARTIFACTS_DIR, "bic_curve.png"), dpi=150)
        print(f"  BIC plot saved → artifacts/bic_curve.png")
    elif plot_bic and not HAS_MPL:
        print("  [INFO] matplotlib not installed — BIC plot skipped")

    # ── 7. Fit final GMM ──────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  STEP 7 — Fit final GMM  (k={k_best}, covariance=diag, n_init=10)")
    print(f"{'─'*62}")
    gmm = GaussianMixture(
        n_components=k_best,
        covariance_type="diag",
        random_state=seed,
        n_init=10,
        max_iter=500,
    )
    gmm.fit(X_matrix)

    hard_labels   = gmm.predict(X_matrix)
    probs         = gmm.predict_proba(X_matrix)
    cluster_sizes = pd.Series(hard_labels).value_counts().sort_index()

    print(f"  Converged : {gmm.converged_}   Iterations: {gmm.n_iter_}")
    print(f"  Cluster sizes (hard assignment): {dict(cluster_sizes)}")
    print(f"  Mean max-probability (confidence): "
          f"{probs.max(axis=1).mean():.3f}")

    # ── 8. Recover unscaled centroids ─────────────────────────────────────
    # GMM means are in the scaled space. Invert only the continuous part.
    means_scaled          = gmm.means_                             # (k, n_feats)
    means_cont_unscaled   = scaler.inverse_transform(
        means_scaled[:, :len(avail_cont)]
    )
    centroids = np.hstack([means_cont_unscaled, means_scaled[:, len(avail_cont):]])

    # ── 9. Derive profile rules ───────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 8 — Derive WHO MEC blocking rules per profile")
    print(f"{'─'*62}")

    profiles = {}
    for k_idx in range(k_best):
        centroid = dict(zip(avail, centroids[k_idx].tolist()))
        n_pts    = int(cluster_sizes.get(k_idx, 0))
        blocked  = derive_blocking_rules(centroid)
        label    = auto_label(k_idx, centroid, pop_means, n_pts)

        # Top elevated binary features vs population (for JSON readability)
        elevated_feats = []
        for feat in avail_bin:
            val = centroid.get(feat, 0.0)
            pop = pop_means.get(feat, 0.0)
            if val >= max(pop * 1.5, 0.05):
                elevated_feats.append({
                    "feature":       feat,
                    "label":         FEATURE_LABEL_MAP.get(feat, feat),
                    "cluster_mean":  round(val, 4),
                    "pop_mean":      round(pop, 4),
                    "elevation":     round(val - pop, 4),
                })
        elevated_feats.sort(key=lambda x: -x["elevation"])

        profiles[str(k_idx)] = {
            "label":                  label,
            "n_patients_train":       n_pts,
            "pct_train":              round(n_pts / len(train_df) * 100, 1),
            "blocked_combos":         {c: r for c, r in blocked.items()},
            "all_blocked_ids":        sorted(blocked.keys()),
            "top_elevated_features":  elevated_feats[:8],
            "centroid":               {f: round(float(v), 4) for f, v in centroid.items()},
        }

        # ── Console summary ───────────────────────────────────────────────
        pct = n_pts / len(train_df) * 100
        print(f"\n  [{k_idx}] {label}  ({n_pts} pts, {pct:.0f}%)")
        if elevated_feats:
            top3 = ",  ".join(
                f"{e['label']} ({e['cluster_mean']:.0%})"
                for e in elevated_feats[:3]
            )
            print(f"       Elevated : {top3}")
        blocked_ids = sorted(blocked.keys())
        print(f"       Blocked  : {blocked_ids if blocked_ids else ['—  none']}")

    # ── 10. Save all artefacts ────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print("  STEP 9 — Save artefacts")
    print(f"{'─'*62}")

    profile_rules = {
        "k":                    k_best,
        "inference_threshold":  threshold,
        "centroid_thresh_cat4": _C4,
        "centroid_thresh_cat3": _C3,
        "feature_order":        avail,
        "continuous_features":  avail_cont,
        "binary_features":      avail_bin,
        "all_pill_ids":         ALL_PILLS,
        "all_combined_ids":     ALL_COMBINED,
        "profiles":             profiles,
    }

    _write_pkl(gmm,            "gmm_model.pkl")
    _write_pkl(scaler,         "scaler.pkl")
    _write_pkl(
        {"continuous": imp_cont, "binary": imp_bin,
         "avail_cont": avail_cont, "avail_bin": avail_bin},
        "imputer.pkl"
    )
    _write_json(avail,         "feature_order.json")
    _write_json(profile_rules, "profile_rules.json")

    print(f"\n  All artefacts → {ARTIFACTS_DIR}")
    print(f"  Patient splits → {SPLITS_DIR}")
    print(f"\n{'═'*62}")
    print(f"  DONE — {k_best} profiles fitted")
    print(f"  Inference threshold : P > {threshold:.0%}  →  apply profile rules")
    print(f"{'═'*62}\n")

    return gmm, profile_rules


# ──────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def _write_pkl(obj, filename: str) -> None:
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ {filename}")


def _write_json(obj, filename: str) -> None:
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  ✓ {filename}")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GMM patient risk profiles + WHO MEC blocking rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patients",  default=PATIENTS_CSV,
                        help="Path to patients_flat.csv")
    parser.add_argument("--k-min",     type=int,   default=3,
                        help="Minimum k for BIC search")
    parser.add_argument("--k-max",     type=int,   default=12,
                        help="Maximum k for BIC search")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Inference P threshold for applying profile rules")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip BIC curve plot")
    args = parser.parse_args()

    train(
        patients_csv=args.patients,
        k_min=args.k_min,
        k_max=args.k_max,
        threshold=args.threshold,
        seed=args.seed,
        plot_bic=not args.no_plot,
    )


if __name__ == "__main__":
    main()
