"""
flatten_patients.py
===================
Post-processes Synthea's CSV output into a flat feature matrix
suitable for training the Cluster Model and Simulator Model.

Input (Synthea CSV export folder):
  - patients.csv       -> demographics, income, education, race, ethnicity
  - conditions.csv     -> diagnoses (pivoted into binary flags per patient)
  - observations.csv   -> vitals: BMI, blood pressure, PHQ-9 score, pain score
  - medications.csv    -> contraceptive history

Output:
  - patients_flat.csv  -> one row per patient, all features as columns

Usage:
    python flatten_patients.py --input ./output/data/output/patients/csv/2026_02_21T23_00_38Z --output ./patients_flat.csv

Requirements:
    pip install pandas numpy
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import date

# ─────────────────────────────────────────────
#  SNOMED-CT codes for condition flags
#  Each entry: (attribute_column_name, snomed_code, description)
# ─────────────────────────────────────────────
CONDITION_FLAGS = [
    # ── Absolute contraindications for combined OCP (estrogen) ──────────
    ("cond_migraine_with_aura",     "230462002",  "Migraine with aura"),
    ("cond_stroke",                 "230690007",  "Cerebrovascular accident (stroke)"),
    ("cond_mi",                     "22298006",   "Myocardial infarction"),
    ("cond_dvt",                    "128053003",  "Deep vein thrombosis"),
    ("cond_breast_cancer",          "254837009",  "Breast cancer"),
    ("cond_lupus",                  "200936003",  "Systemic lupus erythematosus"),
    ("cond_thrombophilia",          "441519001",  "Hereditary thrombophilia"),
    ("cond_atrial_fibrillation",    "49436004",   "Atrial fibrillation"),
    ("cond_liver_disease",          "235856003",  "Liver disease"),

    # ── Relative contraindications ───────────────────────────────────────
    ("cond_hypertension",           "59621000",   "Essential hypertension"),
    ("cond_migraine",               "37796009",   "Migraine (without aura)"),
    ("cond_gallstones",             "235919008",  "Gallstones"),
    ("cond_diabetes",               "44054006",   "Diabetes mellitus type 2"),
    ("cond_prediabetes",            "15777000",   "Prediabetes"),
    ("cond_epilepsy",               "84757009",   "Epilepsy"),  # AED interactions
    ("cond_chronic_kidney_disease", "709044004",  "Chronic kidney disease"),
    ("cond_sleep_apnea",            "73430006",   "Sleep apnea syndrome"),

    # ── OCP indications (conditions it can help) ─────────────────────────
    ("cond_pcos",                   "237055002",  "Polycystic ovarian syndrome"),
    ("cond_endometriosis",          "129103003",  "Endometriosis"),

    # ── Comorbidities relevant to side-effect profile ────────────────────
    ("cond_depression",             "370143000",  "Major depressive disorder"),
    ("cond_hypothyroidism",         "40930008",   "Hypothyroidism"),
    ("cond_rheumatoid_arthritis",   "69896004",   "Rheumatoid arthritis"),
    ("cond_fibromyalgia",           "57676002",   "Fibromyalgia"),
    ("cond_osteoporosis",           "64859006",   "Osteoporosis"),
    ("cond_asthma",                 "195967001",  "Asthma"),
]

# ─────────────────────────────────────────────
#  RxNorm / SNOMED codes for contraceptive medications
# ─────────────────────────────────────────────
OCP_MEDICATION_CODES = {
    # Combined (ethinyl estradiol-based)
    "Ethinyl Estradiol / Levonorgestrel":  "749762",
    "Ethinyl Estradiol / Norgestimate":    "238015",
    "Ethinyl Estradiol / Desogestrel":     "748468",
    "Ethinyl Estradiol / Drospirenone":    "748879",
    "Ethinyl Estradiol / Norethindrone":   "238005",
    # Progestin-only
    "Norethindrone (Minipill)":            "789980",
}

# ─────────────────────────────────────────────
#  LOINC codes for observations
# ─────────────────────────────────────────────
LOINC_BMI               = "39156-5"
LOINC_SYSTOLIC_BP       = "8480-6"
LOINC_DIASTOLIC_BP      = "8462-4"
LOINC_PHQ9              = "44261-6"
LOINC_PAIN_SCORE        = "72514-3"
LOINC_SMOKING_STATUS    = "72166-2"
LOINC_TESTOSTERONE      = "2986-8"


def load_csv(folder: str, filename: str) -> pd.DataFrame:
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  [WARN] {filename} not found in {folder}, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def compute_age(birthdate_str: str, reference_date: date = None) -> float:
    """Return age in years from a birthdate string (YYYY-MM-DD)."""
    if reference_date is None:
        reference_date = date.today()
    try:
        bd = date.fromisoformat(birthdate_str)
        return (reference_date - bd).days / 365.25
    except Exception:
        return np.nan


def build_condition_flags(conditions_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot conditions into one binary column per SNOMED code, per patient."""
    if conditions_df.empty:
        return pd.DataFrame()

    rows = []
    patient_ids = conditions_df["PATIENT"].unique()

    # Build a lookup: patient -> set of active SNOMED codes
    # (STOP is empty string or NaN for active conditions)
    active = conditions_df[
        conditions_df["STOP"].isna() | (conditions_df["STOP"].str.strip() == "")
    ].copy()
    patient_codes = active.groupby("PATIENT")["CODE"].apply(set).to_dict()

    for pid in patient_ids:
        row = {"PATIENT": pid}
        codes_for_patient = patient_codes.get(pid, set())
        for col_name, snomed_code, _ in CONDITION_FLAGS:
            row[col_name] = 1 if snomed_code in codes_for_patient else 0
        rows.append(row)

    return pd.DataFrame(rows)


def build_observation_features(observations_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most recent value of key observations per patient."""
    if observations_df.empty:
        return pd.DataFrame()

    obs = observations_df.copy()
    obs["VALUE_NUM"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs["DATE_DT"] = pd.to_datetime(obs["DATE"], errors="coerce")

    # Sort by date descending so we take the MOST RECENT value
    obs = obs.sort_values("DATE_DT", ascending=False)

    feature_map = {
        "obs_bmi":           LOINC_BMI,
        "obs_systolic_bp":   LOINC_SYSTOLIC_BP,
        "obs_diastolic_bp":  LOINC_DIASTOLIC_BP,
        "obs_phq9_score":    LOINC_PHQ9,
        "obs_pain_score":    LOINC_PAIN_SCORE,
        "obs_testosterone":  LOINC_TESTOSTERONE,
    }

    result_frames = []
    for feature_col, loinc_code in feature_map.items():
        subset = obs[obs["CODE"] == loinc_code][["PATIENT", "VALUE_NUM"]].dropna()
        latest = subset.groupby("PATIENT")["VALUE_NUM"].first().reset_index()
        latest.rename(columns={"VALUE_NUM": feature_col}, inplace=True)
        result_frames.append(latest)

    # Smoking status (categorical LOINC 72166-2)
    # Values: "Never smoker", "Current every day smoker", "Former smoker", etc.
    smoking = obs[obs["CODE"] == LOINC_SMOKING_STATUS][["PATIENT", "VALUE"]].copy()
    smoking_latest = smoking.groupby("PATIENT")["VALUE"].first().reset_index()
    smoking_latest.rename(columns={"VALUE": "obs_smoking_status"}, inplace=True)
    smoking_latest["obs_smoker"] = smoking_latest["obs_smoking_status"].apply(
        lambda v: 1 if isinstance(v, str) and "current" in v.lower() else 0
    )
    result_frames.append(smoking_latest[["PATIENT", "obs_smoker"]])

    # Merge all observation features
    merged = result_frames[0]
    for df in result_frames[1:]:
        merged = merged.merge(df, on="PATIENT", how="outer")

    return merged


def build_medication_features(medications_df: pd.DataFrame) -> pd.DataFrame:
    """Flags for contraceptive medication history."""
    if medications_df.empty:
        return pd.DataFrame()

    meds = medications_df.copy()

    # Current active OCP (no stop date)
    active_meds = meds[
        meds["STOP"].isna() | (meds["STOP"].str.strip() == "")
    ]

    result_rows = []
    for pid, group in meds.groupby("PATIENT"):
        codes = group["CODE"].tolist()
        descriptions = " ".join(group["DESCRIPTION"].fillna("").str.lower().tolist())

        row = {"PATIENT": pid}
        # Ever used any OCP
        row["med_ever_ocp"] = 1 if any(
            c in OCP_MEDICATION_CODES.values() for c in codes
        ) or "contraceptive" in descriptions or "norethindrone" in descriptions else 0

        # Currently on combined OCP
        active_codes = active_meds[active_meds["PATIENT"] == pid]["CODE"].tolist()
        row["med_current_combined_ocp"] = 1 if any(
            c in [v for k, v in OCP_MEDICATION_CODES.items() if "Minipill" not in k]
            for c in active_codes
        ) else 0

        # Currently on progestin-only
        row["med_current_minipill"] = 1 if (
            "789980" in active_codes or
            "minipill" in " ".join(active_meds[active_meds["PATIENT"] == pid]["DESCRIPTION"].fillna("").str.lower().tolist())
        ) else 0

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def flatten(input_folder: str, output_path: str):
    print(f"\n Loading CSV files from: {input_folder}")

    patients_df    = load_csv(input_folder, "patients.csv")
    conditions_df  = load_csv(input_folder, "conditions.csv")
    observations_df = load_csv(input_folder, "observations.csv")
    medications_df = load_csv(input_folder, "medications.csv")

    if patients_df.empty:
        print("ERROR: patients.csv is empty or missing. Aborting.")
        sys.exit(1)

    print(f"  patients.csv:     {len(patients_df)} rows")
    print(f"  conditions.csv:   {len(conditions_df)} rows")
    print(f"  observations.csv: {len(observations_df)} rows")
    print(f"  medications.csv:  {len(medications_df)} rows")

    # ── 1. Base demographics from patients.csv ───────────────────────────
    print("\n Building base patient features...")
    # Only retain patient_id from the raw patients table.
    # birthdate → used only to compute age; gender → used only to filter to females.
    # race, ethnicity, marital_status, income, healthcare_expenses/coverage are
    # not clinically relevant to OCP selection and are not retained.
    id_col = "ID" if "ID" in patients_df.columns else "patient_id"
    base_df = patients_df[[id_col]].copy()
    base_df.rename(columns={id_col: "patient_id"}, inplace=True)

    # Compute age from BIRTHDATE (not retained in output)
    if "BIRTHDATE" in patients_df.columns:
        base_df["age"] = patients_df["BIRTHDATE"].apply(compute_age)

    # Filter to females (GENDER not retained in output)
    if "GENDER" in patients_df.columns:
        before = len(base_df)
        female_ids = set(patients_df[patients_df["GENDER"].str.upper() == "F"][id_col].tolist())
        base_df = base_df[base_df["patient_id"].isin(female_ids)]
        print(f"  Kept {len(base_df)}/{before} female patients")

    # ── 2. Condition flags ───────────────────────────────────────────────
    print(" Building condition flags...")
    if not conditions_df.empty:
        cond_flags_df = build_condition_flags(conditions_df)
        cond_flags_df.rename(columns={"PATIENT": "patient_id"}, inplace=True)
    else:
        cond_flags_df = pd.DataFrame({"patient_id": base_df["patient_id"]})

    # ── 3. Observation features ──────────────────────────────────────────
    print(" Extracting observation features...")
    if not observations_df.empty:
        obs_features_df = build_observation_features(observations_df)
        obs_features_df.rename(columns={"PATIENT": "patient_id"}, inplace=True)
    else:
        obs_features_df = pd.DataFrame({"patient_id": base_df["patient_id"]})

    # ── 4. Medication features ───────────────────────────────────────────
    print(" Building medication features...")
    if not medications_df.empty:
        med_features_df = build_medication_features(medications_df)
        med_features_df.rename(columns={"PATIENT": "patient_id"}, inplace=True)
    else:
        med_features_df = pd.DataFrame({"patient_id": base_df["patient_id"]})

    # ── 5. Merge everything ──────────────────────────────────────────────
    print(" Merging all features...")
    flat = base_df.copy()
    for df in [cond_flags_df, obs_features_df, med_features_df]:
        if not df.empty and "patient_id" in df.columns:
            flat = flat.merge(df, on="patient_id", how="left")

    # Fill binary condition flags with 0 (not having a condition is valid)
    condition_cols = [c for c, _, _ in CONDITION_FLAGS]
    for col in condition_cols:
        if col in flat.columns:
            flat[col] = flat[col].fillna(0).astype(int)

    # Derived safety flags (for quick filtering / model labeling)
    absolute_contraindication_cols = [
        "cond_migraine_with_aura", "cond_stroke", "cond_mi", "cond_dvt",
        "cond_breast_cancer", "cond_lupus", "cond_thrombophilia",
        "cond_atrial_fibrillation",
    ]
    available_abs_cols = [c for c in absolute_contraindication_cols if c in flat.columns]
    if available_abs_cols:
        flat["has_absolute_contraindication_combined_ocp"] = flat[available_abs_cols].max(axis=1)

    # ── 6. Summary ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    flat.to_csv(output_path, index=False)

    print(f"\n ✓ Flat dataset written to: {output_path}")
    print(f"   Shape: {flat.shape[0]} patients × {flat.shape[1]} features")
    print(f"\n Feature overview:")
    print(f"   Demographics   : age")
    print(f"   Condition flags: {len([c for c in flat.columns if c.startswith('cond_')])} binary flags")
    print(f"   Observations   : {len([c for c in flat.columns if c.startswith('obs_')])} numeric/categorical features")
    print(f"   Medications    : {len([c for c in flat.columns if c.startswith('med_')])} medication flags")
    print(f"\n Condition prevalence (% of population):")
    for col in [c for c, _, _ in CONDITION_FLAGS if c in flat.columns]:
        pct = flat[col].mean() * 100
        print(f"   {col:<40} {pct:.1f}%")

    if "has_absolute_contraindication_combined_ocp" in flat.columns:
        pct = flat["has_absolute_contraindication_combined_ocp"].mean() * 100
        print(f"\n   Patients with absolute OCP contraindication: {pct:.1f}%")

    return flat


def main():
    parser = argparse.ArgumentParser(
        description="Flatten Synthea CSV output into a patient feature matrix."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to Synthea CSV output folder (contains patients.csv, conditions.csv, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        default="patients_flat.csv",
        help="Output path for the flat feature CSV (default: patients_flat.csv)"
    )
    args = parser.parse_args()

    flat_df = flatten(args.input, args.output)
    return flat_df


if __name__ == "__main__":
    main()
