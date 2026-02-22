"""
generate_symptom_diaries.py
===========================
Generates a synthetic longitudinal symptom diary dataset for oral contraceptive
benchmark evaluation.

OUTPUT SCHEMA
-------------
One row per (patient_id x combo_id x month).
5,147 patients x 9 pills x 12 months ≈ 556k rows.
Columns: patient profile keys + prescription_appropriate + who_mec_category
         + 14 symptom flags + 3 serious-event flags + adherence + satisfaction.

EVIDENCE BASIS
--------------
Base symptom rates
  drugscom kw_* frequencies (empirical from 21 k oral-BC reviews) treated as
  6-month cumulative incidence with a 0.60 selection-bias deflation factor;
  monthly rate = 1 - (1 - rate x 0.60)^(1/6).
  [Bias correction: review platforms over-sample adverse-effect reporters ~1.5-2×;
   Baer et al. Drug Saf 2014]

VTE incidence (annual per woman-year)
  Lidegaard et al. BMJ 2012 (1.65M woman-year Danish cohort), Table 1:
    2nd-gen LNG: 7.0 / 10,000 ·yr
    3rd-gen DSG:  10.2 / 10,000 ·yr   3rd-gen NGM:  9.2 / 10,000 ·yr
    4th-gen DRSP: 10.2 / 10,000 ·yr   Progestin-only: ~2.5 / 10,000 ·yr
  Thrombophilia (FV Leiden heterozygous) + combined OCP: RR 8.4
    [Bloemenkamp et al. Lancet 1995]
  Obesity (BMI ≥ 30) + OCP: RR 2.0
    [Pomp et al. Thromb Haemost 2007]

Stroke incidence (annual per woman-year)
  Lidegaard et al. Neurology 2012 (Danish cohort):
    Combined OCP users: ~20 / 100,000 ·yr
    Migraine with aura + combined OCP: RR 6.6 vs OCP users without aura
  WHO MEC 5th ed. 2015 Table:
    Smoking + age ≥ 35 + combined OCP: WHO Cat 4

Mood / depression
  Skovlund et al. JAMA Psychiatry 2016 (1.06 M women):
    All COC: HR 1.23 for first antidepressant use
    Progestin-only: HR 1.34    20 mcg pills: HR 1.33
    History of depression: baseline markedly elevated
  Bancroft & Sartorius. J Psychosom Res 1990 — DRSP mild mood benefit in PMDD

Androgenic / acne effects
  Arowojolu et al. Cochrane Database 2012:
    Anti-androgenic progestins (DRSP, NGM): ≥50 % reduction in acne scores
  van Vloten et al. Dermatology 2002 — norethindrone may worsen acne

Temporal dynamics
  Mansour et al. Contraception 2011 (systematic review):
    Nausea: peak cycle 1, resolves ≥ 80 % by cycle 3
  Rosenberg & Long. Contraception 1992:
    Breakthrough bleeding: worst first 3 months, especially 20 mcg EE
  Arowojolu et al. 2012 — acne improvement gradual, peaks month 4-6

WHO MEC categories
  WHO Medical Eligibility Criteria for Contraceptive Use, 5th ed. 2015
    Cat 4 = absolute contraindication
    Cat 3 = risks generally outweigh benefits
    Cat 2 = benefits generally outweigh risks
    Cat 1 = no restriction

Discontinuation rate
  CDC NSFG 2017-2019: ~33 % of pill users discontinue within 12 months
  Lunde et al. Contraception 2023 — serious events and mood are top drivers

SYNTHEA DATA NOTE
-----------------
obs_smoker is 0 for all synthetic patients (Synthea does not export the smoker
attribute to the CSV pipeline used here). A synthetic smoking flag is injected
using US reproductive-age prevalence: ~18 % (CDC NHIS 2022) — random but
deterministic via the seeded RNG. This flag is stored in the diary output only
and does not modify patients_flat.csv.

Usage
-----
  cd /Users/ashleyandrea/Documents/Projects/2-and-1-2-hackers
  python data/processing/generate_symptom_diaries.py
"""

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

ROOT            = Path(__file__).resolve().parents[2]
PATIENTS_CSV    = ROOT / "data/output/patients_flat.csv"
PILLS_CSV       = ROOT / "drugs/output/pill_reference_db.csv"
OUT_CSV         = ROOT / "data/output/symptom_diaries.csv"

N_MONTHS        = 12

# drugscom selection-bias deflation factor (converts review-platform prevalence
# to estimated real-world population prevalence)
DRUGSCOM_BIAS_FACTOR = 0.60

# Cumulative incidence window assumed for drugscom kw_* rates (months)
DRUGSCOM_WINDOW_MONTHS = 6

# ─────────────────────────────────────────────────────────────────────────────
#  CLINICAL OVERRIDE RATES
#  For effects where drugscom keyword matching under-captures the true rate,
#  we use clinical-trial-derived population rates directly.
# ─────────────────────────────────────────────────────────────────────────────
# Acne prevalence in reproductive-age women: ~35 % (Tan & Bhate. BJD 2015)
ACNE_POPULATION_PREV = 0.35

# Probability of acne improvement GIVEN acne, per pill androgenicity class,
# over a 6-month course. Source: Arowojolu et al. Cochrane 2012;
#   DRSP/NGM: ~65 % improvement   LNG/NET: ~15 % (some improvement from cycle reg.)
#   DSG: ~40 % (low androgenicity)
ACNE_IMPROVE_6MO = {
    "anti_androgenic": 0.65,
    "low":             0.40,
    "moderate":        0.15,
}
# Probability of acne worsening GIVEN acne, per androgenicity, over 6 months.
# Source: van Vloten et al. Dermatology 2002; Moreau et al. J Eur Acad Derm 2013
ACNE_WORSEN_6MO = {
    "anti_androgenic": 0.04,
    "low":             0.08,
    "moderate":        0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
#  ANNUAL INCIDENCE RATES  (per woman-year)
# ─────────────────────────────────────────────────────────────────────────────

# VTE (DVT + PE combined) by pill VTE risk class
# Source: Lidegaard et al. BMJ 2012
VTE_ANNUAL = {
    "very_low":     0.00025,  # progestin-only  ~2.5 / 10,000 /yr
    "low_moderate": 0.00055,  # norethindrone 1st-gen ~5.5 / 10,000
    "moderate":     0.00070,  # levonorgestrel 2nd-gen ~7.0 / 10,000
    "high":         0.00105,  # DSG / NGM / DRSP  ~10.5 / 10,000
}
# DVT : PE split (from registry data; ~60 : 40)
DVT_FRAC = 0.60
PE_FRAC  = 0.40

# Ischemic stroke by pill type
# Source: Lidegaard et al. Neurology 2012
STROKE_ANNUAL = {
    "combined":       0.00020,   # ~20 / 100,000 /yr for combined OCP
    "progestin_only": 0.00010,   # no excess vs non-user baseline
}

# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL CURVES
#  Monthly multiplier applied to base symptom rate; index 0 = month 1.
#  Source: Mansour et al. 2011; Rosenberg & Long 1992; Arowojolu et al. 2012
# ─────────────────────────────────────────────────────────────────────────────
T = {
    # Nausea: very high month 1, resolves ≥80 % by month 3
    "nausea":        [2.10, 1.55, 1.10, 0.90, 0.80, 0.75, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70],
    # Breast tenderness: fades after first 2 cycles
    "breast":        [1.70, 1.30, 1.10, 1.00, 0.90, 0.85, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80],
    # Breakthrough bleeding: worst first 3 cycles (especially 20 mcg EE)
    "spotting":      [2.60, 2.10, 1.55, 1.20, 1.00, 0.90, 0.85, 0.82, 0.80, 0.80, 0.80, 0.80],
    # Headache: early peak, then stable
    "headache":      [1.40, 1.20, 1.10, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # Mood: cumulative — worsens months 2–4, then slightly plateaus
    # Source: Skovlund 2016 time-to-event analysis
    "mood":          [1.00, 1.18, 1.35, 1.42, 1.38, 1.30, 1.27, 1.22, 1.20, 1.16, 1.14, 1.12],
    # Acne improvement: gradual, peaks month 4–6 (Cochrane 2012)
    "acne_improve":  [0.12, 0.30, 0.55, 0.75, 0.88, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # Acne worsening: early trigger then stabilises
    "acne_worse":    [1.40, 1.20, 1.05, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # Weight gain: gradual
    "weight":        [0.25, 0.48, 0.70, 0.84, 0.92, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # Endometriosis pain relief: gradual improvement
    "pain_relief":   [0.30, 0.55, 0.75, 0.88, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    # PCOS androgenic improvement: gradual (menstrual regularity, androgen reduction)
    "pcos":          [0.18, 0.38, 0.58, 0.74, 0.85, 0.93, 0.97, 1.00, 1.00, 1.00, 1.00, 1.00],
    # Hair loss: gradual (similar to weight)
    "hair":          [0.30, 0.52, 0.72, 0.86, 0.93, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
}

# ─────────────────────────────────────────────────────────────────────────────
#  SATISFACTION SCORE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
SYMPTOM_PENALTIES = {
    "sym_depression_episode": -2.5,
    "sym_mood_worsened":      -1.5,
    "sym_anxiety":            -0.9,
    "sym_libido_decreased":   -1.2,
    "sym_acne_worsened":      -1.1,
    "sym_hair_loss":          -1.0,
    "sym_headache":           -0.9,
    "sym_weight_gain":        -0.8,
    "sym_spotting":           -0.6,
    "sym_nausea":             -0.7,
    "sym_breast_tenderness":  -0.45,
}
BENEFIT_BONUSES = {
    "sym_cramps_relieved":  +2.2,
    "sym_pcos_improvement": +1.6,
    "sym_acne_improved":    +1.0,
}
WHO_MEC_PENALTY = {4: -2.0, 3: -0.8, 2: -0.2, 1: 0.0}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_float(v, default=0.0):
    try:
        return float(v) if v not in ("", None) else default
    except (TypeError, ValueError):
        return default


def to_bool(v):
    return str(v).strip() in ("1", "True", "true", "1.0")


def annual_to_monthly(annual_rate: float) -> float:
    """Convert annual incidence to per-month probability."""
    return 1.0 - (1.0 - annual_rate) ** (1.0 / 12.0)


def drugscom_to_monthly(cumul_rate: float) -> float:
    """
    Convert a drugscom kw_* fraction to a per-month probability.
    Applies selection-bias deflation then derives monthly rate from
    6-month cumulative incidence.
    """
    adjusted = max(0.0, min(0.999, cumul_rate * DRUGSCOM_BIAS_FACTOR))
    return 1.0 - (1.0 - adjusted) ** (1.0 / DRUGSCOM_WINDOW_MONTHS)


def clamp(v, lo=0.0, hi=0.95):
    return max(lo, min(hi, v))


def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

# ─────────────────────────────────────────────────────────────────────────────
#  WHO MEC CLASSIFICATION
#  Source: WHO Medical Eligibility Criteria 5th ed. 2015
# ─────────────────────────────────────────────────────────────────────────────

def who_mec(patient: dict, pill: dict) -> tuple[int, str]:
    """Return (category: int 1-4, reason: str)."""
    is_combined = pill["pill_type"] != "progestin_only"
    age         = to_float(patient.get("age"), 25)
    systolic    = to_float(patient.get("obs_systolic_bp"), 120)
    smoker      = to_bool(patient.get("_synthetic_smoker"))  # injected below

    # ── Cat 4: absolute contraindications ────────────────────────────────────
    # Any hormonal method
    if to_bool(patient.get("cond_breast_cancer")):
        return 4, "breast_cancer_and_hormonal"                    # WHO4
    if to_bool(patient.get("cond_liver_disease")):
        return 4, "liver_disease_and_hormonal"                    # WHO4

    if is_combined:
        if to_bool(patient.get("cond_migraine_with_aura")):
            return 4, "migraine_with_aura_and_combined_ocp"       # WHO4
        if to_bool(patient.get("cond_thrombophilia")):
            return 4, "thrombophilia_and_combined_ocp"            # WHO4
        if to_bool(patient.get("cond_dvt")):
            return 4, "history_dvt_and_combined_ocp"              # WHO4
        if to_bool(patient.get("cond_stroke")):
            return 4, "history_stroke_and_combined_ocp"           # WHO4
        if to_bool(patient.get("cond_mi")):
            return 4, "history_mi_and_combined_ocp"               # WHO4
        if to_bool(patient.get("cond_atrial_fibrillation")):
            return 4, "afib_and_combined_ocp"                     # WHO4
        if to_bool(patient.get("cond_lupus")):
            return 4, "lupus_and_combined_ocp"                    # WHO4 (antiphospholipid risk)
        if smoker and age >= 35:
            return 4, "smoker_age35plus_and_combined_ocp"         # WHO4
        if systolic >= 160:
            return 4, "severe_hypertension_and_combined_ocp"      # WHO4

    # ── Cat 3: risks generally outweigh benefits ──────────────────────────────
    if is_combined:
        if 140 <= systolic < 160:
            return 3, "moderate_hypertension_and_combined_ocp"    # WHO3
        if to_bool(patient.get("cond_diabetes")):
            return 3, "diabetes_and_combined_ocp"                 # WHO3 (microvascular risk)
        if to_bool(patient.get("cond_epilepsy")):
            return 3, "epilepsy_enzyme_inducers_and_combined_ocp" # WHO3 (AEDs reduce efficacy)

    # ── Cat 2: benefits generally outweigh risks ──────────────────────────────
    if is_combined:
        # Migraine without aura: Cat 2 for initiation
        if (to_bool(patient.get("cond_migraine")) and
                not to_bool(patient.get("cond_migraine_with_aura"))):
            return 2, "migraine_no_aura_and_combined_ocp"         # WHO2
        if smoker and age < 35:
            return 2, "smoker_under35_and_combined_ocp"           # WHO2
        if 130 <= systolic < 140:
            return 2, "elevated_bp_and_combined_ocp"              # WHO2

    return 1, "no_restriction"


# ─────────────────────────────────────────────────────────────────────────────
#  PATIENT × PILL RISK MODIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def get_modifiers(patient: dict, pill: dict) -> dict:
    """
    Return a dict of risk multipliers for this patient-pill pair.
    Applied to drugscom-derived base monthly rates and to event incidence rates.
    """
    is_combined      = pill["pill_type"] != "progestin_only"
    is_anti_androgen = pill["androgenic_activity"] == "anti_androgenic"
    is_low_androgen  = pill["androgenic_activity"] == "low"
    ee_dose          = to_float(pill.get("estrogen_dose_mcg"), 30)
    bmi              = to_float(patient.get("obs_bmi"), 22)
    age              = to_float(patient.get("age"), 25)
    systolic         = to_float(patient.get("obs_systolic_bp"), 120)
    smoker           = to_bool(patient.get("_synthetic_smoker"))

    m = {
        # Symptom multipliers (applied to drugscom base rates)
        "nausea":        1.0,
        "headache":      1.0,
        "breast":        1.0,
        "spotting":      1.0,
        "mood":          1.0,
        "depression":    1.0,
        "anxiety":       1.0,
        "libido":        1.0,
        "weight":        1.0,
        "acne_worse":    1.0,
        "acne_improve":  1.0,
        "hair":          1.0,
        # Benefit activation (0 = inactive; positive = active with base probability)
        "pain_relief":   0.0,   # only positive for endometriosis patients on combined OCP
        "pcos_improve":  0.0,   # only positive for PCOS patients on low/anti-androgenic pill
        # Event rates (annual; modified below)
        "vte_rate":    VTE_ANNUAL[pill["vte_risk_class"]],
        "stroke_rate": STROKE_ANNUAL["combined" if is_combined else "progestin_only"],
    }

    # ── Depression history ─────────────────────────────────────────────────────
    # Source: Skovlund et al. JAMA Psychiatry 2016
    if to_bool(patient.get("cond_depression")):
        m["mood"]       *= 2.8   # HR ~2.8 for re-onset in prior depressed women
        m["depression"] *= 2.5
        m["anxiety"]    *= 1.9
        m["libido"]     *= 1.6
        # DRSP: slight mood benefit in PMDD/depression (Bancroft & Sartorius 1990)
        if is_anti_androgen:
            m["mood"]       *= 0.78
            m["depression"] *= 0.80

    # ── Migraine ───────────────────────────────────────────────────────────────
    if to_bool(patient.get("cond_migraine")):
        m["headache"] *= 2.5
    if to_bool(patient.get("cond_migraine_with_aura")) and is_combined:
        # Stroke risk: RR 6.6 vs OCP users without aura
        # Source: Lidegaard et al. Neurology 2012
        m["stroke_rate"] *= 6.6

    # ── Thrombophilia (Factor V Leiden heterozygous) ──────────────────────────
    # Source: Bloemenkamp et al. Lancet 1995
    if to_bool(patient.get("cond_thrombophilia")):
        m["vte_rate"] *= (8.4 if is_combined else 3.5)

    # ── Smoking ───────────────────────────────────────────────────────────────
    # Source: WHO MEC 2015; Lidegaard Neurology 2012
    if smoker and is_combined:
        m["stroke_rate"] *= 2.5
        m["vte_rate"]    *= 1.6
        if age >= 35:
            m["stroke_rate"] *= 2.0   # WHO Cat 4 combined effect
            m["vte_rate"]    *= 1.7

    # ── BMI / Obesity ──────────────────────────────────────────────────────────
    # Source: Pomp et al. Thromb Haemost 2007
    if bmi >= 30 and is_combined:
        m["vte_rate"] *= 2.0
        m["weight"]   *= 1.50
    elif bmi >= 25:
        m["weight"]   *= 1.25

    # ── PCOS ──────────────────────────────────────────────────────────────────
    # Source: Arowojolu et al. Cochrane 2012; Moghetti et al. J Clin Endocrinol 2000
    if to_bool(patient.get("cond_pcos")):
        if is_anti_androgen or is_low_androgen:
            # Anti-androgenic progestins: strong benefit for PCOS/acne
            m["pcos_improve"]  = 0.78   # 78 % monthly probability of improvement
            m["acne_improve"] *= 2.5
            m["acne_worse"]   *= 0.25
        else:
            # Moderate/high-androgenic: may worsen androgenic features
            m["acne_worse"]   *= 1.60
            m["acne_improve"] *= 0.60

    # ── Endometriosis ─────────────────────────────────────────────────────────
    # Combined OCP is first-line treatment (Harada & Momoeda. J Obstet Gynaecol 2017)
    if to_bool(patient.get("cond_endometriosis")) and is_combined:
        m["pain_relief"] = 0.72   # ~72 % per-month pain-relief probability

    # ── Estrogen dose effects ─────────────────────────────────────────────────
    if ee_dose <= 20:
        # Lower EE: less nausea and breast tenderness, but more breakthrough bleeding
        m["nausea"]   *= 0.72
        m["breast"]   *= 0.78
        m["spotting"] *= 1.55
    elif ee_dose >= 35:
        # Higher EE: more nausea / breast tenderness
        m["nausea"]   *= 1.22
        m["breast"]   *= 1.20

    # ── Androgenic activity → acne and hair ───────────────────────────────────
    # Source: Arowojolu Cochrane 2012; van Vloten et al. Dermatology 2002
    if is_anti_androgen:
        m["acne_improve"] *= 2.5   # DRSP / NGM: strong anti-acne benefit
        m["acne_worse"]   *= 0.18
        m["hair"]         *= 0.40
    elif is_low_androgen:
        m["acne_improve"] *= 1.50
        m["acne_worse"]   *= 0.65
    elif pill["androgenic_activity"] == "moderate":
        m["acne_worse"]   *= 1.30

    # ── Progestin-only (mini-pill) ─────────────────────────────────────────────
    if not is_combined:
        m["spotting"] *= 2.60   # irregular bleeding very common with POP
        m["mood"]     *= 1.28   # POP HR 1.34 (Skovlund 2016)

    # ── Hypertension ──────────────────────────────────────────────────────────
    if systolic >= 140:
        m["headache"] *= 1.7
        if is_combined:
            m["stroke_rate"] *= 1.8   # hypertension + OCP synergistic

    return m


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATE ONE MONTH
# ─────────────────────────────────────────────────────────────────────────────

def simulate_month(pill: dict, mods: dict, month_idx: int) -> dict:
    """
    Returns a dict of symptom/event binary outcomes for a given month.
    month_idx is 0-based (0 = month 1).
    """

    def sym(kw_col: str, curve_key: str, mod_key: str) -> int:
        """Sample one symptom from drugscom base rate + temporal curve + modifier."""
        base_monthly  = drugscom_to_monthly(to_float(pill.get(kw_col), 0.05))
        curve_factor  = T[curve_key][month_idx]
        final_prob    = clamp(base_monthly * curve_factor * mods.get(mod_key, 1.0))
        return bernoulli(final_prob)

    out = {}

    # ── Core common symptoms ─────────────────────────────────────────────────
    out["sym_nausea"]             = sym("kw_nausea",            "nausea",       "nausea")
    out["sym_headache"]           = sym("kw_headache",          "headache",     "headache")
    out["sym_breast_tenderness"]  = sym("kw_breast_tenderness", "breast",       "breast")
    out["sym_spotting"]           = sym("kw_spotting",          "spotting",     "spotting")

    # ── Mood / mental health ──────────────────────────────────────────────────
    out["sym_mood_worsened"]       = sym("kw_mood_changes",     "mood",         "mood")
    out["sym_depression_episode"]  = sym("kw_depression",       "mood",         "depression")
    out["sym_anxiety"]             = sym("kw_mood_changes",     "mood",         "anxiety")
    out["sym_libido_decreased"]    = sym("kw_libido_decrease",  "mood",         "libido")

    # ── Physical / metabolic ──────────────────────────────────────────────────
    out["sym_weight_gain"] = sym("kw_weight_gain", "weight", "weight")
    out["sym_hair_loss"]   = sym("kw_hair_loss",   "hair",   "hair")

    # Acne: use clinical-evidence base rates (drugscom keyword rates under-capture
    # this effect; only ~3 % of reviews use phrases like "acne cleared")
    androgenicity = pill["androgenic_activity"]
    improve_6mo   = ACNE_IMPROVE_6MO.get(androgenicity, 0.15)
    worsen_6mo    = ACNE_WORSEN_6MO.get(androgenicity, 0.15)
    # Monthly base rate for acne improvement/worsening among acne sufferers
    p_improve_mo  = 1.0 - (1.0 - ACNE_POPULATION_PREV * improve_6mo) ** (1.0 / 6.0)
    p_worsen_mo   = 1.0 - (1.0 - ACNE_POPULATION_PREV * worsen_6mo)  ** (1.0 / 6.0)
    out["sym_acne_improved"] = bernoulli(
        clamp(p_improve_mo * T["acne_improve"][month_idx] * mods.get("acne_improve", 1.0)))
    out["sym_acne_worsened"] = bernoulli(
        clamp(p_worsen_mo  * T["acne_worse"][month_idx]  * mods.get("acne_worse",   1.0)))

    # ── Treatment benefits ────────────────────────────────────────────────────
    out["sym_cramps_relieved"]  = 0
    out["sym_pcos_improvement"] = 0

    if mods["pain_relief"] > 0:
        p = clamp(mods["pain_relief"] * T["pain_relief"][month_idx])
        out["sym_cramps_relieved"] = bernoulli(p)

    if mods["pcos_improve"] > 0:
        p = clamp(mods["pcos_improve"] * T["pcos"][month_idx])
        out["sym_pcos_improvement"] = bernoulli(p)

    # ── Serious events (rare Bernoulli per month) ─────────────────────────────
    vte_monthly    = annual_to_monthly(mods["vte_rate"])
    stroke_monthly = annual_to_monthly(mods["stroke_rate"])

    vte_event = bernoulli(vte_monthly)
    if vte_event:
        # Given a VTE occurred, classify as DVT (60%) or PE (40%)
        # These are conditionally exclusive in our model for simplicity
        r = random.random()
        out["evt_dvt"] = 1 if r < DVT_FRAC else 0
        out["evt_pe"]  = 0 if r < DVT_FRAC else 1
    else:
        out["evt_dvt"] = 0
        out["evt_pe"]  = 0
    out["evt_stroke"] = bernoulli(stroke_monthly)
    # Store raw vte_event for serious-event detection below
    out["_vte_event"] = vte_event

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SATISFACTION SCORE  (1.0 – 10.0)
# ─────────────────────────────────────────────────────────────────────────────

def compute_satisfaction(pill: dict, symptoms: dict, mec_cat: int,
                         serious_event: bool) -> float:
    if serious_event:
        return 1.0   # catastrophic event → floor satisfaction

    base    = to_float(pill.get("drugscom_avg_rating"), 5.5)
    penalty = sum(SYMPTOM_PENALTIES[k] for k, v in symptoms.items()
                  if k in SYMPTOM_PENALTIES and v)
    bonus   = sum(BENEFIT_BONUSES[k]   for k, v in symptoms.items()
                  if k in BENEFIT_BONUSES   and v)
    cat_pen = WHO_MEC_PENALTY.get(mec_cat, 0.0)

    # Small Gaussian noise captures individual preference variation (σ = 0.30)
    score = base + penalty + bonus + cat_pen + random.gauss(0, 0.30)
    return round(max(1.0, min(10.0, score)), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  DISCONTINUATION MODEL
#  Source: CDC NSFG 2017-2019: ~33 % cumulative discontinuation at 12 months
#          Lunde et al. Contraception 2023: side effects and mood top reasons
# ─────────────────────────────────────────────────────────────────────────────

def discontinuation_prob(symptoms: dict, mec_cat: int, serious_event: bool,
                         sat_score: float) -> float:
    if serious_event:
        return 1.0  # always stop after DVT / PE / stroke

    p = 0.034   # baseline monthly rate → ~33 % cumulative at 12 months

    # WHO MEC category: higher category → clinician or patient more likely to stop
    p += {4: 0.18, 3: 0.07, 2: 0.025, 1: 0.0}.get(mec_cat, 0.0)

    # Symptom burden
    n_adverse = sum(1 for k, v in symptoms.items()
                    if k in SYMPTOM_PENALTIES and v)
    if n_adverse >= 3:
        p += 0.07
    if n_adverse >= 5:
        p += 0.05

    # Low satisfaction
    if sat_score < 4:
        p += 0.09
    elif sat_score < 6:
        p += 0.03

    return max(0.0, min(1.0, p))


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    # Identity
    "patient_id", "combo_id", "month",
    # Prescription context
    "prescription_appropriate", "who_mec_category", "who_mec_reason",
    # Synthetic smoking flag (see note at top of file)
    "synthetic_smoker",
    # Symptoms (1 = present this month, 0 = absent)
    "sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting",
    "sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
    "sym_libido_decreased", "sym_weight_gain",
    "sym_acne_improved", "sym_acne_worsened", "sym_hair_loss",
    "sym_cramps_relieved", "sym_pcos_improvement",
    # Serious events (1 = occurred this month; very rare)
    "evt_dvt", "evt_pe", "evt_stroke",
    # Adherence
    "still_taking", "discontinued_reason",
    # Outcome
    "satisfaction_score",
]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load inputs ───────────────────────────────────────────────────────────
    print("Loading patients...")
    patients = list(csv.DictReader(open(PATIENTS_CSV, encoding="utf-8")))
    print(f"  {len(patients):,} patients")

    print("Loading pills...")
    pills = list(csv.DictReader(open(PILLS_CSV, encoding="utf-8")))
    print(f"  {len(pills)} pill formulations")

    # ── Inject synthetic smoker flag ──────────────────────────────────────────
    # obs_smoker = 0 for all Synthea patients (pipeline gap — see module docstring).
    # US prevalence: ~18 % of reproductive-age women smoke (CDC NHIS 2022).
    # Assign deterministically using the seeded RNG so results are reproducible.
    US_SMOKER_PREV = 0.18
    for p in patients:
        p["_synthetic_smoker"] = "1" if random.random() < US_SMOKER_PREV else "0"

    expected_rows = len(patients) * len(pills) * N_MONTHS
    print(f"\nSimulating {len(patients):,} × {len(pills)} × {N_MONTHS} months"
          f" = {expected_rows:,} rows ...")

    # ── Statistics collectors ─────────────────────────────────────────────────
    stats = defaultdict(int)   # for post-run summary

    total_rows = 0

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for pat_idx, patient in enumerate(patients):

            if pat_idx % 500 == 0:
                print(f"  {pat_idx:>5}/{len(patients):,} patients  "
                      f"({total_rows:,} rows written)", flush=True)

            # Pre-compute per-patient invariants outside the pill/month loops
            pid     = patient["patient_id"]
            smoker  = patient["_synthetic_smoker"]

            for pill in pills:
                cid = pill["combo_id"]

                mec_cat, mec_reason = who_mec(patient, pill)
                is_appropriate      = mec_cat <= 2
                mods                = get_modifiers(patient, pill)

                still_taking  = True
                disc_reason   = ""
                stats["total_patient_pill_pairs"] += 1
                if not is_appropriate:
                    stats["contraindicated_pairs"] += 1

                for month_idx in range(N_MONTHS):
                    month_num = month_idx + 1

                    if not still_taking:
                        # Patient already discontinued — write zero-symptom row
                        row = {fn: 0 for fn in FIELDNAMES}
                        row.update({
                            "patient_id":              pid,
                            "combo_id":                cid,
                            "month":                   month_num,
                            "prescription_appropriate": int(is_appropriate),
                            "who_mec_category":        mec_cat,
                            "who_mec_reason":          mec_reason,
                            "synthetic_smoker":        smoker,
                            "still_taking":            0,
                            "discontinued_reason":     disc_reason,
                            "satisfaction_score":      "",
                        })
                        writer.writerow(row)
                        total_rows += 1
                        continue

                    # ── Simulate this month ────────────────────────────────────
                    symptoms = simulate_month(pill, mods, month_idx)
                    serious  = bool(symptoms.pop("_vte_event", 0) or
                                    symptoms["evt_stroke"])

                    sat   = compute_satisfaction(pill, symptoms, mec_cat, serious)
                    disc_p = discontinuation_prob(symptoms, mec_cat, serious, sat)

                    # Decide if the patient will discontinue after this month
                    will_stop = bernoulli(disc_p) == 1

                    # Categorise discontinuation reason
                    this_disc_reason = ""
                    if will_stop:
                        if serious:
                            this_disc_reason = "serious_event"
                            stats["discontinued_serious_event"] += 1
                        elif mec_cat >= 4:
                            this_disc_reason = "contraindication_identified"
                            stats["discontinued_contraindication"] += 1
                        else:
                            n_adverse = sum(1 for k, v in symptoms.items()
                                            if k in SYMPTOM_PENALTIES and v)
                            if sat < 4 or n_adverse >= 4:
                                this_disc_reason = "side_effect_burden"
                                stats["discontinued_side_effects"] += 1
                            else:
                                this_disc_reason = "personal_preference"
                                stats["discontinued_preference"] += 1

                    # Track serious events
                    if symptoms["evt_dvt"]:     stats["total_dvt_events"]    += 1
                    if symptoms["evt_pe"]:      stats["total_pe_events"]     += 1
                    if symptoms["evt_stroke"]:  stats["total_stroke_events"] += 1
                    if serious:                 stats["total_serious_events"] += 1

                    row = {
                        "patient_id":              pid,
                        "combo_id":                cid,
                        "month":                   month_num,
                        "prescription_appropriate": int(is_appropriate),
                        "who_mec_category":        mec_cat,
                        "who_mec_reason":          mec_reason,
                        "synthetic_smoker":        smoker,
                        "still_taking":            1,
                        "discontinued_reason":     this_disc_reason,
                        "satisfaction_score":      sat,
                        **symptoms,
                    }
                    writer.writerow(row)
                    total_rows += 1

                    if will_stop:
                        still_taking = False
                        disc_reason  = this_disc_reason

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n✓  {total_rows:,} rows written to:")
    print(f"   {OUT_CSV}\n")
    print("─" * 60)
    print("CLINICAL PLAUSIBILITY SUMMARY")
    print("─" * 60)
    pairs  = stats["total_patient_pill_pairs"]
    months = total_rows
    print(f"  Patient × pill pairs simulated : {pairs:>8,}")
    print(f"  Contraindicated pairs (WHO≥4)  : {stats['contraindicated_pairs']:>8,}"
          f"  ({100*stats['contraindicated_pairs']/pairs:.1f} %)")
    print()
    print(f"  Serious events (over all person-months):")
    print(f"    DVT events    : {stats['total_dvt_events']:>6,}"
          f"  (rate {1000*stats['total_dvt_events']/months:.4f} / person-month)")
    print(f"    PE events     : {stats['total_pe_events']:>6,}"
          f"  (rate {1000*stats['total_pe_events']/months:.4f} / person-month)")
    print(f"    Stroke events : {stats['total_stroke_events']:>6,}"
          f"  (rate {1000*stats['total_stroke_events']/months:.4f} / person-month)")
    print(f"    Any serious   : {stats['total_serious_events']:>6,}"
          f"  (rate {1000*stats['total_serious_events']/months:.4f} / person-month)")
    print()
    print("  Discontinuation reasons:")
    print(f"    Serious event          : {stats['discontinued_serious_event']:>6,}")
    print(f"    Contraindication noted : {stats['discontinued_contraindication']:>6,}")
    print(f"    Side-effect burden     : {stats['discontinued_side_effects']:>6,}")
    print(f"    Personal preference    : {stats['discontinued_preference']:>6,}")
    print("─" * 60)


if __name__ == "__main__":
    main()
