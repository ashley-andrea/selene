"""
Safe Gate Engine — deterministic medical rule module.

Applies hard constraints and relative risk rules based on patient data and
cluster profile. This module is the ONLY place where contraindicated pills
are filtered. The agent never sees excluded pills.

This is NOT a microservice — it runs inside the FastAPI backend to avoid
unnecessary infrastructure complexity and latency.

Rules here are sourced from validated medical guidelines. They are immutable
and not influenced by LLM reasoning.

FILTERING LAYERS:
1. Hard constraints — patient-specific contraindications (age, pathologies, etc.)
2. Cluster exclusions — population-level risk patterns based on cluster assignment
3. Relative risk rules — soft weighting factors for remaining candidates

PATIENT DATA FORMAT:
    Supports two input layouts (both produced by patient_extractor.py):
    • NEW (preferred): cond_* binary flags + obs_* numeric vitals
    • OLD (fallback):  pathologies[], habits[], medical_history[] string lists
    The helper _has_condition() transparently handles both.
"""

import logging

from agent.cluster_exclusions import get_excluded_pills_for_cluster
from agent.pill_database import get_all_pills, get_pill_ids

logger = logging.getLogger(__name__)


# ── Dual-format helper ──────────────────────────────────────────────────────

def _has_condition(patient_data: dict, cond_key: str, *list_values: str) -> bool:
    """
    Check whether a patient has a specific condition.

    Priority:
      1. cond_* binary field (new format from patient_extractor):
         e.g. _has_condition(pd, "cond_dvt")  → pd["cond_dvt"] == 1
      2. String-list fields (old format: pathologies / habits / medical_history):
         Additional list_values are checked across all three lists.

    Usage:
        _has_condition(pd, "cond_dvt", "dvt", "deep_vein_thrombosis")
        _has_condition(pd, "cond_breast_cancer", "breast_cancer")
    """
    # New format — authoritative if key is present
    val = patient_data.get(cond_key)
    if val is not None:
        return bool(val)

    # Old format fallback — search all list fields
    if list_values:
        all_items = (
            set(patient_data.get("pathologies", []))
            | set(patient_data.get("habits", []))
            | set(patient_data.get("medical_history", []))
        )
        return bool(all_items & set(list_values))

    return False


# ── Hard Constraint Rules ───────────────────────────────────────────────────
# Each rule is a function: (patient_data, pill_record) -> bool
# Returns True if the pill is CONTRAINDICATED (should be excluded).

def _is_combined_pill(pill: dict) -> bool:
    """
    Determine if a pill is a combined oral contraceptive.
    Uses the pill_type column from pill_reference_db.csv:
      combined_monophasic, combined_mono_triphasic  → combined
      progestin_only                                → NOT combined
    """
    pill_type = pill.get("pill_type", "")
    return str(pill_type).startswith("combined")


def _contraindicated_combined_vte(patient_data: dict, pill: dict) -> bool:
    """Combined pills are contraindicated with VTE history (DVT, stroke, PE)."""
    return _is_combined_pill(pill) and (
        _has_condition(patient_data, "cond_dvt", "dvt", "deep_vein_thrombosis")
        or _has_condition(patient_data, "cond_stroke", "stroke")
        or _has_condition(patient_data, "cond_mi", "pulmonary_embolism", "pe")
    )


def _contraindicated_combined_smoking_over_35(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated for smokers over 35 (WHO MEC Cat 3/4)."""
    age = patient_data.get("age", 0)
    smoker = bool(patient_data.get("obs_smoker")) or "smoking" in set(patient_data.get("habits", []))
    return _is_combined_pill(pill) and age > 35 and smoker


def _contraindicated_combined_migraines_with_aura(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with migraine with aura (WHO MEC Cat 4)."""
    return _is_combined_pill(pill) and _has_condition(
        patient_data, "cond_migraine_with_aura",
        "migraine_with_aura", "migraines_with_aura", "migraines_aura",
    )


def _contraindicated_combined_breast_cancer(patient_data: dict, pill: dict) -> bool:
    """All hormonal pills contraindicated with history of breast cancer (WHO MEC Cat 4)."""
    return _has_condition(patient_data, "cond_breast_cancer", "breast_cancer")


def _contraindicated_combined_liver_disease(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with active liver disease (WHO MEC Cat 3/4)."""
    return _is_combined_pill(pill) and _has_condition(
        patient_data, "cond_liver_disease", "liver_disease", "hepatitis", "cirrhosis",
    )


def _contraindicated_combined_lupus(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with lupus (WHO MEC Cat 3/4)."""
    return _is_combined_pill(pill) and _has_condition(
        patient_data, "cond_lupus", "lupus", "sle", "systemic_lupus_erythematosus",
    )


def _contraindicated_high_vte_hypertension(patient_data: dict, pill: dict) -> bool:
    """
    High VTE-risk combined pills contraindicated with hypertension (WHO MEC Cat 3).
    pill_reference_db.csv vte_risk_class: 'high' = 3rd/4th gen progestins (DSG, DRP, gestodene).
    """
    if not _is_combined_pill(pill):
        return False
    has_hypertension = _has_condition(
        patient_data, "cond_hypertension", "hypertension", "high_blood_pressure",
    )
    if not has_hypertension:
        return False
    vte_class = str(pill.get("vte_risk_class", "")).lower()
    return vte_class == "high"


# Registry of all hard constraint rules
HARD_CONSTRAINT_RULES = [
    _contraindicated_combined_vte,
    _contraindicated_combined_smoking_over_35,
    _contraindicated_combined_migraines_with_aura,
    _contraindicated_combined_breast_cancer,
    _contraindicated_combined_liver_disease,
    _contraindicated_combined_lupus,
    _contraindicated_high_vte_hypertension,
]


# ── Relative Risk Rules ────────────────────────────────────────────────────
# These are soft rules: they add weighting factors but don't exclude pills.
# Each returns a dict {pill_id: risk_modifier} for relevant pills.

def generate_relative_risk_rules(patient_data: dict, cluster_profile: str) -> list[dict]:
    """
    Generate relative risk rules based on patient data and cluster profile.
    Returns a list of rule dicts for informational/logging purposes.

    Supports both the new cond_*/obs_* format and the old pathologies/habits lists.
    """
    rules = []

    age = patient_data.get("age", 25)
    smoker = bool(patient_data.get("obs_smoker")) or "smoking" in set(patient_data.get("habits", []))

    # Hypertension: note about estrogen risk
    if _has_condition(patient_data, "cond_hypertension", "hypertension", "high_blood_pressure"):
        rules.append({
            "rule_name": "hypertension_estrogen_penalty",
            "description": "Higher estrogen doses carry elevated risk with hypertension",
            "modifier": -0.15,
        })

    # Obesity: effectiveness concern (use obs_bmi if available)
    bmi = patient_data.get("obs_bmi")
    is_obese = (bmi is not None and bmi >= 30) or (
        "obesity" in set(patient_data.get("pathologies", []))
    )
    if is_obese:
        rules.append({
            "rule_name": "obesity_effectiveness_concern",
            "description": "Obesity (BMI ≥ 30) may reduce effectiveness of lower-dose pills",
            "modifier": -0.10,
        })

    # Smoking (under 35): mild penalty for combined pills
    if smoker and age <= 35:
        rules.append({
            "rule_name": "smoking_combined_penalty",
            "description": "Smoking increases VTE risk with combined pills",
            "modifier": -0.10,
        })

    # Depression: progestin preference
    if _has_condition(patient_data, "cond_depression", "depression"):
        rules.append({
            "rule_name": "depression_progestin_preference",
            "description": "Some progestins may worsen mood — prefer levonorgestrel or norethisterone",
            "modifier": -0.08,
        })

    # Epilepsy: interaction concern
    if _has_condition(patient_data, "cond_epilepsy", "epilepsy"):
        rules.append({
            "rule_name": "epilepsy_interaction_concern",
            "description": "Enzyme-inducing AEDs reduce pill effectiveness — prefer higher doses",
            "modifier": -0.12,
        })

    return rules


# ── Main Entry Point ────────────────────────────────────────────────────────

def apply_safe_gate(patient_data: dict, cluster_profile: str) -> dict:
    """
    Apply the Safe Gate Engine to filter contraindicated pills and generate
    relative risk rules.

    Filtering is applied in two stages:
    1. Hard constraints — patient-specific contraindications
    2. Cluster exclusions — population-level risk patterns

    Returns:
        {
            "candidate_pool": [pill_id, ...],   # Allowed pills only
            "relative_risk_rules": [...]         # Soft weighting rules
        }

    The excluded pills (hard_constraints + cluster_exclusions) are computed
    internally but NEVER returned to the agent. The agent only sees candidate_pool.
    """
    all_pills_df = get_all_pills()
    excluded_ids = set()

    # ── STAGE 1: Apply hard constraint rules ───────────────────────────────
    for _, pill_row in all_pills_df.iterrows():
        pill = pill_row.to_dict()
        for rule_fn in HARD_CONSTRAINT_RULES:
            if rule_fn(patient_data, pill):
                excluded_ids.add(pill["combo_id"])
                break  # One contraindication is enough to exclude

    logger.info(
        "Hard constraints: excluded %d pills",
        len(excluded_ids),
    )

    # ── STAGE 2: Apply cluster-specific exclusions ─────────────────────────
    cluster_excluded = get_excluded_pills_for_cluster(cluster_profile)
    excluded_ids.update(cluster_excluded)

    logger.info(
        "Cluster exclusions: excluded %d additional pills for cluster '%s'",
        len(cluster_excluded),
        cluster_profile if isinstance(cluster_profile, str) else cluster_profile.get("profile", "?"),
    )

    # ── Build the allowed candidate pool ───────────────────────────────────
    all_ids = set(get_pill_ids())
    candidate_pool = sorted(all_ids - excluded_ids)

    logger.info(
        "Safe Gate: %d/%d pills allowed, %d excluded (hard=%d, cluster=%d)",
        len(candidate_pool),
        len(all_ids),
        len(excluded_ids),
        len(excluded_ids) - len(cluster_excluded),
        len(cluster_excluded),
    )

    # Generate relative risk rules (soft weighting)
    relative_risk_rules = generate_relative_risk_rules(patient_data, cluster_profile)

    return {
        "candidate_pool": candidate_pool,
        "relative_risk_rules": relative_risk_rules,
    }
