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
"""

import logging

from agent.cluster_exclusions import get_excluded_pills_for_cluster
from agent.pill_database import get_all_pills, get_pill_ids

logger = logging.getLogger(__name__)


# ── Hard Constraint Rules ───────────────────────────────────────────────────
# Each rule is a function: (patient_data, pill_record) -> bool
# Returns True if the pill is CONTRAINDICATED (should be excluded).

def _is_combined_pill(pill: dict) -> bool:
    """
    Determine if a pill is a combined oral contraceptive.
    Combined pills contain both estrogen and progestin.
    We detect this by checking if substance_name contains "ETHINYL ESTRADIOL".
    """
    substance_name = pill.get("substance_name", "")
    return "ETHINYL ESTRADIOL" in substance_name.upper()


def _contraindicated_combined_vte(patient_data: dict, pill: dict) -> bool:
    """Combined pills are contraindicated with VTE history."""
    vte_conditions = {"dvt", "stroke", "pulmonary_embolism"}
    history = set(patient_data.get("medical_history", []))
    return _is_combined_pill(pill) and bool(history & vte_conditions)


def _contraindicated_combined_smoking_over_35(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated for smokers over 35."""
    age = patient_data.get("age", 0)
    habits = set(patient_data.get("habits", []))
    return _is_combined_pill(pill) and age > 35 and "smoking" in habits


def _contraindicated_combined_migraines_with_aura(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with migraines (simplified: all migraines)."""
    pathologies = set(patient_data.get("pathologies", []))
    return _is_combined_pill(pill) and "migraines" in pathologies


def _contraindicated_combined_breast_cancer(patient_data: dict, pill: dict) -> bool:
    """All hormonal pills contraindicated with history of breast cancer."""
    history = set(patient_data.get("medical_history", []))
    return "breast_cancer" in history


def _contraindicated_combined_liver_disease(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with active liver disease."""
    history = set(patient_data.get("medical_history", []))
    return _is_combined_pill(pill) and "liver_disease" in history


def _contraindicated_combined_lupus(patient_data: dict, pill: dict) -> bool:
    """Combined pills contraindicated with lupus."""
    pathologies = set(patient_data.get("pathologies", []))
    return _is_combined_pill(pill) and "lupus" in pathologies


def _contraindicated_high_vte_hypertension(patient_data: dict, pill: dict) -> bool:
    """
    High VTE-risk pills contraindicated with hypertension.
    Note: Without explicit vte_risk_level in pills.csv, we can't apply this rule.
    For now, we'll skip this constraint until the ML team provides risk classifications.
    """
    pathologies = set(patient_data.get("pathologies", []))
    # Disabled until we have vte_risk_level data
    return False


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
    
    Note: The actual pills.csv does not contain structured fields like estrogen_dose,
    progestin type, etc. These rules are placeholders until the ML team provides
    structured pill metadata. For now, they serve as documentation of risk factors
    that should be considered.
    """
    rules = []
    pathologies = set(patient_data.get("pathologies", []))
    habits = set(patient_data.get("habits", []))
    age = patient_data.get("age", 25)

    # Hypertension: note about estrogen risk
    if "hypertension" in pathologies:
        rules.append({
            "rule_name": "hypertension_estrogen_penalty",
            "description": "Higher estrogen doses carry elevated risk with hypertension",
            "modifier": -0.15,
        })

    # Obesity: effectiveness concern
    if "obesity" in pathologies:
        rules.append({
            "rule_name": "obesity_effectiveness_concern",
            "description": "Obesity may reduce effectiveness of lower-dose pills",
            "modifier": -0.10,
        })

    # Smoking (under 35): mild penalty for combined pills
    if "smoking" in habits and age <= 35:
        rules.append({
            "rule_name": "smoking_combined_penalty",
            "description": "Smoking increases VTE risk with combined pills",
            "modifier": -0.10,
        })

    # Depression: progestin preference
    if "depression" in pathologies:
        rules.append({
            "rule_name": "depression_progestin_preference",
            "description": "Some progestins may worsen mood — prefer levonorgestrel or norethisterone",
            "modifier": -0.08,
        })

    # Epilepsy: interaction concern
    if "epilepsy" in pathologies:
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
                excluded_ids.add(pill["set_id"])
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
