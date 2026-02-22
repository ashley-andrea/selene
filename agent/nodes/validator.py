"""
Validator Node — validates and normalizes patient input data.

This is the entry point of the graph. It ensures the patient_data object
is structurally correct, fields are properly typed, and string arrays are
normalized to lowercase. Initializes all loop control fields.
"""

import logging

from agent.state import SystemState

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["age"]
AGE_MIN = 15
AGE_MAX = 60  # GMM training data covers up to ~60; raised from 55


def run(state: SystemState) -> dict:
    """Validates and normalizes patient_data. Initializes loop state."""
    patient = state.get("patient_data")
    if not patient or not isinstance(patient, dict):
        raise ValueError("patient_data is required and must be a dict")

    # Check required fields (age is the only mandatory one now;
    # pathologies/habits/medical_history default to [] if absent since
    # the new cond_*/obs_* format is now the primary input)
    for field in REQUIRED_FIELDS:
        if field not in patient:
            raise ValueError(f"Missing required patient field: {field}")

    # Ensure list fields exist (default to []) for backwards compat
    patient.setdefault("pathologies", [])
    patient.setdefault("habits", [])
    patient.setdefault("medical_history", [])

    # Normalize and validate age
    try:
        patient["age"] = int(patient["age"])
    except (TypeError, ValueError):
        raise ValueError(f"patient.age must be an integer, got: {patient['age']}")

    if not (AGE_MIN <= patient["age"] <= AGE_MAX):
        raise ValueError(
            f"patient.age must be between {AGE_MIN} and {AGE_MAX}, got: {patient['age']}"
        )

    # Normalize string arrays to lowercase
    patient["pathologies"] = [p.lower().strip() for p in patient.get("pathologies", [])]
    patient["habits"] = [h.lower().strip() for h in patient.get("habits", [])]
    patient["medical_history"] = [m.lower().strip() for m in patient.get("medical_history", [])]

    logger.info(
        "Validated patient: age=%d, pathologies=%s, habits=%s, history=%s",
        patient["age"],
        patient["pathologies"],
        patient["habits"],
        patient["medical_history"],
    )

    return {
        "patient_data": patient,
        "iteration": 0,
        "converged": False,
        "cluster_profile": None,
        "cluster_confidence": None,
        "relative_risk_rules": [],
        "candidate_pool": [],
        "simulated_results": {},
        "utility_scores": {},
        "best_candidate": None,
        "previous_best_utility": None,
        "reason_codes": [],
        "top3_reason_codes": None,
    }
