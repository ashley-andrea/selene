"""
Candidate Generation Node — queries the pill database and applies the Safe Gate.

This node applies all hard constraints via the Safe Gate Engine to produce
the filtered candidate_pool. The agent never sees excluded pills.
Relative risk rules (soft weighting) are also generated here.
"""

import logging

from agent.safe_gate import apply_safe_gate
from agent.state import SystemState

logger = logging.getLogger(__name__)


def run(state: SystemState) -> dict:
    """
    Applies the Safe Gate Engine and generates the candidate pool.
    Hard constraints are enforced deterministically — the LLM has no input here.
    """
    patient_data = state["patient_data"]
    cluster_profile = state.get("cluster_profile", "")

    # Resolve cluster profile string (may be enriched dict from low-confidence path)
    profile_str = cluster_profile
    if isinstance(cluster_profile, dict):
        profile_str = cluster_profile.get("profile", "")

    # Apply Safe Gate — deterministic filtering
    gate_result = apply_safe_gate(patient_data, profile_str)

    candidate_pool = gate_result["candidate_pool"]
    relative_risk_rules = gate_result["relative_risk_rules"]

    if not candidate_pool:
        logger.error("Safe Gate excluded ALL pills — no candidates available")
        raise ValueError(
            "No candidate pills available after applying safety constraints. "
            "Patient profile may have too many contraindications for any oral contraceptive."
        )

    logger.info(
        "Candidate pool: %d pills — %s",
        len(candidate_pool),
        candidate_pool,
    )

    return {
        "candidate_pool": candidate_pool,
        "relative_risk_rules": relative_risk_rules,
    }
