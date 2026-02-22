"""
Cluster-Specific Pill Exclusions — defines which pills are contraindicated
for each patient cluster.

This module is loaded by the Safe Gate Engine to filter pills based on the
cluster assignment. Each cluster has a list of pill_ids that should be
excluded before the agent sees the candidate pool.

CONFIGURATION:
Edit the CLUSTER_EXCLUSIONS dictionary below to define cluster-specific rules.
The keys are cluster profile names (e.g. "cluster_0", "cluster_1").
Each value is a list of pill_ids that should be excluded for that cluster.

These exclusions are applied IN ADDITION to the hard constraint rules in
safe_gate.py. The complete filtering logic is:
1. Apply hard constraints (patient-specific contraindications)
2. Apply cluster exclusions (population-level risk patterns)
3. Return filtered candidate pool to the agent

Example use cases for cluster exclusions:
- Cluster of patients with hormonal sensitivity → exclude high-dose pills
- Cluster with cardiovascular risk factors → exclude high VTE-risk pills
- Cluster with mood disorder patterns → exclude certain progestins
- Cluster with metabolic concerns → exclude specific combinations
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Cluster Exclusion Configuration ────────────────────────────────────────
#
# Define which pills should NOT be shown to patients in each cluster.
# This is based on population-level patterns learned by the clustering model.
#
# Format:
# {
#     "cluster_name": ["pill_id_1", "pill_id_2", ...],
# }
#
# Example:
# {
#     "cluster_0": ["pill_cyproterone_35", "pill_drospirenone_30"],
#     "cluster_1": ["pill_gestodene_30", "pill_gestodene_20"],
# }
#
CLUSTER_EXCLUSIONS: Dict[str, List[str]] = {
    # ── CLUSTER 0 ───────────────────────────────────────────────────────────
    # Example: High cardiovascular risk profile
    # Exclude high-dose estrogen and high VTE-risk pills
    "cluster_0": [
        # "pill_cyproterone_35",    # High VTE risk
        # "pill_norethisterone_35",  # High estrogen dose
        # "pill_norgestimate_35",    # High estrogen dose
    ],
    
    # ── CLUSTER 1 ───────────────────────────────────────────────────────────
    # Example: Hormonal sensitivity / mood concerns
    # Exclude certain progestins known to affect mood
    "cluster_1": [
        # "pill_drospirenone_30",
        # "pill_drospirenone_20",
        # "pill_cyproterone_35",
    ],
    
    # ── CLUSTER 2 ───────────────────────────────────────────────────────────
    # Example: Young, healthy, no significant risk factors
    # Few exclusions (most pills appropriate)
    "cluster_2": [
        # No cluster-specific exclusions for low-risk group
    ],
    
    # ── CLUSTER 3 ───────────────────────────────────────────────────────────
    # Example: History of side effects with 3rd/4th gen progestins
    # Prefer 2nd generation or progestin-only options
    "cluster_3": [
        # "pill_desogestrel_30",
        # "pill_desogestrel_20",
        # "pill_gestodene_30",
        # "pill_gestodene_20",
        # "pill_drospirenone_30",
        # "pill_drospirenone_20",
    ],
    
    # ── CLUSTER 4 ───────────────────────────────────────────────────────────
    # Example: Metabolic syndrome / PCOS pattern
    # Exclude pills that may worsen insulin resistance
    "cluster_4": [
        # "pill_levonorgestrel_30",
        # "pill_levonorgestrel_20",
    ],
    
    # ── DEFAULT ─────────────────────────────────────────────────────────────
    # Used when cluster is unknown or not configured
    # Conservative: no cluster-specific exclusions (rely on hard constraints only)
    "default": [],
}


def get_excluded_pills_for_cluster(cluster_profile: str) -> List[str]:
    """
    Return the list of pill_ids that should be excluded for the given cluster.
    Falls back to empty list if cluster not found (no cluster-specific exclusions).
    
    Args:
        cluster_profile: Cluster identifier (e.g. "cluster_0", "cluster_1")
    
    Returns:
        List of pill_ids to exclude
    """
    # Handle low-confidence cluster assignments (dict format)
    if isinstance(cluster_profile, dict):
        profile_str = cluster_profile.get("profile", "default")
    else:
        profile_str = cluster_profile or "default"
    
    excluded = CLUSTER_EXCLUSIONS.get(profile_str, CLUSTER_EXCLUSIONS["default"])
    
    if excluded:
        logger.info(
            "Cluster '%s': excluding %d pills — %s", 
            profile_str, len(excluded), excluded
        )
    else:
        logger.info("Cluster '%s': no cluster-specific exclusions", profile_str)
    
    return excluded


def add_cluster_exclusion(cluster_profile: str, pill_id: str) -> None:
    """
    Dynamically add a pill exclusion for a cluster (useful for testing or runtime config).
    
    Args:
        cluster_profile: Cluster identifier
        pill_id: Pill ID to exclude
    """
    if cluster_profile not in CLUSTER_EXCLUSIONS:
        CLUSTER_EXCLUSIONS[cluster_profile] = []
    
    if pill_id not in CLUSTER_EXCLUSIONS[cluster_profile]:
        CLUSTER_EXCLUSIONS[cluster_profile].append(pill_id)
        logger.info("Added exclusion: cluster '%s' → pill '%s'", cluster_profile, pill_id)


def get_all_cluster_exclusions() -> Dict[str, List[str]]:
    """Return the complete cluster exclusion configuration (for debugging/inspection)."""
    return CLUSTER_EXCLUSIONS.copy()
