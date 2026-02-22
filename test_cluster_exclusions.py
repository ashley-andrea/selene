"""
Test script to demonstrate cluster-specific pill filtering.

This script shows how the Safe Gate Engine now filters pills based on both:
1. Hard constraints (patient-specific contraindications)
2. Cluster exclusions (population-level risk patterns)

Run this script to verify the cluster exclusion system is working correctly.
"""

import logging

from agent.cluster_exclusions import CLUSTER_EXCLUSIONS, add_cluster_exclusion
from agent.pill_database import get_pill_ids
from agent.safe_gate import apply_safe_gate

logging.basicConfig(level=logging.INFO, format="%(name)-30s | %(message)s")

print("=" * 80)
print("CLUSTER-SPECIFIC PILL FILTERING DEMONSTRATION")
print("=" * 80)

# Test 1: Default cluster (no exclusions)
print("\n1. Healthy patient, cluster_2 (no cluster-specific exclusions)")
print("-" * 80)
patient_healthy = {
    "age": 28,
    "pathologies": [],
    "habits": [],
    "medical_history": [],
}

result = apply_safe_gate(patient_healthy, "cluster_2")
print(f"   Candidate pool size: {len(result['candidate_pool'])}/{len(get_pill_ids())} pills")
print(f"   Allowed pills: {result['candidate_pool']}")

# Test 2: Add cluster exclusions and verify they're applied
print("\n2. Add exclusions to cluster_0 and test")
print("-" * 80)
add_cluster_exclusion("cluster_0", "pill_cyproterone_35")
add_cluster_exclusion("cluster_0", "pill_drospirenone_30")

result = apply_safe_gate(patient_healthy, "cluster_0")
print(f"   Candidate pool size: {len(result['candidate_pool'])}/{len(get_pill_ids())} pills")
print(f"   Excluded by cluster: pill_cyproterone_35, pill_drospirenone_30")
print(f"   Verify exclusions: pill_cyproterone_35 in pool? {('pill_cyproterone_35' in result['candidate_pool'])}")
print(f"   Verify exclusions: pill_drospirenone_30 in pool? {('pill_drospirenone_30' in result['candidate_pool'])}")

# Test 3: Combined hard constraints + cluster exclusions
print("\n3. High-risk patient (VTE history) in cluster_0")
print("-" * 80)
patient_high_risk = {
    "age": 35,
    "pathologies": [],
    "habits": [],
    "medical_history": ["dvt"],  # Hard constraint: excludes combined pills
}

result = apply_safe_gate(patient_high_risk, "cluster_0")
print(f"   Candidate pool size: {len(result['candidate_pool'])}/{len(get_pill_ids())} pills")
print(f"   Hard constraints excluded: all combined pills (VTE history)")
print(f"   Cluster exclusions excluded: pill_cyproterone_35, pill_drospirenone_30")
print(f"   Allowed pills: {result['candidate_pool']}")

# Test 4: Show current cluster exclusion configuration
print("\n4. Current cluster exclusion configuration")
print("-" * 80)
for cluster, exclusions in CLUSTER_EXCLUSIONS.items():
    if exclusions:
        print(f"   {cluster}: {len(exclusions)} pills excluded")
        print(f"      → {exclusions}")
    else:
        print(f"   {cluster}: no exclusions")

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
print("\nTo configure cluster exclusions, edit:")
print("   agent/cluster_exclusions.py → CLUSTER_EXCLUSIONS dict")
print("\nThe Safe Gate Engine will automatically apply these exclusions")
print("BEFORE the agent sees the candidate pool.")
print("=" * 80)
