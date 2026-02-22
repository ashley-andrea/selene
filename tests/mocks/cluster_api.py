"""
Mock replacement for agent/tools/cluster_api.py.

Returns hardcoded cluster assignments based on patient age.
Used in unit tests via pytest's patch decorator to avoid any HTTP calls.

REMOVE WHEN: Real Cluster Model is deployed on Red Hat OpenShift.
"""


def call_cluster_model(patient_data: dict) -> dict:
    """
    Returns a realistic cluster assignment based on patient age.
    Simulates the behavior of the real Cluster Model API.

    Note: The 45+ age group deliberately returns confidence below 0.70
    to exercise the low-confidence LLM weight adjustment path.
    """
    age = patient_data.get("age", 25)

    if age < 25:
        return {"cluster_profile": "cluster_0", "cluster_confidence": 0.91}
    elif age < 35:
        return {"cluster_profile": "cluster_2", "cluster_confidence": 0.88}
    elif age < 45:
        return {"cluster_profile": "cluster_3", "cluster_confidence": 0.76}
    else:
        return {"cluster_profile": "cluster_4", "cluster_confidence": 0.65}
