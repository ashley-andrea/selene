"""
Mock replacement for agent/tools/simulator_api.py.

Returns deterministic fake simulation results per pill ID.
Determinism is critical: the same pill always returns the same numbers,
so the utility optimizer always picks the same winner, making tests
repeatable and assertions reliable.

REMOVE WHEN: Real Simulator Model is deployed on Red Hat OpenShift.
"""

import random

# Hardcoded results for known test pills.
# pill_a is intentionally the best candidate so tests can assert on it.
FAKE_SIMULATION_RESULTS = {
    "pill_a": {
        "discontinuation_probability": 0.10,
        "severe_event_probability": 0.005,
        "mild_side_effect_score": 0.20,
        "contraceptive_effectiveness": 0.98,
    },
    "pill_b": {
        "discontinuation_probability": 0.18,
        "severe_event_probability": 0.009,
        "mild_side_effect_score": 0.35,
        "contraceptive_effectiveness": 0.97,
    },
    "pill_c": {
        "discontinuation_probability": 0.27,
        "severe_event_probability": 0.014,
        "mild_side_effect_score": 0.55,
        "contraceptive_effectiveness": 0.95,
    },
    "pill_d_high_risk": {
        "discontinuation_probability": 0.40,
        "severe_event_probability": 0.030,
        "mild_side_effect_score": 0.70,
        "contraceptive_effectiveness": 0.93,
    },
}


async def call_simulator(candidate_id: str, patient_data: dict) -> dict:
    """
    Returns deterministic simulation results for a candidate pill.
    Known pills return hardcoded values; unknown pills use seeded RNG.
    """
    if candidate_id in FAKE_SIMULATION_RESULTS:
        return FAKE_SIMULATION_RESULTS[candidate_id]

    # Unknown pill: deterministic noise based on pill name
    # Same pill ID always produces the same result across test runs
    seed = sum(ord(c) for c in candidate_id)
    rng = random.Random(seed)
    return {
        "discontinuation_probability": round(rng.uniform(0.05, 0.40), 3),
        "severe_event_probability": round(rng.uniform(0.001, 0.025), 4),
        "mild_side_effect_score": round(rng.uniform(0.10, 0.80), 2),
        "contraceptive_effectiveness": round(rng.uniform(0.90, 0.99), 3),
    }
