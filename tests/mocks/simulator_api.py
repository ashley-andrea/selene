"""
Mock replacement for agent/tools/simulator_api.py.

Returns deterministic fake simulation results per pill ID.
Determinism is critical: the same pill always returns the same numbers,
so the utility optimizer always picks the same winner, making tests
repeatable and assertions reliable.

DEPRECATED: The real Simulator Model (models/simulation/serve.py) is now
deployed on Red Hat OpenShift. These mocks are used ONLY by unit tests
that need fully deterministic, network-free simulation results.

The response shape now mirrors the real model response — 4 backward-compatible
summary metrics PLUS full 12-month trajectory fields (symptom_probs, satisfaction).
"""

import random

# 12-month trajectory helpers for deterministic test data
_FLAT_PROBS = lambda p, n=12: [round(p, 4)] * n  # noqa: E731
_INCR_PROBS = lambda start, end, n=12: [  # noqa: E731
    round(start + (end - start) * i / max(n - 1, 1), 4) for i in range(n)
]

# Hardcoded results for known test pills.
# pill_a is intentionally the best candidate so tests can assert on it.
FAKE_SIMULATION_RESULTS = {
    "pill_a": {
        # Summary metrics — consumed by utility node
        "discontinuation_probability": 0.10,
        "severe_event_probability": 0.005,
        "mild_side_effect_score": 0.20,
        "contraceptive_effectiveness": 0.98,
        # Full trajectory — mirrors real model response shape
        "combo_id": "pill_a",
        "n_months": 12,
        "months": list(range(1, 13)),
        "symptom_probs": {
            "sym_nausea":             _FLAT_PROBS(0.05),
            "sym_headache":           _FLAT_PROBS(0.04),
            "sym_breast_tenderness":  _FLAT_PROBS(0.03),
            "sym_spotting":           _FLAT_PROBS(0.02),
            "sym_mood_worsened":      _FLAT_PROBS(0.04),
            "sym_depression_episode": _FLAT_PROBS(0.01),
            "sym_anxiety":            _FLAT_PROBS(0.03),
            "sym_libido_decreased":   _FLAT_PROBS(0.04),
            "sym_weight_gain":        _FLAT_PROBS(0.03),
            "sym_acne_improved":      _FLAT_PROBS(0.30),
            "sym_acne_worsened":      _FLAT_PROBS(0.02),
            "sym_hair_loss":          _FLAT_PROBS(0.01),
            "sym_cramps_relieved":    _FLAT_PROBS(0.40),
            "sym_pcos_improvement":   _FLAT_PROBS(0.10),
            "evt_dvt":                _FLAT_PROBS(0.0005),
            "evt_pe":                 _FLAT_PROBS(0.0003),
            "evt_stroke":             _FLAT_PROBS(0.0002),
            "still_taking":           _INCR_PROBS(0.97, 0.90),
        },
        "satisfaction": _INCR_PROBS(7.5, 8.0),
    },
    "pill_b": {
        "discontinuation_probability": 0.18,
        "severe_event_probability": 0.009,
        "mild_side_effect_score": 0.35,
        "contraceptive_effectiveness": 0.97,
        "combo_id": "pill_b",
        "n_months": 12,
        "months": list(range(1, 13)),
        "symptom_probs": {
            "sym_nausea":             _FLAT_PROBS(0.12),
            "sym_headache":           _FLAT_PROBS(0.10),
            "sym_breast_tenderness":  _FLAT_PROBS(0.08),
            "sym_spotting":           _FLAT_PROBS(0.07),
            "sym_mood_worsened":      _FLAT_PROBS(0.09),
            "sym_depression_episode": _FLAT_PROBS(0.03),
            "sym_anxiety":            _FLAT_PROBS(0.06),
            "sym_libido_decreased":   _FLAT_PROBS(0.08),
            "sym_weight_gain":        _FLAT_PROBS(0.07),
            "sym_acne_improved":      _FLAT_PROBS(0.20),
            "sym_acne_worsened":      _FLAT_PROBS(0.06),
            "sym_hair_loss":          _FLAT_PROBS(0.03),
            "sym_cramps_relieved":    _FLAT_PROBS(0.30),
            "sym_pcos_improvement":   _FLAT_PROBS(0.07),
            "evt_dvt":                _FLAT_PROBS(0.0009),
            "evt_pe":                 _FLAT_PROBS(0.0006),
            "evt_stroke":             _FLAT_PROBS(0.0004),
            "still_taking":           _INCR_PROBS(0.95, 0.82),
        },
        "satisfaction": _INCR_PROBS(6.5, 7.0),
    },
    "pill_c": {
        "discontinuation_probability": 0.27,
        "severe_event_probability": 0.014,
        "mild_side_effect_score": 0.55,
        "contraceptive_effectiveness": 0.95,
        "combo_id": "pill_c",
        "n_months": 12,
        "months": list(range(1, 13)),
        "symptom_probs": {
            "sym_nausea":             _FLAT_PROBS(0.20),
            "sym_headache":           _FLAT_PROBS(0.18),
            "sym_breast_tenderness":  _FLAT_PROBS(0.15),
            "sym_spotting":           _FLAT_PROBS(0.14),
            "sym_mood_worsened":      _FLAT_PROBS(0.16),
            "sym_depression_episode": _FLAT_PROBS(0.05),
            "sym_anxiety":            _FLAT_PROBS(0.12),
            "sym_libido_decreased":   _FLAT_PROBS(0.15),
            "sym_weight_gain":        _FLAT_PROBS(0.14),
            "sym_acne_improved":      _FLAT_PROBS(0.10),
            "sym_acne_worsened":      _FLAT_PROBS(0.12),
            "sym_hair_loss":          _FLAT_PROBS(0.06),
            "sym_cramps_relieved":    _FLAT_PROBS(0.20),
            "sym_pcos_improvement":   _FLAT_PROBS(0.04),
            "evt_dvt":                _FLAT_PROBS(0.0014),
            "evt_pe":                 _FLAT_PROBS(0.0010),
            "evt_stroke":             _FLAT_PROBS(0.0007),
            "still_taking":           _INCR_PROBS(0.92, 0.73),
        },
        "satisfaction": _INCR_PROBS(5.5, 6.0),
    },
    "pill_d_high_risk": {
        "discontinuation_probability": 0.40,
        "severe_event_probability": 0.030,
        "mild_side_effect_score": 0.70,
        "contraceptive_effectiveness": 0.93,
        "combo_id": "pill_d_high_risk",
        "n_months": 12,
        "months": list(range(1, 13)),
        "symptom_probs": {
            "sym_nausea":             _FLAT_PROBS(0.30),
            "sym_headache":           _FLAT_PROBS(0.28),
            "sym_breast_tenderness":  _FLAT_PROBS(0.25),
            "sym_spotting":           _FLAT_PROBS(0.22),
            "sym_mood_worsened":      _FLAT_PROBS(0.26),
            "sym_depression_episode": _FLAT_PROBS(0.10),
            "sym_anxiety":            _FLAT_PROBS(0.20),
            "sym_libido_decreased":   _FLAT_PROBS(0.25),
            "sym_weight_gain":        _FLAT_PROBS(0.22),
            "sym_acne_improved":      _FLAT_PROBS(0.05),
            "sym_acne_worsened":      _FLAT_PROBS(0.20),
            "sym_hair_loss":          _FLAT_PROBS(0.10),
            "sym_cramps_relieved":    _FLAT_PROBS(0.10),
            "sym_pcos_improvement":   _FLAT_PROBS(0.02),
            "evt_dvt":                _FLAT_PROBS(0.003),
            "evt_pe":                 _FLAT_PROBS(0.002),
            "evt_stroke":             _FLAT_PROBS(0.001),
            "still_taking":           _INCR_PROBS(0.88, 0.60),
        },
        "satisfaction": _INCR_PROBS(4.5, 5.0),
    },
}


async def call_simulator(pill_record: dict, patient_data: dict, n_months: int = 12) -> dict:
    """
    Returns deterministic simulation results for a candidate pill.

    Signature matches the real agent/tools/simulator_api.call_simulator.
    Known pills return hardcoded values; unknown pills use seeded RNG.
    Response shape mirrors the real model: trajectory + 4 summary metrics.
    """
    # Extract pill identifier (same logic as the real simulator_api.py)
    candidate_id = (
        pill_record.get("combo_id")
        or pill_record.get("set_id")
        or pill_record.get("pill_id")
        or str(pill_record.get("brand_name", "unknown"))
    ) if isinstance(pill_record, dict) else str(pill_record)

    if candidate_id in FAKE_SIMULATION_RESULTS:
        return FAKE_SIMULATION_RESULTS[candidate_id]

    # Unknown pill: deterministic noise seeded by pill name
    seed = sum(ord(c) for c in candidate_id)
    rng = random.Random(seed)
    disc  = round(rng.uniform(0.05, 0.40), 3)
    severe = round(rng.uniform(0.001, 0.025), 4)
    mild   = round(rng.uniform(0.10, 0.80), 2)
    eff    = round(rng.uniform(0.90, 0.99), 3)
    return {
        "discontinuation_probability": disc,
        "severe_event_probability":    severe,
        "mild_side_effect_score":      mild,
        "contraceptive_effectiveness": eff,
        "combo_id":  candidate_id,
        "n_months":  n_months,
        "months":    list(range(1, n_months + 1)),
        "symptom_probs": {t: [round(rng.uniform(0.01, 0.30), 4)] * n_months for t in [
            "sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting",
            "sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
            "sym_libido_decreased", "sym_weight_gain", "sym_acne_improved",
            "sym_acne_worsened", "sym_hair_loss", "sym_cramps_relieved",
            "sym_pcos_improvement", "evt_dvt", "evt_pe", "evt_stroke", "still_taking",
        ]},
        "satisfaction": [round(rng.uniform(4.0, 9.0), 2)] * n_months,
    }

