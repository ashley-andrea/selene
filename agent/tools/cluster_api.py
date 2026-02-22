"""
HTTP client for the Cluster Model API on Red Hat OpenShift.

This is one of exactly two files that touch the ML model boundary.
If the ML team changes the request/response schema, update ONLY this file
and the mock server. Nothing else in the agent needs to change.

SCHEMA SOURCE OF TRUTH: ML_models_API.md
"""

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15.0
RETRY_DELAY_503 = 2.0


def _transform_patient_data_for_cluster_model(patient_data: dict) -> dict:
    """
    Transform internal patient data format to the format expected by the Cluster Model API.
    
    The agent internally uses simplified format (pathologies, habits, medical_history arrays).
    The ML model expects structured numeric fields (cond_*, obs_*, med_* as specified in ML_inputs.txt).
    
    This function maps from the old format to the new format.
    If patient_data already contains the new format fields, they are used as-is.
    Otherwise, we derive them from the old format where possible, defaulting to 0/null.
    """
    # If the data is already in the new format, return it
    if "cond_migraine_with_aura" in patient_data or "obs_bmi" in patient_data:
        return patient_data
    
    # Otherwise, transform from old format to new format
    # Extract old-format fields
    age = patient_data.get("age", 25)
    pathologies = set(patient_data.get("pathologies", []))
    habits = set(patient_data.get("habits", []))
    medical_history = set(patient_data.get("medical_history", []))
    
    # Map to new format - all condition fields (cond_*)
    transformed = {
        "age": age,
        "cond_migraine_with_aura": 1 if "migraine_with_aura" in pathologies or "migraines_with_aura" in pathologies else 0,
        "cond_stroke": 1 if "stroke" in medical_history else 0,
        "cond_mi": 1 if "mi" in medical_history or "myocardial_infarction" in medical_history else 0,
        "cond_dvt": 1 if "dvt" in medical_history or "deep_vein_thrombosis" in medical_history else 0,
        "cond_breast_cancer": 1 if "breast_cancer" in medical_history else 0,
        "cond_lupus": 1 if "lupus" in pathologies else 0,
        "cond_thrombophilia": 1 if "thrombophilia" in pathologies or "thrombophilia" in medical_history else 0,
        "cond_atrial_fibrillation": 1 if "atrial_fibrillation" in pathologies else 0,
        "cond_liver_disease": 1 if "liver_disease" in pathologies or "liver_disease" in medical_history else 0,
        "cond_hypertension": 1 if "hypertension" in pathologies or "high_blood_pressure" in pathologies else 0,
        "cond_migraine": 1 if "migraines" in pathologies or "migraine" in pathologies else 0,
        "cond_gallstones": 1 if "gallstones" in pathologies or "gallstones" in medical_history else 0,
        "cond_diabetes": 1 if "diabetes" in pathologies else 0,
        "cond_prediabetes": 1 if "prediabetes" in pathologies else 0,
        "cond_epilepsy": 1 if "epilepsy" in pathologies else 0,
        "cond_chronic_kidney_disease": 1 if "chronic_kidney_disease" in pathologies or"kidney_disease" in pathologies else 0,
        "cond_sleep_apnea": 1 if "sleep_apnea" in pathologies else 0,
        "cond_pcos": 1 if "pcos" in pathologies or "polycystic_ovary" in pathologies else 0,
        "cond_endometriosis": 1 if "endometriosis" in pathologies else 0,
        "cond_depression": 1 if "depression" in pathologies else 0,
        "cond_hypothyroidism": 1 if "hypothyroidism" in pathologies else 0,
        "cond_rheumatoid_arthritis": 1 if "rheumatoid_arthritis" in pathologies or "arthritis" in pathologies else 0,
        "cond_fibromyalgia": 1 if "fibromyalgia" in pathologies else 0,
        "cond_osteoporosis": 1 if "osteoporosis" in pathologies else 0,
        "cond_asthma": 1 if "asthma" in pathologies else 0,
    }
    
    # Observation fields (obs_*) - use provided values or reasonable defaults
    transformed["obs_bmi"] = patient_data.get("bmi", patient_data.get("obs_bmi", 25.0))
    transformed["obs_systolic_bp"] = patient_data.get("systolic_bp", patient_data.get("obs_systolic_bp", 120))
    transformed["obs_diastolic_bp"] = patient_data.get("diastolic_bp", patient_data.get("obs_diastolic_bp", 80))
    transformed["obs_phq9_score"] = patient_data.get("phq9_score", patient_data.get("obs_phq9_score", 0))
    transformed["obs_pain_score"] = patient_data.get("pain_score", patient_data.get("obs_pain_score", 0))
    transformed["obs_testosterone"] = patient_data.get("testosterone", patient_data.get("obs_testosterone", 40.0))
    transformed["obs_smoker"] = 1 if "smoking" in habits or "smoker" in habits else 0
    
    # Medication history fields (med_*)
    transformed["med_ever_ocp"] = patient_data.get("ever_used_ocp", patient_data.get("med_ever_ocp", 0))
    transformed["med_current_combined_ocp"] = patient_data.get("current_combined_ocp", patient_data.get("med_current_combined_ocp", 0))
    transformed["med_current_minipill"] = patient_data.get("current_minipill", patient_data.get("med_current_minipill", 0))
    
    # Absolute contraindication field
    transformed["has_absolute_contraindication_combined_oc"] = patient_data.get(
        "has_absolute_contraindication_combined_oc", 0
    )
    
    return transformed


def call_cluster_model(patient_data: dict) -> dict:
    """
    Synchronous call to the Cluster Model API.

    Request:
        POST {CLUSTER_API_URL}
        {"patient": {age, cond_*, obs_*, med_*, has_absolute_contraindication_combined_oc}}

    Response:
        {"cluster_profile": str, "cluster_confidence": float}

    Error handling:
        - 400/422 → raises immediately (treated as a bug)
        - 503     → retries once after 2 s
        - 500     → raises with logged context
    """
    url = os.getenv("CLUSTER_API_URL")
    if not url:
        raise ValueError("CLUSTER_API_URL environment variable not set")

    # Transform patient data to the format expected by the ML model
    transformed_patient = _transform_patient_data_for_cluster_model(patient_data)
    payload = {"patient": transformed_patient}

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        try:
            response = client.post(url, json=payload)
        except httpx.RequestError as exc:
            logger.error("Cluster API request failed: %s", exc)
            raise

        # Retry once on 503 (model warming up)
        if response.status_code == 503:
            logger.warning("Cluster API returned 503 — retrying in %.1fs", RETRY_DELAY_503)
            time.sleep(RETRY_DELAY_503)
            response = client.post(url, json=payload)

        if response.status_code != 200:
            logger.error(
                "Cluster API error %d: %s",
                response.status_code,
                response.text,
            )
            response.raise_for_status()

        data = response.json()
        logger.info(
            "Cluster assignment: %s (confidence=%.2f)",
            data.get("cluster_profile"),
            data.get("cluster_confidence", 0.0),
        )
        return data
