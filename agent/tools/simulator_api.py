"""
HTTP client for the Simulator Model API on Red Hat OpenShift.

This is one of exactly two files that touch the ML model boundary.
If the ML team changes the request/response schema, update ONLY this file
and the mock server. Nothing else in the agent needs to change.

SCHEMA SOURCE OF TRUTH: ML_models_API.md
"""

import logging
import os
import math

import httpx

from agent.tools.cluster_api import _transform_patient_data_for_cluster_model

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15.0
RETRY_DELAY_503 = 2.0


def _clean_pill_record(pill_record: dict) -> dict:
    """
    Clean pill record by replacing NaN values with None.
    Pandas DataFrames can have NaN values which are not JSON-compliant.
    """
    cleaned = {}
    for key, value in pill_record.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


async def call_simulator(pill_record: dict, patient_data: dict) -> dict:
    """
    Async call to the Simulator Model API.

    Request:
        POST {SIMULATOR_API_URL}
        {"candidate_pill": {...pill fields...}, "patient": {age, cond_*, obs_*, med_*, ...}}

    Response:
        {
            "discontinuation_probability": float,
            "severe_event_probability": float,
            "mild_side_effect_score": float,
            "contraceptive_effectiveness": float
        }

    Error handling:
        - 400/422 → raises immediately (treated as a bug)
        - 503     → retries once after 2 s (via asyncio.sleep)
        - 500     → logged, candidate skipped by the caller
    """
    url = os.getenv("SIMULATOR_API_URL")
    if not url:
        raise ValueError("SIMULATOR_API_URL environment variable not set")

    # Transform patient data to the format expected by the  ML model
    transformed_patient = _transform_patient_data_for_cluster_model(patient_data)
    
    # Clean pill record (replace NaN with None for JSON serialization)
    cleaned_pill = _clean_pill_record(pill_record)
    
    # Extract pill_id for logging
    pill_id = cleaned_pill.get("pill_id") or cleaned_pill.get("set_id", "unknown")
    
    payload = {
        "candidate_pill": cleaned_pill,
        "patient": transformed_patient,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        try:
            response = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            logger.error("Simulator API request failed for %s: %s", pill_id, exc)
            raise

        # Retry once on 503 (model warming up)
        if response.status_code == 503:
            import asyncio

            logger.warning(
                "Simulator API returned 503 for %s — retrying in %.1fs",
                pill_id,
                RETRY_DELAY_503,
            )
            await asyncio.sleep(RETRY_DELAY_503)
            response = await client.post(url, json=payload)

        # 500 → skip this candidate (caller handles missing results)
        if response.status_code == 500:
            logger.error(
                "Simulator API 500 for %s: %s — candidate will be skipped",
                pill_id,
                response.text,
            )
            return None

        if response.status_code != 200:
            logger.error(
                "Simulator API error %d for %s: %s",
                response.status_code,
                pill_id,
                response.text,
            )
            response.raise_for_status()

        data = response.json()
        logger.info(
            "Simulation for %s: disc=%.3f severe=%.4f mild=%.2f eff=%.3f",
            pill_id,
            data.get("discontinuation_probability", 0),
            data.get("severe_event_probability", 0),
            data.get("mild_side_effect_score", 0),
            data.get("contraceptive_effectiveness", 0),
        )
        return data
