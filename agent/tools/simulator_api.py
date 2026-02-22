"""
HTTP client for the Simulator Model API on Red Hat OpenShift.

This is one of exactly two files that touch the ML model boundary.
If the ML team changes the request/response schema, update ONLY this file.
Nothing else in the agent needs to change.

SCHEMA SOURCE OF TRUTH: models/simulation/SIMULATION_MODEL.md  and  _docs/ML_models_API.md

Real model endpoint (models/simulation/serve.py):
    POST /api/v1/simulator/simulate

Request payload:
    {
        "candidate_pill": { "combo_id": "EE30_DRSP3", ...pill record fields... },
        "patient":        { "age": 28, "cond_*": 0/1, "obs_*": float|null, ... },
        "n_months":       12   (optional, default 12)
    }

Response (full trajectory + backward-compatible summary):
    {
        "combo_id":     "EE30_DRSP3",
        "n_months":     12,
        "months":       [1, 2, ..., 12],
        "symptom_probs": {
            "sym_nausea": [0.02, ...],
            "still_taking": [0.91, ...],
            "evt_dvt": [0.0, ...],
            ...  (18 binary targets total)
        },
        "satisfaction": [6.1, 6.3, ...],
        // Derived summary metrics — consumed by the agent utility node:
        "discontinuation_probability": 0.09,
        "severe_event_probability":    0.0002,
        "mild_side_effect_score":      0.18,
        "contraceptive_effectiveness": 0.63
    }

Set SIMULATOR_API_URL to the real OpenShift route, e.g.:
    SIMULATOR_API_URL=https://simulator-model.apps.<cluster>/api/v1/simulator/simulate
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


async def call_simulator(pill_record: dict, patient_data: dict, n_months: int = 12) -> dict:
    """
    Async call to the real Simulator Model API (models/simulation/serve.py).

    Request:
        POST {SIMULATOR_API_URL}
        {
            "candidate_pill": {...pill record from pill_reference_db.csv...},
            "patient":        {age, cond_*, obs_*, med_*, ...},
            "n_months":       12
        }

    Response (full trajectory + summary):
        {
            "combo_id":     str,
            "n_months":     int,
            "months":       [1..N],
            "symptom_probs": {"sym_nausea": [...], "still_taking": [...], ...},
            "satisfaction": [...],
            // Summary metrics (consumed by utility node):
            "discontinuation_probability": float,
            "severe_event_probability":    float,
            "mild_side_effect_score":      float,
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
    
    # Extract pill_id for logging (uses combo_id from pill_reference_db.csv)
    pill_id = cleaned_pill.get("combo_id") or cleaned_pill.get("set_id", "unknown")
    
    payload = {
        "candidate_pill": cleaned_pill,
        "patient": transformed_patient,
        "n_months": n_months,
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
            "Simulation for %s (%d months): disc=%.3f severe=%.5f mild=%.3f eff=%.3f",
            pill_id,
            data.get("n_months", n_months),
            data.get("discontinuation_probability", 0),
            data.get("severe_event_probability", 0),
            data.get("mild_side_effect_score", 0),
            data.get("contraceptive_effectiveness", 0),
        )
        return data
