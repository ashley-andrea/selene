"""
Cluster Node — calls the Cluster Model API and evaluates confidence.

If the confidence score is below the threshold (0.70), the LLM is invoked
to produce a weight adjustment for relative risk rules. This is the only
place in the graph where the cluster model is called.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.state import SystemState
from agent.tools.cluster_api import call_cluster_model

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.70
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "weight_adjustment.txt"


def _load_prompt_template() -> str:
    """Load the weight adjustment prompt template from file."""
    return _PROMPT_PATH.read_text()


def run(state: SystemState) -> dict:
    """
    Calls the Cluster Model and evaluates confidence.
    Low-confidence assignments trigger an LLM-driven weight adjustment.
    """
    patient_data = state["patient_data"]
    result = call_cluster_model(patient_data)

    cluster_profile = result["cluster_profile"]
    cluster_confidence = result["cluster_confidence"]

    update = {
        "cluster_profile": cluster_profile,
        "cluster_confidence": cluster_confidence,
    }

    # Low-confidence: invoke LLM for weight adjustment
    if cluster_confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            "Low cluster confidence %.2f (threshold=%.2f) — requesting LLM weight adjustment",
            cluster_confidence,
            CONFIDENCE_THRESHOLD,
        )
        try:
            llm = get_llm()
            template = _load_prompt_template()
            prompt = template.format(
                cluster_profile=cluster_profile,
                cluster_confidence=cluster_confidence,
                patient_age=patient_data.get("age", "unknown"),
                patient_pathologies=", ".join(patient_data.get("pathologies", [])) or "none",
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            adjustment = json.loads(response.content)

            logger.info(
                "LLM weight adjustment: %.2f — %s",
                adjustment.get("weight_adjustment", 1.0),
                adjustment.get("rationale", "no rationale"),
            )

            # Enrich cluster_profile with adjustment metadata
            update["cluster_profile"] = {
                "profile": cluster_profile,
                "weight_adjustment": adjustment.get("weight_adjustment", 1.0),
                "low_confidence": True,
                "rationale": adjustment.get("rationale", ""),
            }

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse LLM weight adjustment: %s", exc)
            # Fall back to default — no adjustment
            update["cluster_profile"] = {
                "profile": cluster_profile,
                "weight_adjustment": 1.0,
                "low_confidence": True,
                "rationale": "LLM parsing failed — using default weight",
            }

    return update
