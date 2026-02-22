"""
Quick script to run the agent pipeline with test patient data.
This will show the full execution on LangSmith.

Usage:
    python run_pipeline.py [patient_type]
    
Examples:
    python run_pipeline.py standard
    python run_pipeline.py high_risk
    python run_pipeline.py young_healthy
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agent.graph import build_graph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_patient_fixture(patient_type: str = "standard") -> dict:
    """Load a patient fixture from tests/fixtures/patients.json"""
    fixtures_file = Path(__file__).parent / "tests" / "fixtures" / "patients.json"
    with open(fixtures_file) as f:
        fixtures = json.load(f)
    
    if patient_type not in fixtures:
        available = [k for k in fixtures.keys() if not k.startswith("_")]
        logger.error("Patient type '%s' not found. Available: %s", patient_type, available)
        sys.exit(1)
    
    patient = fixtures[patient_type]
    if "description" in patient:
        logger.info("Patient: %s", patient["description"])
        patient = {k: v for k, v in patient.items() if k != "description"}
    
    return patient


async def run_pipeline(patient_data: dict):
    """Run the agent pipeline with the given patient data"""
    
    logger.info("=" * 80)
    logger.info("STARTING PIPELINE RUN")
    logger.info("=" * 80)
    logger.info("Patient data: %s", json.dumps(patient_data, indent=2))
    logger.info("CLUSTER_API_URL: %s", os.getenv("CLUSTER_API_URL", "NOT SET"))
    logger.info("SIMULATOR_API_URL: %s", os.getenv("SIMULATOR_API_URL", "NOT SET"))
    logger.info("LANGCHAIN_TRACING_V2: %s", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    logger.info("LANGCHAIN_PROJECT: %s", os.getenv("LANGCHAIN_PROJECT", "default"))
    logger.info("=" * 80)
    
    initial_state = {
        "patient_data": patient_data,
        "iteration": 0,
        "converged": False,
        "cluster_profile": None,
        "cluster_confidence": None,
        "relative_risk_rules": [],
        "candidate_pool": [],
        "risk_scores": None,
        "pills_to_simulate": None,
        "simulated_results": {},
        "utility_scores": {},
        "utility_weights": None,
        "best_candidate": None,
        "previous_best_utility": None,
        "reason_codes": [],
    }
    
    graph = build_graph()
    
    try:
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 50})
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 80)
        logger.info("Converged: %s", final_state["converged"])
        logger.info("Iterations: %s", final_state["iteration"])
        logger.info("Cluster: %s (confidence: %.2f)", 
                   final_state["cluster_profile"], 
                   final_state.get("cluster_confidence", 0))
        logger.info("Candidate pool size: %d", len(final_state["candidate_pool"]))
        logger.info("Best candidate: %s", final_state["best_candidate"])
        
        if final_state["best_candidate"]:
            from agent.pill_database import get_pill_by_id
            pill = get_pill_by_id(final_state["best_candidate"])
            if pill:
                logger.info("  Brand: %s", pill.get("brand_name"))
                logger.info("  Generic: %s", pill.get("generic_name"))
                logger.info("  Substance: %s", pill.get("substance_name"))
        
        logger.info("Reason codes: %s", final_state.get("reason_codes", []))
        logger.info("=" * 80)
        
        return final_state
        
    except Exception as exc:
        logger.exception("Pipeline failed with error")
        raise


def main():
    """Main entry point"""
    patient_type = sys.argv[1] if len(sys.argv) > 1 else "standard"
    
    logger.info("Loading patient: %s", patient_type)
    patient_data = load_patient_fixture(patient_type)
    
    # Run the pipeline
    final_state = asyncio.run(run_pipeline(patient_data))
    
    logger.info("\n✅ Pipeline completed successfully!")
    
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        logger.info("🔍 View trace on LangSmith: https://smith.langchain.com/o/[your-org]/projects/p/%s", project)


if __name__ == "__main__":
    main()
