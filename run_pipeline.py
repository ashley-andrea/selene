"""
Quick script to run the agent pipeline with test patient data or a medical PDF.

Usage:
    python run_pipeline.py [patient_type]           # from tests/fixtures
    python run_pipeline.py --json path/to/file.json # from a JSON file directly
    python run_pipeline.py --pdf  path/to/file.pdf  # extract from PDF then run

Examples:
    python run_pipeline.py standard
    python run_pipeline.py high_risk
    python run_pipeline.py --json test_patient.json
    python run_pipeline.py --pdf Medical_record_pdf/UMNwriteup.pdf
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


def load_patient_json(json_path: str) -> dict:
    """Load a patient record directly from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        logger.error("JSON file not found: %s", json_path)
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    # Strip meta-only keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def extract_patient_from_pdf(pdf_path: str) -> dict:
    """Parse a medical PDF and extract patient data via LLM."""
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    from agent.pdf_parser import pages_to_document, parse_pdf
    from agent.patient_extractor import extract_patient_data

    logger.info("Parsing PDF: %s (%d bytes)", path.name, path.stat().st_size)
    pages = parse_pdf(path.read_bytes())
    document_text = pages_to_document(pages)
    logger.info("Extracted %d characters of text across %d page(s)", len(document_text), len(pages))

    logger.info("Running LLM extraction…")
    patient_data = extract_patient_data(document_text)
    logger.info("Extracted patient data:\n%s", json.dumps(patient_data, indent=2))
    return patient_data


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
                logger.info("  Pill type:    %s", pill.get("pill_type"))
                logger.info("  Estrogen:     %s %s mcg", pill.get("estrogen", "—"), pill.get("estrogen_dose_mcg", ""))
                logger.info("  Progestin:    %s %s mg", pill.get("progestin"), pill.get("progestin_dose_mg"))
                logger.info("  VTE risk:     %s", pill.get("vte_risk_class"))
                logger.info("  Known brands: %s", pill.get("known_brand_examples", ""))
        
        logger.info("Reason codes: %s", final_state.get("reason_codes", []))
        logger.info("=" * 80)
        
        return final_state
        
    except Exception as exc:
        logger.exception("Pipeline failed with error")
        raise


def main():
    """Main entry point"""
    args = sys.argv[1:]

    if args and args[0] == "--pdf":
        if len(args) < 2:
            logger.error("Usage: python run_pipeline.py --pdf path/to/file.pdf")
            sys.exit(1)
        mode = "pdf"
        patient_data = extract_patient_from_pdf(args[1])

    elif args and args[0] == "--json":
        if len(args) < 2:
            logger.error("Usage: python run_pipeline.py --json path/to/file.json")
            sys.exit(1)
        mode = "json"
        patient_data = load_patient_json(args[1])
        logger.info("Loaded patient from %s", args[1])

    else:
        patient_type = args[0] if args else "standard"
        mode = "fixture"
        logger.info("Loading patient fixture: %s", patient_type)
        patient_data = load_patient_fixture(patient_type)

    logger.info("Mode: %s | Patient: %s", mode, json.dumps(patient_data, indent=2))

    # Run the pipeline
    final_state = asyncio.run(run_pipeline(patient_data))

    logger.info("\n✅ Pipeline completed successfully!")

    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        logger.info("🔍 View trace on LangSmith: https://smith.langchain.com/projects/p/%s", project)


if __name__ == "__main__":
    main()
