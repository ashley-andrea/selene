"""
FastAPI Backend — Birth Control Recommendation Agent.

This is the main entry point for the backend. It exposes the /recommend
endpoint that accepts patient data, runs the LangGraph agent loop, and
returns a structured recommendation.

The Safe Gate Engine runs inside this backend before the agent sees any
data — the agent only operates on the filtered candidate pool.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

from agent.graph import agent_graph
from agent.pdf_parser import pages_to_document, parse_pdf
from agent.patient_extractor import extract_patient_data

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BC Recommendation Agent starting up")
    logger.info("LLM_PROVIDER=%s", os.getenv("LLM_PROVIDER", "claude"))
    logger.info("CLUSTER_API_URL=%s", os.getenv("CLUSTER_API_URL", "NOT SET"))
    logger.info("SIMULATOR_API_URL=%s", os.getenv("SIMULATOR_API_URL", "NOT SET"))
    yield
    logger.info("BC Recommendation Agent shutting down")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BC Recommendation Agent",
    version="0.1.0",
    description="LangGraph-based agent for oral contraceptive recommendations",
    lifespan=lifespan,
)

# CORS — permissive for development, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ───────────────────────────────────────────────
class PatientInput(BaseModel):
    """Patient data submitted for a recommendation."""

    age: int = Field(..., ge=15, le=55, description="Patient age in years")
    pathologies: list[str] = Field(default_factory=list, description="Active diagnosed conditions")
    habits: list[str] = Field(default_factory=list, description="Lifestyle habits")
    medical_history: list[str] = Field(default_factory=list, description="Past medical events")


class PDFExtractionResult(BaseModel):
    """Patient data extracted from an uploaded medical PDF.  Returned to the
    frontend so the user can review and adjust values before submitting the
    final recommendation request."""

    age: int | None = Field(None, description="Patient age, or null if not found in the document")
    pathologies: list[str] = Field(default_factory=list, description="Active diagnosed conditions")
    habits: list[str] = Field(default_factory=list, description="Lifestyle habits")
    medical_history: list[str] = Field(default_factory=list, description="Past medical events")
    pages_parsed: int = Field(0, description="Number of PDF pages that were parsed")
    parser_backend: str = Field("", description="Which parser was used: 'dots_ocr' or 'pymupdf'")


class PillRecommendation(BaseModel):
    """Detailed information about a recommended pill."""
    
    pill_id: str
    rank: int  # 1, 2, or 3
    utility_score: float
    predicted_discontinuation: float
    severe_risk: float
    mild_side_effect_score: float
    contraceptive_effectiveness: float
    reason_codes: list[str]


class RecommendationOutput(BaseModel):
    """Structured recommendation returned to the frontend - TOP 3 pills with full details."""

    # Top 3 recommendations in order
    recommendations: list[PillRecommendation] = Field(
        ..., 
        min_items=1, 
        max_items=3,
        description="Top 3 pill recommendations with full simulation details and reasoning"
    )
    
    # Metadata about the agent's process
    cluster_profile: str
    cluster_confidence: float
    iterations: int
    total_candidates_evaluated: int
    
    # The primary recommendation (same as recommendations[0])
    selected_pill: str


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendationOutput)
async def recommend(patient: PatientInput):
    """
    Run the full agent loop for a patient and return a recommendation.

    The Safe Gate Engine filters contraindicated pills before the agent
    starts. The agent iterates: simulate → score → converge → (repeat or exit).
    """
    logger.info("Received recommendation request for patient age=%d", patient.age)

    initial_state = {
        "patient_data": patient.model_dump(),
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

    try:
        final_state = await agent_graph.ainvoke(
            initial_state,
            config={"recursion_limit": 50}  # Allow up to 50 node executions for complex cases
        )
    except ValueError as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected agent error")
        raise HTTPException(status_code=500, detail="Internal agent error")

    return _build_output(final_state)


@app.post("/api/v1/patient/upload-pdf", response_model=PDFExtractionResult)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a medical PDF (cartella clinica / FSE) and extract patient data.

    The response pre-fills the intake form on the frontend.  The user reviews
    the extracted values and submits the confirmed data via POST /recommend.

    Supported formats: PDF only.
    Parser: dots.ocr (if DOTS_OCR_URL env var is set) with pymupdf fallback.
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Some browsers send application/octet-stream for PDFs — accept both.
        # We do a secondary check by inspecting the file magic bytes below.
        pass

    logger.info("PDF upload: filename=%s, content_type=%s", file.filename, file.content_type)

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 5 or pdf_bytes[:4] != b"%PDF":
        raise HTTPException(
            status_code=422,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    try:
        pages = parse_pdf(pdf_bytes)
    except Exception as exc:
        logger.exception("PDF parsing failed")
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {exc}")

    if not pages:
        raise HTTPException(status_code=422, detail="PDF contains no parseable pages.")

    document_text = pages_to_document(pages)

    try:
        patient_data = extract_patient_data(document_text)
    except Exception as exc:
        logger.exception("Patient data extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    # Determine which backend was used (all pages come from the same backend)
    parser_backend = pages[0].get("source", "unknown") if pages else "unknown"

    return PDFExtractionResult(
        age=patient_data.get("age"),
        pathologies=patient_data.get("pathologies", []),
        habits=patient_data.get("habits", []),
        medical_history=patient_data.get("medical_history", []),
        pages_parsed=len(pages),
        parser_backend=parser_backend,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "bc-recommendation-agent",
        "llm_provider": os.getenv("LLM_PROVIDER", "claude"),
    }


@app.get("/pills")
async def list_pills():
    """List all pills in the database (for development/debugging)."""
    from agent.pill_database import get_all_pills

    df = get_all_pills()
    return df.to_dict(orient="records")


# ── Helpers ─────────────────────────────────────────────────────────────────
def _build_output(state: dict) -> RecommendationOutput:
    """Transform the final agent state into the API response with top 3 pills."""
    utility_scores = state.get("utility_scores", {})
    simulated_results = state.get("simulated_results", {})
    
    # Get top 3 pills by utility score
    sorted_pills = sorted(
        utility_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    if not sorted_pills:
        raise HTTPException(status_code=500, detail="No candidates evaluated")
    
    cluster = state.get("cluster_profile", "")
    if isinstance(cluster, dict):
        cluster = cluster.get("profile", "unknown")
    
    # Build detailed recommendations for each of top 3
    recommendations = []
    for rank, (pill_id, utility) in enumerate(sorted_pills, start=1):
        sim = simulated_results.get(pill_id, {})
        
        # Generate pill-specific reason codes
        # For rank 1, use the agent's reason codes
        # For ranks 2 and 3, generate comparative reason codes
        if rank == 1:
            reason_codes = state.get("reason_codes", [f"highest utility score ({utility:.3f})"])
        else:
            reason_codes = _generate_comparative_reasons(
                pill_id, sim, utility, rank, sorted_pills[0][0], utility_scores[sorted_pills[0][0]]
            )
        
        recommendations.append(PillRecommendation(
            pill_id=pill_id,
            rank=rank,
            utility_score=round(utility, 4),
            predicted_discontinuation=sim.get("discontinuation_probability", 0.0),
            severe_risk=sim.get("severe_event_probability", 0.0),
            mild_side_effect_score=sim.get("mild_side_effect_score", 0.0),
            contraceptive_effectiveness=sim.get("contraceptive_effectiveness", 0.0),
            reason_codes=reason_codes,
        ))
    
    return RecommendationOutput(
        recommendations=recommendations,
        cluster_profile=cluster,
        cluster_confidence=state.get("cluster_confidence", 0.0),
        iterations=state.get("iteration", 0),
        total_candidates_evaluated=len(utility_scores),
        selected_pill=sorted_pills[0][0],  # Best pill
    )


def _generate_comparative_reasons(
    pill_id: str,
    sim: dict,
    utility: float,
    rank: int,
    best_pill_id: str,
    best_utility: float,
) -> list[str]:
    """Generate comparative reason codes for non-primary recommendations."""
    reasons = []
    
    # Utility comparison
    utility_diff = best_utility - utility
    reasons.append(f"utility score {utility:.3f} ({utility_diff:.3f} below top choice)")
    
    # Highlight strengths
    if sim.get("severe_event_probability", 1.0) < 0.005:
        reasons.append(f"very low severe event risk ({sim['severe_event_probability']:.1%})")
    
    if sim.get("discontinuation_probability", 1.0) < 0.15:
        reasons.append(f"low discontinuation probability ({sim['discontinuation_probability']:.1%})")
    
    if sim.get("contraceptive_effectiveness", 0.0) >= 0.95:
        reasons.append(f"high contraceptive effectiveness ({sim['contraceptive_effectiveness']:.1%})")
    
    # Add rank context
    if rank == 2:
        reasons.append("strong alternative if primary choice is not tolerated")
    elif rank == 3:
        reasons.append("viable backup option for this patient profile")
    
    return reasons[:4]  # Limit to 4 reasons
