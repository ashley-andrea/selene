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

# CORS — allow_origins from env var for production (Vercel URL), wildcard for local dev.
# allow_credentials must be False when allow_origins=["*"] — browsers reject the combo.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
_allow_origins: list[str] | str = (
    [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=_raw_origins != "*",  # True only when specific origins are set
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ───────────────────────────────────────────────
class PatientInput(BaseModel):
    """Patient data submitted for a recommendation.

    Accepts both the compact form-based format (age + lists) AND the full
    structured format produced by the PDF extractor (obs_*, cond_* fields).
    When obs_* / cond_* fields are supplied they are passed directly to the
    cluster/simulator models; when absent the models use imputed defaults.
    """

    # Core demographics
    age: int = Field(..., ge=15, le=60, description="Patient age in years")

    # Human-readable lists (used by safe_gate for hard constraints)
    pathologies: list[str] = Field(default_factory=list, description="Active diagnosed conditions")
    habits: list[str] = Field(default_factory=list, description="Lifestyle habits")
    medical_history: list[str] = Field(default_factory=list, description="Past medical events")

    # ── Numeric vitals (optional — imputed by the ML model if absent) ────────
    obs_bmi: float | None = Field(None, ge=10.0, le=60.0, description="BMI")
    obs_systolic_bp: float | None = Field(None, ge=70.0, le=220.0, description="Systolic BP (mmHg)")
    obs_diastolic_bp: float | None = Field(None, ge=40.0, le=140.0, description="Diastolic BP (mmHg)")
    obs_phq9_score: float | None = Field(None, ge=0.0, le=27.0, description="PHQ-9 depression score")
    obs_testosterone: float | None = Field(None, ge=0.0, le=300.0, description="Testosterone (ng/dL)")
    obs_smoker: int = Field(0, ge=0, le=1, description="Current smoker: 1=yes, 0=no")
    obs_pain_score: float = Field(0.0, ge=0.0, le=10.0, description="Pain score 0-10")

    # ── Binary condition flags (optional — derived from lists if absent) ─────
    cond_migraine_with_aura: int = Field(0, ge=0, le=1)
    cond_stroke: int = Field(0, ge=0, le=1)
    cond_mi: int = Field(0, ge=0, le=1)
    cond_dvt: int = Field(0, ge=0, le=1)
    cond_breast_cancer: int = Field(0, ge=0, le=1)
    cond_lupus: int = Field(0, ge=0, le=1)
    cond_thrombophilia: int = Field(0, ge=0, le=1)
    cond_atrial_fibrillation: int = Field(0, ge=0, le=1)
    cond_liver_disease: int = Field(0, ge=0, le=1)
    cond_hypertension: int = Field(0, ge=0, le=1)
    cond_migraine: int = Field(0, ge=0, le=1)
    cond_gallstones: int = Field(0, ge=0, le=1)
    cond_diabetes: int = Field(0, ge=0, le=1)
    cond_prediabetes: int = Field(0, ge=0, le=1)
    cond_epilepsy: int = Field(0, ge=0, le=1)
    cond_chronic_kidney_disease: int = Field(0, ge=0, le=1)
    cond_sleep_apnea: int = Field(0, ge=0, le=1)
    cond_pcos: int = Field(0, ge=0, le=1)
    cond_endometriosis: int = Field(0, ge=0, le=1)
    cond_depression: int = Field(0, ge=0, le=1)
    cond_hypothyroidism: int = Field(0, ge=0, le=1)
    cond_rheumatoid_arthritis: int = Field(0, ge=0, le=1)
    cond_fibromyalgia: int = Field(0, ge=0, le=1)
    cond_osteoporosis: int = Field(0, ge=0, le=1)
    cond_asthma: int = Field(0, ge=0, le=1)

    # Absolute contraindication flag (computed by safe_gate but can be pre-set)
    med_ever_ocp: int = Field(0, ge=0, le=1)
    med_current_combined_ocp: int = Field(0, ge=0, le=1)
    med_current_minipill: int = Field(0, ge=0, le=1)
    has_absolute_contraindication_combined_oc: int = Field(0, ge=0, le=1)


class PDFExtractionResult(BaseModel):
    """Patient data extracted from an uploaded medical PDF.  Returned to the
    frontend so the user can review and adjust values before submitting the
    final recommendation request."""

    age: int | None = Field(None, description="Patient age, or null if not found in the document")
    pathologies: list[str] = Field(default_factory=list, description="Active diagnosed conditions")
    habits: list[str] = Field(default_factory=list, description="Lifestyle habits")
    medical_history: list[str] = Field(default_factory=list, description="Past medical events")

    # Numeric vitals extracted from the document
    obs_bmi: float | None = Field(None, description="BMI, null if not in document")
    obs_systolic_bp: float | None = Field(None, description="Systolic BP (mmHg), null if not in document")
    obs_diastolic_bp: float | None = Field(None, description="Diastolic BP (mmHg), null if not in document")
    obs_phq9_score: float | None = Field(None, description="PHQ-9 score, null if not in document")
    obs_testosterone: float | None = Field(None, description="Testosterone (ng/dL), null if not in document")
    obs_smoker: int = Field(0, description="Current smoker: 1=yes, 0=no")

    # Binary condition flags extracted by the LLM
    cond_migraine_with_aura: int = Field(0)
    cond_stroke: int = Field(0)
    cond_mi: int = Field(0)
    cond_dvt: int = Field(0)
    cond_breast_cancer: int = Field(0)
    cond_lupus: int = Field(0)
    cond_thrombophilia: int = Field(0)
    cond_atrial_fibrillation: int = Field(0)
    cond_liver_disease: int = Field(0)
    cond_hypertension: int = Field(0)
    cond_migraine: int = Field(0)
    cond_gallstones: int = Field(0)
    cond_diabetes: int = Field(0)
    cond_prediabetes: int = Field(0)
    cond_epilepsy: int = Field(0)
    cond_chronic_kidney_disease: int = Field(0)
    cond_sleep_apnea: int = Field(0)
    cond_pcos: int = Field(0)
    cond_endometriosis: int = Field(0)
    cond_depression: int = Field(0)
    cond_hypothyroidism: int = Field(0)
    cond_rheumatoid_arthritis: int = Field(0)
    cond_fibromyalgia: int = Field(0)
    cond_osteoporosis: int = Field(0)
    cond_asthma: int = Field(0)

    pages_parsed: int = Field(0, description="Number of PDF pages that were parsed")
    parser_backend: str = Field("", description="Which parser was used: 'dots_ocr' or 'pymupdf'")


class PillRecommendation(BaseModel):
    """Detailed information about a recommended pill."""

    pill_id: str
    rank: int  # 1, 2, or 3
    utility_score: float
    risk_score: float = Field(0.0, description="Agent-assessed risk score (0.0 = lowest risk, 1.0 = highest)")
    predicted_discontinuation: float
    severe_risk: float
    mild_side_effect_score: float
    contraceptive_effectiveness: float
    reason_codes: list[str]

    # ── Time-series trajectory data for plotting ─────────────────────────────
    # months: [1, 2, ..., 12] — x-axis shared by all curves below
    months: list[int] = Field(
        default_factory=list,
        description="Month indices [1..N] shared by all trajectory arrays",
    )
    # symptom_probs: all 18 binary-target channels from the simulator
    #   Key names (always present when months is non-empty):
    #     still_taking          — adherence probability (1 − discontinuation curve)
    #     sym_nausea, sym_headache, sym_breast_tenderness, sym_spotting,
    #     sym_mood_worsened, sym_depression_episode, sym_anxiety,
    #     sym_libido_decreased, sym_weight_gain, sym_acne_worsened, sym_hair_loss
    #     evt_dvt, evt_pe, evt_stroke  — serious adverse events
    #     sym_bloating, sym_fatigue, sym_irregular_bleeding  (remaining targets)
    symptom_probs: dict[str, list[float]] = Field(
        default_factory=dict,
        description=(
            "Per-month probabilities for all 18 simulator binary targets. "
            "Keys: still_taking, sym_*, evt_*. Values: list of floats aligned with months."
        ),
    )
    # satisfaction: monthly score 1-10 from the satisfaction regression model
    satisfaction: list[float] = Field(
        default_factory=list,
        description="Predicted monthly satisfaction score (1-10 scale) aligned with months",
    )


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
@app.post("/api/v1/recommend", response_model=RecommendationOutput)
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
        "top3_reason_codes": None,
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
        # Numeric vitals (may be None if not found in the PDF)
        obs_bmi=patient_data.get("obs_bmi"),
        obs_systolic_bp=patient_data.get("obs_systolic_bp"),
        obs_diastolic_bp=patient_data.get("obs_diastolic_bp"),
        obs_phq9_score=patient_data.get("obs_phq9_score"),
        obs_testosterone=patient_data.get("obs_testosterone"),
        obs_smoker=int(patient_data.get("obs_smoker", 0)),
        # Binary condition flags extracted by the LLM
        cond_migraine_with_aura=int(patient_data.get("cond_migraine_with_aura", 0)),
        cond_stroke=int(patient_data.get("cond_stroke", 0)),
        cond_mi=int(patient_data.get("cond_mi", 0)),
        cond_dvt=int(patient_data.get("cond_dvt", 0)),
        cond_breast_cancer=int(patient_data.get("cond_breast_cancer", 0)),
        cond_lupus=int(patient_data.get("cond_lupus", 0)),
        cond_thrombophilia=int(patient_data.get("cond_thrombophilia", 0)),
        cond_atrial_fibrillation=int(patient_data.get("cond_atrial_fibrillation", 0)),
        cond_liver_disease=int(patient_data.get("cond_liver_disease", 0)),
        cond_hypertension=int(patient_data.get("cond_hypertension", 0)),
        cond_migraine=int(patient_data.get("cond_migraine", 0)),
        cond_gallstones=int(patient_data.get("cond_gallstones", 0)),
        cond_diabetes=int(patient_data.get("cond_diabetes", 0)),
        cond_prediabetes=int(patient_data.get("cond_prediabetes", 0)),
        cond_epilepsy=int(patient_data.get("cond_epilepsy", 0)),
        cond_chronic_kidney_disease=int(patient_data.get("cond_chronic_kidney_disease", 0)),
        cond_sleep_apnea=int(patient_data.get("cond_sleep_apnea", 0)),
        cond_pcos=int(patient_data.get("cond_pcos", 0)),
        cond_endometriosis=int(patient_data.get("cond_endometriosis", 0)),
        cond_depression=int(patient_data.get("cond_depression", 0)),
        cond_hypothyroidism=int(patient_data.get("cond_hypothyroidism", 0)),
        cond_rheumatoid_arthritis=int(patient_data.get("cond_rheumatoid_arthritis", 0)),
        cond_fibromyalgia=int(patient_data.get("cond_fibromyalgia", 0)),
        cond_osteoporosis=int(patient_data.get("cond_osteoporosis", 0)),
        cond_asthma=int(patient_data.get("cond_asthma", 0)),
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
    risk_scores = state.get("risk_scores") or {}
    top3_reason_codes = state.get("top3_reason_codes") or {}

    # Get top 3 pills by utility score
    sorted_pills = sorted(
        utility_scores.items(),
        key=lambda x: x[1],
        reverse=True,
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

        # Per-pill reason codes come from the LLM via top3_reason_codes.
        # Fall back to the global reason_codes for rank 1, or a minimal
        # placeholder for ranks 2-3 when the LLM didn't produce them.
        pill_reasons = top3_reason_codes.get(pill_id)
        if not pill_reasons:
            if rank == 1:
                pill_reasons = state.get("reason_codes") or [f"highest utility score ({utility:.3f})"]
            else:
                pill_reasons = [f"rank {rank} by utility score ({utility:.3f})"]

        # Risk score from the risk assessor LLM (0.0 = lowest risk, 1.0 = highest).
        agent_risk_score = float(
            (risk_scores.get(pill_id) or {}).get("risk_score", 0.0)
        )

        recommendations.append(PillRecommendation(
            pill_id=pill_id,
            rank=rank,
            utility_score=round(utility, 4),
            risk_score=round(agent_risk_score, 4),
            predicted_discontinuation=sim.get("discontinuation_probability", 0.0),
            severe_risk=sim.get("severe_event_probability", 0.0),
            mild_side_effect_score=sim.get("mild_side_effect_score", 0.0),
            contraceptive_effectiveness=sim.get("contraceptive_effectiveness", 0.0),
            reason_codes=pill_reasons,
            # Trajectory data for time-series plots
            months=sim.get("months", []),
            symptom_probs=sim.get("symptom_probs", {}),
            satisfaction=sim.get("satisfaction", []),
        ))

    return RecommendationOutput(
        recommendations=recommendations,
        cluster_profile=cluster,
        cluster_confidence=state.get("cluster_confidence", 0.0),
        iterations=state.get("iteration", 0),
        total_candidates_evaluated=len(utility_scores),
        selected_pill=sorted_pills[0][0],
    )
