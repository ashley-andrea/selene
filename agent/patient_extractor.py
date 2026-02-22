"""
Patient Extractor — maps raw document text (from pdf_parser) to the
patient_data schema understood by the agent.

The LLM is given the full document text and asked to extract:
  - age (int)
  - obs_* numeric vitals: BMI, blood pressure, PHQ-9, testosterone, smoker flag
  - pathologies (list[str])  — kept for safe_gate.py hard-constraint rules
  - habits (list[str])
  - medical_history (list[str])
  - cond_* binary flags for all 25 conditions expected by the clustering model

Fields that cannot be determined from the document are left as empty
lists / None / 0.  The extractor never invents information.

The output is a dict that can be passed directly to the validator node
as patient_data.  It is also a valid input for cluster_api.py without
any further transformation (all cond_* and obs_* fields are present).
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical data extraction assistant.
You will receive the text content of a medical record (cartella clinica / fascicolo sanitario elettronico).
Your task is to extract structured patient information and return it as a single JSON object.

Return ONLY a valid JSON object with EXACTLY these keys:

DEMOGRAPHICS & VITALS (use null / 0 if not found — never invent values):
- "age": integer (patient age in years, null if not found)
- "obs_bmi": float (body mass index, null if not found)
- "obs_systolic_bp": integer (systolic blood pressure mmHg, null if not found)
- "obs_diastolic_bp": integer (diastolic blood pressure mmHg, null if not found)
- "obs_phq9_score": integer (PHQ-9 depression screening score 0-27, null if not found)
- "obs_testosterone": float (serum testosterone ng/dL, null if not found)
- "obs_smoker": 1 if the patient is a current smoker, 0 otherwise

ACTIVE CONDITIONS (1 = condition present / diagnosed, 0 = absent or unknown):
- "cond_migraine_with_aura": 1 or 0
- "cond_stroke": 1 or 0
- "cond_mi": 1 or 0  (myocardial infarction / heart attack)
- "cond_dvt": 1 or 0  (deep vein thrombosis / venous thromboembolism)
- "cond_breast_cancer": 1 or 0
- "cond_lupus": 1 or 0  (systemic lupus erythematosus)
- "cond_thrombophilia": 1 or 0  (any inherited or acquired clotting disorder)
- "cond_atrial_fibrillation": 1 or 0
- "cond_liver_disease": 1 or 0  (hepatitis, cirrhosis, etc.)
- "cond_hypertension": 1 or 0  (elevated blood pressure / ipertensione)
- "cond_migraine": 1 or 0  (migraine without aura)
- "cond_gallstones": 1 or 0  (cholelithiasis / colelitiasi)
- "cond_diabetes": 1 or 0  (type 1 or type 2 diabetes)
- "cond_prediabetes": 1 or 0
- "cond_epilepsy": 1 or 0
- "cond_chronic_kidney_disease": 1 or 0
- "cond_sleep_apnea": 1 or 0
- "cond_pcos": 1 or 0  (polycystic ovary syndrome)
- "cond_endometriosis": 1 or 0
- "cond_depression": 1 or 0
- "cond_hypothyroidism": 1 or 0
- "cond_rheumatoid_arthritis": 1 or 0
- "cond_fibromyalgia": 1 or 0
- "cond_osteoporosis": 1 or 0
- "cond_asthma": 1 or 0

LISTS (for human-readable context — can overlap with the flags above):
- "pathologies": array of strings (active diagnosed conditions in English, lowercase)
- "habits": array of strings (e.g. ["smoking", "alcohol use", "sedentary lifestyle"])
- "medical_history": array of strings (past events, surgeries, e.g. ["dvt 2018", "appendectomy 2015"])

Rules:
- Translate from Italian or any other language to English.
- Do NOT invent information not present in the document.
- If a binary cond_* field cannot be determined, default to 0.
- If a numeric obs_* field is not in the document, use null.
- Do NOT include current medications in pathologies — only diagnosed conditions.
- Return ONLY the JSON object, no surrounding text, no markdown code fences.
"""

USER_PROMPT_TEMPLATE = """Medical record text:

{document_text}

Extract the patient data and return only the JSON object."""


# ── Public API ────────────────────────────────────────────────────────────────

def extract_patient_data(document_text: str) -> dict:
    """
    Send the document text to the LLM and return a patient_data dict.

    Returns a dict with keys: age, pathologies, habits, medical_history.
    Missing values default to None / empty list.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(document_text=document_text)
        ),
    ]

    logger.info("Patient extractor: invoking LLM on %d chars of document text", len(document_text))
    response = llm.invoke(messages)
    raw = response.content.strip()
    logger.debug("LLM extraction response:\n%s", raw)

    extracted = _parse_llm_json(raw)
    patient_data = _normalize(extracted)
    logger.info(
        "Extracted: age=%s, pathologies=%s, habits=%s, history=%s",
        patient_data.get("age"),
        patient_data.get("pathologies"),
        patient_data.get("habits"),
        patient_data.get("medical_history"),
    )
    return patient_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """
    Parse the LLM output as JSON.  Tolerant of markdown fences and leading/
    trailing whitespace that the model may add despite instructions.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s). Raw output: %s", exc, raw[:500])
        # Return safe defaults if extraction fails
        return {}


def _normalize(raw: dict) -> dict:
    """
    Normalise extracted values to match the patient_data schema.
    Ensures correct types and sensible defaults.

    Output contains:
      - age, obs_*, cond_* fields — consumed directly by cluster_api.py / safe_gate.py
      - pathologies, habits, medical_history — kept for human-readable context and
        backwards-compatible safe_gate.py rule evaluation
    """
    # ── age ───────────────────────────────────────────────────────────────
    age = raw.get("age")
    if age is not None:
        try:
            age = int(age)
        except (TypeError, ValueError):
            age = None

    def _str_list(val) -> list[str]:
        if not val or not isinstance(val, list):
            return []
        return [str(item).lower().strip() for item in val if item]

    def _int_flag(val, default: int = 0) -> int:
        """Coerce a cond_* value to 0 or 1."""
        try:
            return 1 if int(val) else 0
        except (TypeError, ValueError):
            return default

    def _float_or_none(val) -> float | None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # ── binary condition flags (25 cond_*) ───────────────────────────────
    cond_fields = [
        "cond_migraine_with_aura", "cond_stroke", "cond_mi", "cond_dvt",
        "cond_breast_cancer", "cond_lupus", "cond_thrombophilia",
        "cond_atrial_fibrillation", "cond_liver_disease", "cond_hypertension",
        "cond_migraine", "cond_gallstones", "cond_diabetes", "cond_prediabetes",
        "cond_epilepsy", "cond_chronic_kidney_disease", "cond_sleep_apnea",
        "cond_pcos", "cond_endometriosis", "cond_depression", "cond_hypothyroidism",
        "cond_rheumatoid_arthritis", "cond_fibromyalgia", "cond_osteoporosis",
        "cond_asthma",
    ]
    conditions = {f: _int_flag(raw.get(f, 0)) for f in cond_fields}

    return {
        # ── core ──────────────────────────────────────────────────────────
        "age": age,
        # ── vitals / observations ─────────────────────────────────────────
        "obs_bmi":          _float_or_none(raw.get("obs_bmi")),
        "obs_systolic_bp":  _float_or_none(raw.get("obs_systolic_bp")),
        "obs_diastolic_bp": _float_or_none(raw.get("obs_diastolic_bp")),
        "obs_phq9_score":   _float_or_none(raw.get("obs_phq9_score")),
        "obs_testosterone": _float_or_none(raw.get("obs_testosterone")),
        "obs_smoker":       _int_flag(raw.get("obs_smoker", 0)),
        # ── binary conditions ─────────────────────────────────────────────
        **conditions,
        # ── list fields (human-readable + safe_gate.py fallback) ──────────
        "pathologies":     _str_list(raw.get("pathologies")),
        "habits":          _str_list(raw.get("habits")),
        "medical_history": _str_list(raw.get("medical_history")),
    }
