"""
Patient Extractor — maps raw document text (from pdf_parser) to the
patient_data schema understood by the agent.

The LLM is given the full document text and asked to extract:
  - age (int)
  - pathologies (list[str])
  - habits (list[str])
  - medical_history (list[str])

Fields that cannot be determined from the document are left as empty
lists / None.  The extractor never invents information.

The output is a dict that can be passed directly to the validator node
as patient_data.
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
Your task is to extract structured patient information and return it as a JSON object.

Return ONLY a valid JSON object with these keys:
- "age": integer (patient age in years, null if not found)
- "pathologies": array of strings (active diagnosed conditions, e.g. ["hypertension", "type 2 diabetes"])
- "habits": array of strings (lifestyle habits relevant to health, e.g. ["smoking", "alcohol use", "sedentary lifestyle"])
- "medical_history": array of strings (past medical events, procedures, surgeries, e.g. ["appendectomy 2015", "dvt 2018"])

Rules:
- Use English, lowercase, concise terms (translate from Italian or other languages if needed).
- Do NOT invent information. If a field is not present in the document, return an empty array (or null for age).
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
    """
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

    return {
        "age": age,
        "pathologies": _str_list(raw.get("pathologies")),
        "habits": _str_list(raw.get("habits")),
        "medical_history": _str_list(raw.get("medical_history")),
    }
