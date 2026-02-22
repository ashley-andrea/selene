# Medical PDF Upload Feature

## Overview

During the onboarding flow, users can optionally upload their medical record — in Italy this is the **cartella clinica** or the **fascicolo sanitario elettronico (FSE)**. These documents already contain structured medical information: diagnoses, past procedures, current medications, and relevant history. Parsing them directly means the user doesn't have to manually fill in every field, reduces input errors, and produces a richer, more accurate patient profile for the recommendation agent.

This feature is **optional**. Users who don't have a digital medical record, or who prefer not to upload one, continue through the standard manual intake form. The PDF upload is an enrichment layer on top of that form, not a replacement for it.

---

## Why dots.ocr

The library chosen for PDF parsing is **[rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)**.

Italian medical PDFs are non-trivial to parse. They contain mixed layouts — flowing text sections, lab result tables, medication lists, header/footer noise — often with scanned content or complex formatting. A naive text extraction library (e.g. PyPDF2, pdfplumber) would produce unstructured text that is hard to reason about reliably.

dots.ocr is a vision-language model (1.7B parameters) that performs layout detection and content recognition in a single pass. It produces structured JSON output identifying each element's category (text block, table, section header, etc.) and its content in clean Markdown or HTML. This structured output is what makes the downstream extraction step tractable.

Key reasons it fits this use case:

- **Multilingual support**: explicitly handles non-English documents and achieves strong performance on languages beyond English and Chinese, which matters for Italian clinical language
- **Table recognition**: lab results and medication lists in medical records are typically tables — dots.ocr handles these well, outputting clean HTML tables
- **Layout awareness**: it identifies section headers and reading order, which lets the extraction step know which section of the document it's reading (e.g. "Diagnosi", "Farmaci in uso", "Anamnesi")
- **Structured JSON output**: each detected element has a category and content, making programmatic extraction straightforward
- **MIT licensed**: no legal complexity for a hackathon project

---

## Feature Architecture

The PDF upload feature sits entirely in the **FastAPI backend**, upstream of the agent. It runs once at onboarding, produces a structured `patient_data` object, and hands it to the `validator.py` node exactly as if the user had filled in the form manually. The agent has no knowledge of whether data came from a PDF or a form.

```
User uploads PDF
       │
       ▼
FastAPI endpoint: POST /api/v1/patient/upload-pdf
       │
       ▼
PDF Parser (dots.ocr)
  → converts PDF pages to images
  → runs layout + OCR model
  → returns structured JSON with categorized elements
       │
       ▼
Medical Extractor (LLM-assisted)
  → reads structured JSON from dots.ocr
  → maps recognized fields to patient_data schema
  → fills in: age, pathologies, habits, medical_history
       │
       ▼
Merge with manual form data
  → PDF-extracted values pre-fill the form
  → user reviews and confirms before submission
       │
       ▼
Standard patient_data object → Validator node → Agent loop
```

---

For deployment, dots.ocr recommends vLLM for serving:

```bash
vllm serve rednote-hilab/dots.ocr \
  --trust-remote-code \
  --async-scheduling \
  --gpu-memory-utilization 0.95
```

The backend then calls it via the vLLM OpenAI-compatible API endpoint. This keeps the model server separate from the FastAPI backend and is the recommended production setup.

For local development without a GPU, use the HuggingFace CPU inference path (significantly slower, but functional for testing the pipeline):

```bash
# In dots_ocr/parser.py calls, add use_hf=True
```

---

## Implementation

### Step 1 — PDF Parser (`pdf_intake/parser.py`)

This module receives the uploaded PDF file, runs it through dots.ocr, and returns the structured JSON output.

```python
# backend/pdf_intake/parser.py
"""
Wraps dots.ocr to parse an uploaded medical PDF into structured layout data.
Uses the vLLM server for inference (recommended) or HuggingFace as fallback.

dots.ocr output is a JSON array of layout elements, each with:
  - category: one of Text, Table, Section-header, Title, List-item, etc.
  - bbox: [x1, y1, x2, y2]
  - text: extracted content (Markdown for text, HTML for tables)
"""
import os
import tempfile
from pathlib import Path
from dots_ocr.parser import DotsOCRParser

VLLM_BASE_URL = os.getenv("DOTS_OCR_VLLM_URL", "http://localhost:8000/v1")
USE_HF = os.getenv("DOTS_OCR_USE_HF", "false").lower() == "true"

# Recommended prompt: full layout parsing (text + tables + section headers)
PROMPT_MODE = "prompt_layout_all_en"


def parse_medical_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Accepts raw PDF bytes, runs dots.ocr, and returns the structured layout JSON.

    Returns a list of element dicts:
    [
      {"category": "Section-header", "text": "Diagnosi", "bbox": [...]},
      {"category": "Text", "text": "Ipertensione arteriosa...", "bbox": [...]},
      {"category": "Table", "text": "<table>...</table>", "bbox": [...]},
      ...
    ]
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        parser = DotsOCRParser(
            base_url=VLLM_BASE_URL,
            use_hf=USE_HF
        )
        # num_thread controls parallelism for multi-page PDFs
        result = parser.parse(
            tmp_path,
            prompt_mode=PROMPT_MODE,
            num_thread=8
        )
        # result.cells is the list of layout elements
        return result.cells if hasattr(result, "cells") else result

    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

### Step 2 — Medical Extractor (`pdf_intake/extractor.py`)

dots.ocr gives us clean structured text from the document. The extractor then maps that content to the `patient_data` fields the agent expects. This step uses the LLM (via the same factory as the agent) to handle the variability of medical language — different hospitals use different terminology, abbreviations, and section naming conventions.

```python
# backend/pdf_intake/extractor.py
"""
Takes dots.ocr structured output and extracts patient_data fields using
an LLM. The LLM reads the parsed document content and maps it to the
known schema fields.

This is a one-shot extraction — not an agent loop. The LLM is given the
full parsed document text and asked to return a single structured JSON object.
"""
import json
from agent.llm import get_llm
from langchain_core.messages import HumanMessage

# Fields we want to extract — must match patient_data schema
TARGET_FIELDS = {
    "age": "integer — patient age in years",
    "pathologies": "array of strings — active diagnosed conditions, lowercase",
    "habits": "array of strings — from: smoking, alcohol, sedentary_lifestyle",
    "medical_history": "array of strings — past medical events, lowercase"
}

# Known Italian medical section names that map to our fields
SECTION_HINTS = """
Italian medical documents may use these section names:
- Age/Date of birth: "Data di nascita", "Età", "Anno di nascita"
- Diagnoses: "Diagnosi", "Diagnosi principale", "Problemi attivi", "Patologie"
- Medications: "Farmaci", "Terapia in corso", "Farmaci prescritti"
- Medical history: "Anamnesi", "Anamnesi patologica", "Storia clinica", "Precedenti"
- Habits: "Abitudini", "Stile di vita", "Fumo", "Alcol"
"""


def extract_patient_data(layout_elements: list[dict]) -> dict:
    """
    Takes the structured layout JSON from dots.ocr and extracts patient_data fields.
    Returns a partial patient_data dict — only fields that could be found.
    Missing fields will be filled by the user in the confirmation form.
    """
    # Reconstruct readable document text from layout elements, preserving structure
    doc_text = _layout_to_text(layout_elements)

    llm = get_llm()
    prompt = f"""
You are extracting structured medical information from an Italian medical record (cartella clinica or fascicolo sanitario elettronico).

{SECTION_HINTS}

The document content, parsed from PDF with layout awareness, is:

---
{doc_text}
---

Extract the following fields and return ONLY a JSON object with these keys:
{json.dumps(TARGET_FIELDS, indent=2)}

Rules:
- For pathologies and medical_history: use lowercase English medical terms
- For habits: only include values from the allowed list: smoking, alcohol, sedentary_lifestyle
- If a field cannot be found in the document, omit it from the JSON entirely
- Do not invent or infer values that are not explicitly stated in the document
- Age should be an integer (compute from date of birth if needed, current year is 2026)

Return ONLY the JSON object, no explanation, no markdown.
"""
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        extracted = json.loads(response.content)
    except json.JSONDecodeError:
        # If parsing fails, return empty dict — user fills everything manually
        extracted = {}

    return extracted


def _layout_to_text(elements: list[dict]) -> str:
    """
    Converts dots.ocr layout elements back to readable text,
    preserving section structure for the LLM to reason about.
    """
    lines = []
    for el in elements:
        category = el.get("category", "")
        text = el.get("text", "").strip()

        if not text:
            continue

        if category in ("Title", "Section-header"):
            lines.append(f"\n## {text}\n")
        elif category == "Table":
            lines.append(f"\n[TABLE]\n{text}\n[/TABLE]\n")
        elif category == "List-item":
            lines.append(f"- {text}")
        elif category in ("Page-header", "Page-footer"):
            # Skip headers and footers — noise for our purposes
            continue
        else:
            lines.append(text)

    return "\n".join(lines)
```

### Step 3 — Merger (`pdf_intake/merger.py`)

The extracted data pre-fills the intake form. The user sees what was extracted and can correct or add to it before confirming. This is important — the model may miss fields or extract incorrectly, and the user must always have final say over their medical data before it drives a recommendation.

```python
# backend/pdf_intake/merger.py
"""
Merges PDF-extracted patient data with manual form input.
PDF extraction pre-fills the form; manual input takes precedence
if the user edits any field.
"""

def merge_patient_data(pdf_extracted: dict, manual_input: dict) -> dict:
    """
    Merges two partial patient_data dicts.
    Manual input always wins for scalar fields (age).
    For array fields (pathologies, habits, medical_history),
    merges and deduplicates — preserving all information from both sources.
    """
    merged = dict(pdf_extracted)

    for key, value in manual_input.items():
        if not value:
            continue
        if isinstance(value, list) and key in merged:
            # Merge arrays, deduplicate, preserve order
            existing = merged.get(key, [])
            merged[key] = list(dict.fromkeys(existing + value))
        else:
            # Scalar fields: manual input wins
            merged[key] = value

    return merged
```

### Step 4 — FastAPI Endpoint (`routers/patient.py`)

```python
# backend/routers/patient.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pdf_intake.parser import parse_medical_pdf
from pdf_intake.extractor import extract_patient_data

router = APIRouter(prefix="/api/v1/patient", tags=["patient"])

MAX_PDF_SIZE_MB = 20

@router.post("/upload-pdf")
async def upload_medical_pdf(file: UploadFile = File(...)):
    """
    Accepts a medical PDF upload, parses it with dots.ocr,
    and returns pre-extracted patient data fields for form pre-fill.

    The frontend uses this response to pre-populate the intake form.
    The user reviews and confirms before the data is submitted.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_PDF_SIZE_MB}MB."
        )

    try:
        layout_elements = parse_medical_pdf(pdf_bytes)
        extracted_data = extract_patient_data(layout_elements)
    except Exception as e:
        # Parsing failed — return empty extraction, user fills manually
        return {
            "extracted": {},
            "warning": "Could not parse the document automatically. Please fill in your details manually.",
            "error_detail": str(e)
        }

    return {
        "extracted": extracted_data,
        "fields_found": list(extracted_data.keys()),
        "warning": None
    }
```

---

## Frontend Flow

The user experience should be:

1. During onboarding, a clearly optional upload area appears: *"Hai la tua cartella clinica in formato PDF? Caricala per compilare il modulo automaticamente."*
2. User uploads the PDF. A loading indicator appears while parsing runs (this takes a few seconds).
3. The intake form is pre-filled with the extracted values. Each pre-filled field is visually marked (e.g. a small tag: *"Estratto dal documento"*).
4. The user reviews every field, corrects anything wrong, and adds anything missing.
5. User confirms and submits. From this point, the flow is identical to the manual path.

The key UX principle: **the user always has final control**. The PDF extraction is a convenience, not an authority. No extracted value should ever be submitted without the user having reviewed it.

---

## Environment Configuration

```bash
# dots.ocr vLLM server URL (set if using vLLM deployment)
DOTS_OCR_VLLM_URL=http://localhost:8000/v1

# Set to true to use HuggingFace inference instead of vLLM (slower, CPU-compatible)
DOTS_OCR_USE_HF=false
```

---

## Known Limitations

dots.ocr is a strong general-purpose parser, but a few limitations are worth knowing upfront:

**Complex tables**: the model handles most medical tables well but may struggle with very dense or multi-level header tables (e.g. detailed lab result sheets). The extracted HTML may be incomplete. The LLM extractor is instructed not to invent values, so missing table data will simply result in an omitted field rather than a wrong one.

**Scanned PDFs**: dots.ocr works on page images, so scanned documents are supported. However, very low-resolution scans (below ~150 DPI) will degrade accuracy. The parser uses 200 DPI by default which covers most clinical PDFs.

**Abbreviations and local terminology**: Italian medical abbreviations vary by hospital and region. The LLM extractor handles this reasonably well but may miss uncommon shorthand. The user review step is the safety net here.

**Inference speed**: on GPU, parsing a typical 5-10 page medical record takes 10-30 seconds depending on complexity. On CPU it can take several minutes. For the hackathon demo, set expectations accordingly and show a clear loading state in the UI.

**Pictures in documents**: dots.ocr does not parse images embedded in PDFs (charts, diagrams). This is an upstream library limitation, but it doesn't affect our use case since the medical fields we need are always in text or table form.

---

## What This Does NOT Change

- The agent loop, nodes, and state are completely unaffected
- The `patient_data` schema is unchanged — PDF extraction produces the same object as manual input
- The validator node receives identical input regardless of the source
- The ML model API contracts are unaffected
- The mock testing strategy is unaffected — mock the `/upload-pdf` endpoint the same way as any other during testing

---

## Dependencies to Add

```bash
# dots.ocr (install from source as described above)
# No pip package — install with: pip install -e /path/to/dots.ocr

# Additional backend dependencies
pip install python-multipart   # Required for FastAPI file uploads
pip install pymupdf            # Used internally by dots.ocr for PDF→image conversion
```

Add to `pyproject.toml`:
```toml
python-multipart = "^0.0.9"
pymupdf = "^1.24"
```