"""
PDF Parser — converts a medical PDF into plain text using PyMuPDF.

Returns a list of page dicts, one per page:
  [{"page_no": int, "text": str, "source": "pymupdf"}, ...]

The caller (patient_extractor.py) concatenates the pages and sends them to
the LLM for field extraction.
"""

import logging

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_bytes: bytes) -> list[dict]:
    """Parse a PDF from raw bytes using PyMuPDF."""
    import pymupdf as fitz  # PyMuPDF 1.24+

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []

    for idx, page in enumerate(doc):
        text = page.get_text("text")
        results.append({"page_no": idx, "text": text.strip(), "source": "pymupdf"})

    doc.close()
    logger.info("pymupdf: extracted %d page(s)", len(results))
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def pages_to_document(pages: list[dict]) -> str:
    """
    Concatenate per-page text into a single document string, adding
    page-break markers so the LLM extractor can orient itself.
    """
    parts = [f"--- Page {p['page_no'] + 1} ---\n{p['text']}" for p in pages]
    return "\n\n".join(parts)
