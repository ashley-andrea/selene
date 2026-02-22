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
LLM agent
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

## Known Limitations

dots.ocr is a strong general-purpose parser, but a few limitations are worth knowing upfront:

**Complex tables**: the model handles most medical tables well but may struggle with very dense or multi-level header tables (e.g. detailed lab result sheets). The extracted HTML may be incomplete. The LLM extractor is instructed not to invent values, so missing table data will simply result in an omitted field rather than a wrong one.

**Scanned PDFs**: dots.ocr works on page images, so scanned documents are supported. However, very low-resolution scans (below ~150 DPI) will degrade accuracy. The parser uses 200 DPI by default which covers most clinical PDFs.

**Abbreviations and local terminology**: Italian medical abbreviations vary by hospital and region. The LLM extractor handles this reasonably well but may miss uncommon shorthand. The user review step is the safety net here.

**Inference speed**: on GPU, parsing a typical 5-10 page medical record takes 10-30 seconds depending on complexity. For now we do it locally.

**Pictures in documents**: dots.ocr does not parse images embedded in PDFs (charts, diagrams). This is an upstream library limitation, but it doesn't affect our use case since the medical fields we need are always in text or table form.

---

## What This Does NOT Change

- The agent loop, nodes, and state are completely unaffected
- The `patient_data` schema is unchanged — PDF extraction produces the same object as manual input
- The validator node receives identical input regardless of the source
- The ML model API contracts are unaffected
- The mock testing strategy is unaffected — mock the `/upload-pdf` endpoint the same way as any other during testing
