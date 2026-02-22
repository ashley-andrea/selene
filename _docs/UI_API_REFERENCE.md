# UI API Reference

All endpoints are served by the **FastAPI backend** (`main.py`).

**Base URL (local dev):** `http://localhost:8000`  
**Base URL (production):** `https://bc-recommendation-agent.onrender.com`

CORS is fully open in local dev (`*`). In production the backend reads `ALLOWED_ORIGINS` from its environment and restricts to those origins (set to your Vercel URL in the Render dashboard env vars).

---

## Endpoints

### 1. `POST /api/v1/recommend`

Run the full recommendation pipeline for a patient and receive the top-3 pill suggestions.

#### Request body — `application/json`

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `age` | `int` | ✅ | 15 – 60 | Patient age in years |
| `pathologies` | `string[]` | no | — | Active diagnosed conditions, e.g. `["hypertension", "migraine"]` |
| `habits` | `string[]` | no | — | Lifestyle habits, e.g. `["smoking"]` |
| `medical_history` | `string[]` | no | — | Past events, e.g. `["DVT"]` |
| `obs_bmi` | `float \| null` | no | 10.0 – 60.0 | BMI |
| `obs_systolic_bp` | `float \| null` | no | 70 – 220 | Systolic blood pressure (mmHg) |
| `obs_diastolic_bp` | `float \| null` | no | 40 – 140 | Diastolic blood pressure (mmHg) |
| `obs_phq9_score` | `float \| null` | no | 0 – 27 | PHQ-9 depression score |
| `obs_testosterone` | `float \| null` | no | 0 – 300 | Testosterone (ng/dL) |
| `obs_smoker` | `int` | no | 0 or 1 | 1 = current smoker |
| `obs_pain_score` | `float` | no | 0.0 – 10.0 | Pain score |
| `cond_migraine_with_aura` | `int` | no | 0 or 1 | — |
| `cond_stroke` | `int` | no | 0 or 1 | — |
| `cond_mi` | `int` | no | 0 or 1 | — |
| `cond_dvt` | `int` | no | 0 or 1 | — |
| `cond_breast_cancer` | `int` | no | 0 or 1 | — |
| `cond_lupus` | `int` | no | 0 or 1 | — |
| `cond_thrombophilia` | `int` | no | 0 or 1 | — |
| `cond_atrial_fibrillation` | `int` | no | 0 or 1 | — |
| `cond_liver_disease` | `int` | no | 0 or 1 | — |
| `cond_hypertension` | `int` | no | 0 or 1 | — |
| `cond_migraine` | `int` | no | 0 or 1 | — |
| `cond_gallstones` | `int` | no | 0 or 1 | — |
| `cond_diabetes` | `int` | no | 0 or 1 | — |
| `cond_prediabetes` | `int` | no | 0 or 1 | — |
| `cond_epilepsy` | `int` | no | 0 or 1 | — |
| `cond_chronic_kidney_disease` | `int` | no | 0 or 1 | — |
| `cond_sleep_apnea` | `int` | no | 0 or 1 | — |
| `cond_pcos` | `int` | no | 0 or 1 | — |
| `cond_endometriosis` | `int` | no | 0 or 1 | — |
| `cond_depression` | `int` | no | 0 or 1 | — |
| `cond_hypothyroidism` | `int` | no | 0 or 1 | — |
| `cond_rheumatoid_arthritis` | `int` | no | 0 or 1 | — |
| `cond_fibromyalgia` | `int` | no | 0 or 1 | — |
| `cond_osteoporosis` | `int` | no | 0 or 1 | — |
| `cond_asthma` | `int` | no | 0 or 1 | — |
| `med_ever_ocp` | `int` | no | 0 or 1 | Has ever taken combined OCP |
| `med_current_combined_ocp` | `int` | no | 0 or 1 | Currently on combined OCP |
| `med_current_minipill` | `int` | no | 0 or 1 | Currently on mini-pill |
| `has_absolute_contraindication_combined_oc` | `int` | no | 0 or 1 | Pre-computed contraindication flag |

> **Minimal viable request** — the backend only strictly requires `age`. All `cond_*` / `obs_*` fields default to `0` / `null` and the ML models impute missing values. Provide as many as you have.

**Example minimal request:**
```json
{
  "age": 28,
  "pathologies": ["migraine"],
  "habits": ["smoking"],
  "medical_history": []
}
```

**Example full request (from PDF extraction):**
```json
{
  "age": 34,
  "pathologies": ["hypertension", "pcos"],
  "habits": [],
  "medical_history": ["DVT"],
  "obs_bmi": 24.5,
  "obs_systolic_bp": 138,
  "obs_diastolic_bp": 88,
  "obs_smoker": 0,
  "cond_hypertension": 1,
  "cond_pcos": 1,
  "cond_dvt": 1
}
```

#### Response body — `200 OK`

```json
{
  "recommendations": [
    {
      "pill_id": "DRSP_30EE",
      "rank": 1,
      "utility_score": 0.847,
      "predicted_discontinuation": 0.12,
      "severe_risk": 0.0021,
      "mild_side_effect_score": 0.31,
      "contraceptive_effectiveness": 0.97,
      "reason_codes": [
        "lowest risk in cluster",
        "high predicted adherence",
        "matches patient profile"
      ],
      "months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      "symptom_probs": {
        "still_taking": [0.95, 0.93, 0.91, 0.90, 0.89, 0.88, 0.87, 0.87, 0.86, 0.86, 0.85, 0.85],
        "sym_nausea": [0.12, 0.09, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03],
        "sym_headache": [...],
        "sym_breast_tenderness": [...],
        "sym_spotting": [...],
        "sym_mood_worsened": [...],
        "sym_depression_episode": [...],
        "sym_anxiety": [...],
        "sym_libido_decreased": [...],
        "sym_weight_gain": [...],
        "sym_acne_worsened": [...],
        "sym_hair_loss": [...],
        "evt_dvt": [...],
        "evt_pe": [...],
        "evt_stroke": [...],
        "sym_bloating": [...],
        "sym_fatigue": [...],
        "sym_irregular_bleeding": [...]
      },
      "satisfaction": [7.2, 7.4, 7.5, 7.6, 7.6, 7.7, 7.7, 7.8, 7.8, 7.8, 7.9, 7.9]
    },
    {
      "pill_id": "LNG_30EE",
      "rank": 2,
      "utility_score": 0.791,
      ...
    },
    {
      "pill_id": "NET_35EE",
      "rank": 3,
      "utility_score": 0.743,
      ...
    }
  ],
  "cluster_profile": "cluster_3",
  "cluster_confidence": 0.88,
  "iterations": 2,
  "total_candidates_evaluated": 14,
  "selected_pill": "DRSP_30EE"
}
```

| Response field | Type | Description |
|---|---|---|
| `recommendations` | `PillRecommendation[]` | Ordered list of top 3 pills (rank 1 = best) |
| `recommendations[].pill_id` | `string` | Pill identifier |
| `recommendations[].rank` | `int` | 1, 2, or 3 |
| `recommendations[].utility_score` | `float` | Agent's composite utility score (higher = better) |
| `recommendations[].predicted_discontinuation` | `float` | Probability [0–1] patient stops the pill within the simulation window |
| `recommendations[].severe_risk` | `float` | Probability [0–1] of a serious adverse event (DVT, stroke, PE) |
| `recommendations[].mild_side_effect_score` | `float` | Composite mild side-effect score [0–1] |
| `recommendations[].contraceptive_effectiveness` | `float` | Predicted contraceptive effectiveness [0–1] |
| `recommendations[].reason_codes` | `string[]` | Human-readable reasons for this recommendation |
| `recommendations[].months` | `int[]` | Month indices `[1, 2, ..., N]` shared by all trajectory arrays |
| `recommendations[].symptom_probs` | `object` | Per-month probability per symptom/event channel (keys listed below) |
| `recommendations[].satisfaction` | `float[]` | Predicted monthly satisfaction score (1–10 scale) |
| `cluster_profile` | `string` | Patient's risk cluster assigned by the ML model (e.g. `"cluster_3"`) |
| `cluster_confidence` | `float` | Confidence of the cluster assignment [0–1] |
| `iterations` | `int` | Number of agent loop iterations run |
| `total_candidates_evaluated` | `int` | Total number of pills scored during the run |
| `selected_pill` | `string` | Shortcut — same as `recommendations[0].pill_id` |

**`symptom_probs` keys** (all present when `months` is non-empty):

| Key | Description |
|---|---|
| `still_taking` | Adherence probability (= 1 − cumulative discontinuation curve) |
| `sym_nausea` | Nausea |
| `sym_headache` | Headache |
| `sym_breast_tenderness` | Breast tenderness |
| `sym_spotting` | Spotting |
| `sym_mood_worsened` | Mood worsening |
| `sym_depression_episode` | Depression episode |
| `sym_anxiety` | Anxiety |
| `sym_libido_decreased` | Decreased libido |
| `sym_weight_gain` | Weight gain |
| `sym_acne_worsened` | Acne worsening |
| `sym_hair_loss` | Hair loss |
| `evt_dvt` | Deep vein thrombosis event |
| `evt_pe` | Pulmonary embolism event |
| `evt_stroke` | Stroke event |
| `sym_bloating` | Bloating |
| `sym_fatigue` | Fatigue |
| `sym_irregular_bleeding` | Irregular bleeding |

#### Error responses

| Status | When |
|---|---|
| `422` | Validation failure (age out of range, etc.) or agent-level rejection |
| `500` | Internal agent error |

```json
{ "detail": "human-readable error message" }
```

---

### 2. `POST /api/v1/patient/upload-pdf`

Upload a medical PDF and extract patient data from it. Returns a pre-filled patient object that the UI can use to populate the intake form before the user confirms and calls `/recommend`.

#### Request — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | `File` | The PDF file to upload (`Content-Type: application/pdf`) |

**Example (fetch):**
```js
const formData = new FormData();
formData.append("file", pdfFile);

const res = await fetch("/api/v1/patient/upload-pdf", {
  method: "POST",
  body: formData,
});
const extracted = await res.json();
```

#### Response body — `200 OK`

Returns a `PDFExtractionResult` object. All fields may be `null` / default if the information was not found in the document. **The user should always review extracted values before submission.**

```json
{
  "age": 34,
  "pathologies": ["hypertension", "migraine"],
  "habits": ["smoking"],
  "medical_history": ["DVT"],
  "obs_bmi": 26.1,
  "obs_systolic_bp": 138.0,
  "obs_diastolic_bp": 88.0,
  "obs_phq9_score": null,
  "obs_testosterone": null,
  "obs_smoker": 1,
  "cond_migraine_with_aura": 0,
  "cond_stroke": 0,
  "cond_mi": 0,
  "cond_dvt": 1,
  "cond_breast_cancer": 0,
  "cond_lupus": 0,
  "cond_thrombophilia": 0,
  "cond_atrial_fibrillation": 0,
  "cond_liver_disease": 0,
  "cond_hypertension": 1,
  "cond_migraine": 1,
  "cond_gallstones": 0,
  "cond_diabetes": 0,
  "cond_prediabetes": 0,
  "cond_epilepsy": 0,
  "cond_chronic_kidney_disease": 0,
  "cond_sleep_apnea": 0,
  "cond_pcos": 0,
  "cond_endometriosis": 0,
  "cond_depression": 0,
  "cond_hypothyroidism": 0,
  "cond_rheumatoid_arthritis": 0,
  "cond_fibromyalgia": 0,
  "cond_osteoporosis": 0,
  "cond_asthma": 0,
  "pages_parsed": 5,
  "parser_backend": "pymupdf"
}
```

| Field | Type | Description |
|---|---|---|
| `age` | `int \| null` | Extracted age, or `null` if not found |
| `pathologies` | `string[]` | Extracted conditions |
| `habits` | `string[]` | Extracted habits |
| `medical_history` | `string[]` | Extracted past events |
| `obs_*` | `float \| null` | Numeric vitals; `null` if not found in document |
| `obs_smoker` | `int` | 1 = smoker, 0 = not (defaults to 0) |
| `cond_*` | `int` | Binary flags; default `0` |
| `pages_parsed` | `int` | Number of PDF pages processed |
| `parser_backend` | `string` | `"dots_ocr"` or `"pymupdf"` |

#### Error responses

| Status | When |
|---|---|
| `422` | File is not a valid PDF, or PDF has no parseable pages |
| `500` | PDF parsing or LLM extraction failed |

---

### 3. `GET /health`

Basic health check. Use this to confirm the backend is reachable.

#### Response — `200 OK`

```json
{
  "status": "ok",
  "service": "bc-recommendation-agent",
  "llm_provider": "crusoe"
}
```

---

### 4. `GET /pills`

Returns the full pill reference database. Intended for development and debugging only — not needed for the core user flow.

#### Response — `200 OK`

Array of pill objects from `data/pill_reference_db.csv`. Each object contains fields like `combo_id`, `progestin`, `estrogen_dose`, `pill_type`, etc.

---

## Recommended UI flow

```
1. (Optional) User uploads PDF  →  POST /api/v1/patient/upload-pdf
                                    → pre-fill intake form with response

2. User reviews / fills intake form

3. Submit form                  →  POST /api/v1/recommend
                                    → display top-3 recommendations
                                    → render symptom_probs as time-series charts
                                    → render satisfaction as time-series chart
```

---

## Interactive API docs

The FastAPI backend auto-generates full interactive docs at:

**Local dev:**
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

**Production (live):**
- **Swagger UI:** `https://bc-recommendation-agent.onrender.com/docs`
- **ReDoc:** `https://bc-recommendation-agent.onrender.com/redoc`

These are the fastest way to explore request/response shapes and run test calls during development.
