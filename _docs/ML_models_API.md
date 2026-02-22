# ML Model API Contract

## Purpose

This document defines the HTTP API contract between the agent layer and the two ML models hosted on Red Hat OpenShift. Both the agent team and the ML team must treat this document as the source of truth. Any change to a request or response schema must be updated here first and communicated to both teams before implementation changes.

The two models are:

- **Cluster Model** — assigns the patient to a population cluster
- **Simulator Model** — predicts patient-specific outcomes for a single candidate pill

---

## Contract Status — READ THIS FIRST

⚠️ **This contract is provisional.**

The ML team is still finalizing model architecture, input features, and output formats. The schemas defined in this document represent our best current understanding and are used as a working baseline so the agent team can build and test without being blocked. They are not final.

**What this means in practice:**

Fields may be added, renamed, or removed. The ML team may determine that the models need different inputs than what is defined here — for example, additional patient features, a different representation of pathologies, or extra output metrics. The simulation time horizon is not yet confirmed, which may affect the output fields entirely.

**How we handle this on the agent side:**

The agent is deliberately architected so that schema changes are cheap to absorb. All interaction with the ML models is isolated to exactly two files:

```
agent/tools/cluster_api.py
agent/tools/simulator_api.py
```

If the ML team changes the request or response schema, only these two files need to be updated. The graph, the nodes, the state, the LLM calls, and the utility logic are all insulated from API changes. The mock server (`tests/mock_server.py`) will also need updating when schemas change, but nothing else.

**Change process:**

When the ML team decides to change any part of the schema, the steps are:
1. Update this document first
2. Notify the agent team
3. Agent team updates `cluster_api.py` or `simulator_api.py` and the mock server
4. Re-run integration tests to confirm alignment

Fields currently marked with 🔄 in this document are explicitly expected to change. Fields with no marker are considered relatively stable but not guaranteed.

---

## Base URL

```
https://<openshift-host>/api/v1
```

The full `<openshift-host>` will be provided by the ML team once the OpenShift environment is provisioned. It is configured via environment variable in the agent:

```bash
MODEL_API_BASE_URL=https://<openshift-host>/api/v1
```

The two derived URLs used by the agent are then:

```bash
CLUSTER_API_URL=${MODEL_API_BASE_URL}/cluster/predict
SIMULATOR_API_URL=${MODEL_API_BASE_URL}/simulator/simulate
```

---

## Authentication

⚠️ **Not yet decided.** The ML team must confirm the authentication method before either team implements the HTTP clients. Two options are under consideration:

**Option A — API Key in header**
```
X-API-Key: <key>
```

**Option B — Bearer token**
```
Authorization: Bearer <token>
```

Until this is confirmed, the mock server runs with no authentication. When confirmed, the agent's tool clients (`cluster_api.py` and `simulator_api.py`) and the mock server will be updated in a single pass. All other code is unaffected.

---

## General Conventions

- All requests and responses are `application/json`
- All endpoints accept only `POST`
- The agent sends requests with a `timeout` of 15 seconds — the models must respond within this window
- HTTP `200` always means a valid response was returned
- All errors return a structured JSON body (see Error Format below)
- Field names use `snake_case` throughout

---

## Model 1 — Cluster Model

### Endpoint

```
POST /api/v1/cluster/predict
```

### Purpose

Assigns the patient to one of the pre-trained population clusters. The agent uses the cluster profile to apply relative risk rules and to weight candidate selection. The confidence score is used to detect low-confidence assignments, which trigger an LLM-driven weight adjustment.

### Request

```json
{
  "patient": {
    "age": 28,
    "cond_migraine_with_aura": 0,
    "cond_stroke": 0,
    "cond_mi": 0,
    "cond_dvt": 0,
    "cond_breast_cancer": 0,
    "cond_lupus": 0,
    "cond_thrombophilia": 0,
    "cond_atrial_fibrillation": 0,
    "cond_liver_disease": 0,
    "cond_hypertension": 0,
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
    "obs_bmi": 24.5,
    "obs_systolic_bp": 118,
    "obs_diastolic_bp": 75,
    "obs_phq9_score": 3,
    "obs_pain_score": 2,
    "obs_testosterone": 45.0,
    "obs_smoker": 1,
    "med_ever_ocp": 0,
    "med_current_combined_ocp": 0,
    "med_current_minipill": 0,
    "has_absolute_contraindication_combined_oc": 0
  }
}
```

#### Request Field Reference

| Field | Type | Required | Stable? | Description |
|---|---|---|---|---|
| `patient` | object | Yes | ✅ | Top-level wrapper for all patient data |
| `patient.age` | integer/float | Yes | ✅ | Patient age in years. Range: 15-55 |
| `patient.cond_*` | integer (0/1) | Yes | ✅ | Binary indicators for medical conditions. 1=present, 0=absent |
| `patient.obs_bmi` | float | Yes | ✅ | Body Mass Index |
| `patient.obs_systolic_bp` | integer/float | Yes | ✅ | Systolic blood pressure (mmHg) |
| `patient.obs_diastolic_bp` | integer/float | Yes | ✅ | Diastolic blood pressure (mmHg) |
| `patient.obs_phq9_score` | integer/float | Yes | ✅ | PHQ-9 depression screening score (0-27) |
| `patient.obs_pain_score` | integer/float | Yes | ✅ | Pain assessment score |
| `patient.obs_testosterone` | float | Yes | ✅ | Testosterone level (ng/dL) |
| `patient.obs_smoker` | integer (0/1) | Yes | ✅ | Binary indicator: 1=smoker, 0=non-smoker |
| `patient.med_ever_ocp` | integer (0/1) | Yes | ✅ | Ever used oral contraceptive pills |
| `patient.med_current_combined_ocp` | integer (0/1) | Yes | ✅ | Currently using combined OCP |
| `patient.med_current_minipill` | integer (0/1) | Yes | ✅ | Currently using progestin-only pill |
| `patient.has_absolute_contraindication_combined_oc` | integer (0/1) | Yes | ✅ | Has absolute contraindication to combined OCs |

#### Valid Condition Fields (cond_*)

All condition fields are binary (0 or 1):
- `cond_migraine_with_aura`: Migraine with aura
- `cond_stroke`: History of stroke
- `cond_mi`: Myocardial infarction (heart attack)
- `cond_dvt`: Deep vein thrombosis
- `cond_breast_cancer`: Breast cancer
- `cond_lupus`: Systemic lupus erythematosus
- `cond_thrombophilia`: Blood clotting disorder
- `cond_atrial_fibrillation`: Atrial fibrillation
- `cond_liver_disease`: Liver disease
- `cond_hypertension`: High blood pressure
- `cond_migraine`: Migraine (without aura)
- `cond_gallstones`: Gallstones
- `cond_diabetes`: Diabetes mellitus
- `cond_prediabetes`: Prediabetes
- `cond_epilepsy`: Epilepsy
- `cond_chronic_kidney_disease`: Chronic kidney disease
- `cond_sleep_apnea`: Sleep apnea
- `cond_pcos`: Polycystic ovary syndrome
- `cond_endometriosis`: Endometriosis
- `cond_depression`: Depression
- `cond_hypothyroidism`: Hypothyroidism
- `cond_rheumatoid_arthritis`: Rheumatoid arthritis
- `cond_fibromyalgia`: Fibromyalgia
- `cond_osteoporosis`: Osteoporosis
- `cond_asthma`: Asthma



### Response

```json
{
  "cluster_profile": "cluster_2",
  "cluster_confidence": 0.88
}
```

#### Response Field Reference

| Field | Type | Stable? | Description |
|---|---|---|---|
| `cluster_profile` | string | ✅ | Cluster identifier. Format: `cluster_N` where N is an integer |
| `cluster_confidence` | float | ✅ | Model confidence in this assignment. Range: 0.0-1.0 |

The response shape is considered stable — the agent's convergence and weight adjustment logic depends directly on both fields. If the ML team needs to change the output format, this must be flagged early as it affects node logic, not just the API client.

#### Confidence Threshold Behavior

The agent treats `cluster_confidence` below `0.70` as a low-confidence assignment and activates a weight adjustment step. The model does not need to know this — it simply returns the score. The threshold is defined in the agent and may be tuned independently.

### Example — Full Request/Response

```
POST /api/v1/cluster/predict
Content-Type: application/json

{
  "patient": {
    "age": 42,
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
    "obs_bmi": 28.3,
    "obs_systolic_bp": 135,
    "obs_diastolic_bp": 88,
    "obs_phq9_score": 5,
    "obs_pain_score": 3,
    "obs_testosterone": 42.0,
    "obs_smoker": 0,
    "med_ever_ocp": 1,
    "med_current_combined_ocp": 0,
    "med_current_minipill": 0,
    "has_absolute_contraindication_combined_oc": 0
  }
}

HTTP 200 OK
{
  "cluster_profile": "cluster_4",
  "cluster_confidence": 0.63
}
```

---

## Model 2 — Simulator Model

### Endpoint

```
POST /api/v1/simulator/simulate
```

### Purpose

Predicts patient-specific outcomes for a single candidate pill over a defined time horizon. The agent calls this endpoint once per candidate in the pool. Calls are made concurrently via `asyncio.gather` — the model must be able to handle parallel requests without race conditions.

The agent uses the outputs to compute a utility score via the formula:

```
Utility = -alpha * severe_event_probability
          -beta  * discontinuation_probability
          -gamma * mild_side_effect_score
          +delta * contraceptive_effectiveness
```

### Request

```json
{
  "candidate_pill": {
    "set_id": "021df70a-0c37-4237-a2f7-2597ff6d41f5",
    "brand_name": "Hailey 1.5/30",
    "generic_name": "NORETHINDRONE ACETATE AND ETHINYL ESTRADIOL",
    "manufacturer_name": "Glenmark Pharmaceuticals Inc., USA",
    "product_ndc": "68462-504",
    "substance_name": "ETHINYL ESTRADIOL | NORETHINDRONE ACETATE",
    "adverse_reactions": "...",
    "warnings": "...",
    "warnings_and_cautions": "...",
    "boxed_warning": "...",
    "contraindications": "...",
    "drug_interactions": "...",
    "clinical_pharmacology": "...",
    "indications_and_usage": "...",
    "dosage_and_administration": "...",
    "description": "..."
  },
  "patient": {
    "age": 28,
    "cond_migraine_with_aura": 0,
    "cond_stroke": 0,
    "..." : "... (all patient fields as per Cluster Model)"
  }
}
```

#### Request Field Reference

| Field | Type | Required | Stable? | Description |
|---|---|---|---|---|
| `candidate_pill` | object | Yes | ✅ | Complete pill record from pills.csv |
| `candidate_pill.set_id` | string | Yes | ✅ | Unique pill identifier (UUID) |
| `candidate_pill.brand_name` | string | Yes | ✅ | Brand name of the pill |
| `candidate_pill.generic_name` | string | Yes | ✅ | Generic/chemical name |
| `candidate_pill.manufacturer_name` | string | Yes | ✅ | Manufacturer name |
| `candidate_pill.product_ndc` | string | Yes | ✅ | National Drug Code |
| `candidate_pill.substance_name` | string | Yes | ✅ | Active substance(s) |
| `candidate_pill.adverse_reactions` | string | Yes | 🔄 | Full adverse reactions text from FDA labeling |
| `candidate_pill.warnings` | string | Yes | 🔄 | Warnings text from FDA labeling |
| `candidate_pill.warnings_and_cautions` | string | Yes | 🔄 | Warnings and cautions text |
| `candidate_pill.boxed_warning` | string | Yes | 🔄 | Black box warning text |
| `candidate_pill.contraindications` | string | Yes | 🔄 | Contraindications text |
| `candidate_pill.drug_interactions` | string | Yes | 🔄 | Drug interactions text |
| `candidate_pill.clinical_pharmacology` | string | Yes | 🔄 | Clinical pharmacology text |
| `candidate_pill.indications_and_usage` | string | Yes | 🔄 | Indications and usage text |
| `candidate_pill.dosage_and_administration` | string | Yes | 🔄 | Dosage information text |
| `candidate_pill.description` | string | Yes | 🔄 | Product description text |
| `patient` | object | Yes | ✅ | Same patient object as the Cluster Model request (see above) |

The `patient` object schema is identical to the Cluster Model request. The agent reuses the same `patient_data` object for both calls with no transformation. 

🔄 The pill text fields (adverse_reactions, warnings, etc.) are FDA labeling data and may be very long strings. The ML team is still determining which fields they will actually use for the simulation model.

### Response

```json
{
  "discontinuation_probability": 0.14,
  "severe_event_probability": 0.007,
  "mild_side_effect_score": 0.31,
  "contraceptive_effectiveness": 0.97
}
```

#### Response Field Reference

| Field | Type | Range | Stable? | Description |
|---|---|---|---|---|
| `discontinuation_probability` | float | 0.0-1.0 | ✅ | Predicted probability the patient stops taking this pill within the simulation horizon |
| `severe_event_probability` | float | 0.0-1.0 | ✅ | Predicted probability of a severe adverse event (e.g. thrombosis, stroke) |
| `mild_side_effect_score` | float | 0.0-1.0 | 🔄 | Composite score of predicted mild side effects (nausea, mood, headaches). Higher = worse |
| `contraceptive_effectiveness` | float | 0.0-1.0 | 🔄 | Predicted contraceptive effectiveness for this patient-pill combination |

🔄 `mild_side_effect_score` may be split into separate fields per side effect type if the ML team determines the model can predict them individually. `contraceptive_effectiveness` depends on the simulation time horizon which is not yet confirmed — the field may be renamed or restructured accordingly.

The two fields marked ✅ feed directly into the safety weighting of the utility formula (`severe_event_probability` carries the highest penalty coefficient). Changes to these would require retuning the utility weights and should be flagged to the agent team immediately.

### Example — Full Request/Response

```
POST /api/v1/simulator/simulate
Content-Type: application/json

{
  "candidate_pill": {
    "set_id": "036a15be-8c2e-4f65-b5dc-f333e2994d4a",
    "brand_name": "Dolishale",
    "generic_name": "LEVONORGESTREL AND ETHINYL ESTRADIOL",
    "manufacturer_name": "Ingenus Pharmaceuticals, LLC",
    "product_ndc": "50742-659",
    "substance_name": "ETHINYL ESTRADIOL | LEVONORGESTREL",
    "adverse_reactions": "...",
    "warnings": "...",
    "warnings_and_cautions": "...",
    "boxed_warning": "...",
    "contraindications": "...",
    "drug_interactions": "...",
    "clinical_pharmacology": "...",
    "indications_and_usage": "...",
    "dosage_and_administration": "...",
    "description": "..."
  },
  "patient": {
    "age": 28,
    "cond_migraine_with_aura": 0,
    "cond_stroke": 0,
    "..." : "... (all patient fields)",
    "obs_smoker": 1
  }
}

HTTP 200 OK
{
  "discontinuation_probability": 0.12,
  "severe_event_probability": 0.006,
  "mild_side_effect_score": 0.25,
  "contraceptive_effectiveness": 0.98
}
```

---

## Health Check

Both models must expose a health check endpoint. The agent does not call this during normal operation, but it is used by the mock server, CI pipelines, and OpenShift readiness probes.

```
GET /api/v1/health
```

```json
{
  "status": "ok",
  "model": "cluster",
  "version": "1.0.0"
}
```

---

## Error Format

All errors return a consistent JSON body regardless of the failure type.

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Field 'patient.age' is required and must be an integer.",
    "field": "patient.age"
  }
}
```

#### Error Code Reference

| HTTP Status | Code | When |
|---|---|---|
| `400` | `INVALID_INPUT` | Missing required field, wrong type, value out of range |
| `400` | `UNKNOWN_PILL` | `candidate_pill` ID not recognized by the simulator |
| `422` | `UNPROCESSABLE_INPUT` | Input is structurally valid but cannot be processed |
| `500` | `MODEL_ERROR` | Internal model failure |
| `503` | `MODEL_UNAVAILABLE` | Model not loaded or warming up |

The agent handles these as follows: `400` and `422` are treated as bugs and raise immediately. `503` is retried once after 2 seconds. `500` is logged and the candidate is skipped in the simulation loop.

---

## Mock Server Alignment

The mock server defined in `tests/mock_server.py` implements exactly these endpoints and schemas. When the real models are deployed on OpenShift, the only change is updating `CLUSTER_API_URL` and `SIMULATOR_API_URL` to point to the real host. The agent tool clients (`cluster_api.py`, `simulator_api.py`) do not change.

The mock server will be deleted once real endpoints are confirmed working. See `mock-testing-strategy.md` for the full removal checklist.

---

## Open Items

| Item | Owner | Impact if delayed |
|---|---|---|
| Confirm authentication method (API Key vs Bearer token) | ML team | Blocks HTTP client implementation |
| Provide OpenShift base URL for staging environment | ML team | Blocks integration testing |
| Confirm full list of valid `candidate_pill` identifiers | ML team | Blocks candidate generation node |
| Confirm simulation time horizon (30-day, 90-day, etc.) | Both teams | Affects output field names and utility weight tuning |
| Confirm final patient input features for both models | ML team | 🔄 fields above may all change — agent team needs notice to update tool clients and mock server |
| Confirm whether `mild_side_effect_score` stays composite or splits into multiple fields | ML team | Affects utility formula and `SystemState` schema |
| Confirm whether both models take identical `patient` objects | ML team | If different, agent needs separate serialization logic per model |