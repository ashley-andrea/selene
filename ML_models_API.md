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
    "pathologies": ["migraines"],
    "habits": ["smoking"],
    "medical_history": ["appendectomy"]
  }
}
```

#### Request Field Reference

| Field | Type | Required | Stable? | Description |
|---|---|---|---|---|
| `patient` | object | Yes | ✅ | Top-level wrapper for all patient data |
| `patient.age` | integer | Yes | ✅ | Patient age in years. Range: 15-55 |
| `patient.pathologies` | array of strings | Yes | 🔄 | Active diagnosed conditions. Lowercase. Empty array if none |
| `patient.habits` | array of strings | Yes | 🔄 | Lifestyle habits relevant to contraceptive risk. Lowercase. Empty array if none |
| `patient.medical_history` | array of strings | Yes | 🔄 | Past medical events. Lowercase. Empty array if none |

🔄 `pathologies`, `habits`, and `medical_history` are the most likely fields to change. The ML team may decide to encode these differently (e.g. structured objects instead of string arrays, or a different set of recognized values entirely) once model training is underway.

#### Valid Values — pathologies

The agent normalizes these to lowercase before sending. The model must handle at minimum:

`hypertension`, `migraines`, `diabetes`, `lupus`, `obesity`, `depression`, `pcos`, `endometriosis`, `epilepsy`, `hypothyroidism`

🔄 This list is provisional. The ML team will confirm the full set of recognized values based on the training data.

#### Valid Values — habits

`smoking`, `alcohol`, `sedentary_lifestyle`

🔄 Provisional. May expand or be restructured.

#### Valid Values — medical_history

`dvt`, `stroke`, `pulmonary_embolism`, `breast_cancer`, `liver_disease`, `appendectomy`

🔄 Provisional. May expand or be restructured.

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
    "pathologies": ["hypertension", "migraines"],
    "habits": [],
    "medical_history": ["dvt"]
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
  "candidate_pill": "pill_levonorgestrel_30",
  "patient": {
    "age": 28,
    "pathologies": ["migraines"],
    "habits": ["smoking"],
    "medical_history": []
  }
}
```

#### Request Field Reference

| Field | Type | Required | Stable? | Description |
|---|---|---|---|---|
| `candidate_pill` | string | Yes | ✅ | Unique pill identifier from the pill database |
| `patient` | object | Yes | ✅ | Same patient object as the Cluster Model request |
| `patient.age` | integer | Yes | ✅ | Patient age in years |
| `patient.pathologies` | array of strings | Yes | 🔄 | Active conditions. Lowercase. Empty array if none |
| `patient.habits` | array of strings | Yes | 🔄 | Lifestyle habits. Lowercase. Empty array if none |
| `patient.medical_history` | array of strings | Yes | 🔄 | Past events. Lowercase. Empty array if none |

The `patient` object schema is identical to the Cluster Model request. The agent reuses the same `patient_data` object for both calls with no transformation. If the ML team decides the two models need different patient inputs, this will need to be revisited.

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
  "candidate_pill": "pill_levonorgestrel_30",
  "patient": {
    "age": 28,
    "pathologies": [],
    "habits": ["smoking"],
    "medical_history": []
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