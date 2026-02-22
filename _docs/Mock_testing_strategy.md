# Mock Testing Strategy — BC Recommendation Agent

## Why This Document Exists

The ML models (Cluster Model and Simulator Model) are being developed in parallel and are not yet deployed on Red Hat OpenShift. This means the agent layer cannot make real API calls during development.

Rather than waiting, we build the full agent infrastructure now and replace the real API clients with believable fake responses. This document explains the strategy, how to implement it, and — critically — how to cleanly remove it when the models are ready.

**This entire mock layer is temporary.** It exists only to unblock development. Every mock file created here will be deleted when real model endpoints are available on OpenShift.

---

## The Core Principle: Mock at the Boundary

The agent communicates with the ML models through exactly two files:

```
agent/tools/cluster_api.py
agent/tools/simulator_api.py
```

These are the only points where the agent touches the outside world. Everything above this boundary — the graph, the nodes, the state, the LLM calls — is pure Python that we fully control and can test right now.

The mock strategy is simple: **replace only these two files with fake implementations that return realistic data.** The rest of the codebase does not change at all.

```
Real architecture:
  Agent → cluster_api.py → [HTTP] → Cluster Model on OpenShift
  Agent → simulator_api.py → [HTTP] → Simulator Model on OpenShift

Mocked architecture:
  Agent → cluster_api.py → [fake response]
  Agent → simulator_api.py → [fake response]
```

This means when the real models are ready, the switch is just updating two files and two environment variables. No agent logic changes.

---

## Two Levels of Mocking

We use two levels depending on what we're testing.

### Level 1 — Static Mocks (Unit Tests)

Hardcoded Python functions used inside `pytest`. No network, no server, fully deterministic. Used to test individual node logic in isolation.

### Level 2 — Mock API Server (Integration Tests)

A lightweight FastAPI server that mimics the real OpenShift endpoints. The real agent code makes actual HTTP calls to `localhost` instead of OpenShift. Used to test the full stack end to end.

Both levels live in the `tests/` directory and are completely separate from the production `agent/` code.

---

## Project Structure

```
tests/
├── mocks/
│   ├── __init__.py
│   ├── cluster_api.py       # Level 1: static mock for cluster model
│   └── simulator_api.py     # Level 1: static mock for simulator model
├── mock_server.py           # Level 2: full HTTP mock server
├── fixtures/
│   └── patients.json        # Patient test profiles
├── test_nodes.py            # Unit tests using Level 1 mocks
└── test_graph.py            # Integration tests using Level 2 mock server
```

---

## Level 1 Implementation — Static Mocks

### `tests/mocks/cluster_api.py`

```python
"""
Mock replacement for agent/tools/cluster_api.py.

Returns hardcoded cluster assignments. Used in unit tests via pytest's
patch decorator to avoid any HTTP calls.

REMOVE WHEN: Real Cluster Model is deployed on Red Hat OpenShift.
"""

def call_cluster_model(patient_data: dict) -> dict:
    """
    Returns a realistic cluster assignment based on patient age.
    Simulates the behavior of the real Cluster Model API.
    """
    age = patient_data.get("age", 25)

    if age < 25:
        return {"cluster_profile": "cluster_0", "cluster_confidence": 0.91}
    elif age < 35:
        return {"cluster_profile": "cluster_2", "cluster_confidence": 0.88}
    elif age < 45:
        return {"cluster_profile": "cluster_3", "cluster_confidence": 0.76}
    else:
        return {"cluster_profile": "cluster_4", "cluster_confidence": 0.65}
```

Note that the 45+ age group returns a confidence of `0.65` — deliberately below the `0.70` threshold defined in `cluster.py`. This triggers the low-confidence LLM weight adjustment path, which is important to keep tested.

### `tests/mocks/simulator_api.py`

```python
"""
Mock replacement for agent/tools/simulator_api.py.

Returns deterministic fake simulation results per pill ID.
Determinism is important: the same pill always returns the same numbers,
which means the utility optimizer always picks the same winner, making
tests repeatable and assertions reliable.

REMOVE WHEN: Real Simulator Model is deployed on Red Hat OpenShift.
"""
import random

# Hardcoded results for known test pills.
# pill_a is intentionally the best candidate so tests can assert on it.
FAKE_SIMULATION_RESULTS = {
    "pill_a": {
        "discontinuation_probability": 0.10,
        "severe_event_probability": 0.005,
        "mild_side_effect_score": 0.20,
        "contraceptive_effectiveness": 0.98
    },
    "pill_b": {
        "discontinuation_probability": 0.18,
        "severe_event_probability": 0.009,
        "mild_side_effect_score": 0.35,
        "contraceptive_effectiveness": 0.97
    },
    "pill_c": {
        "discontinuation_probability": 0.27,
        "severe_event_probability": 0.014,
        "mild_side_effect_score": 0.55,
        "contraceptive_effectiveness": 0.95
    },
    "pill_d_high_risk": {
        "discontinuation_probability": 0.40,
        "severe_event_probability": 0.030,
        "mild_side_effect_score": 0.70,
        "contraceptive_effectiveness": 0.93
    }
}

async def call_simulator(candidate_id: str, patient_data: dict) -> dict:
    if candidate_id in FAKE_SIMULATION_RESULTS:
        return FAKE_SIMULATION_RESULTS[candidate_id]

    # Unknown pill: deterministic noise based on pill name
    # (same pill always gets same result across test runs)
    seed = sum(ord(c) for c in candidate_id)
    rng = random.Random(seed)
    return {
        "discontinuation_probability": round(rng.uniform(0.05, 0.40), 3),
        "severe_event_probability": round(rng.uniform(0.001, 0.025), 4),
        "mild_side_effect_score": round(rng.uniform(0.10, 0.80), 2),
        "contraceptive_effectiveness": round(rng.uniform(0.90, 0.99), 3),
    }
```

### Using Level 1 Mocks in Tests

```python
# tests/test_nodes.py
from unittest.mock import patch
from agent.nodes import utility, convergence
from tests.mocks import cluster_api as mock_cluster
from tests.mocks import simulator_api as mock_simulator

def make_state(**overrides) -> dict:
    """Factory for a minimal valid SystemState for testing."""
    base = {
        "patient_data": {"age": 28, "pathologies": [], "habits": [], "medical_history": []},
        "cluster_profile": "cluster_2",
        "cluster_confidence": 0.88,
        "relative_risk_rules": [],
        "candidate_pool": ["pill_a", "pill_b", "pill_c"],
        "simulated_results": {},
        "utility_scores": {},
        "best_candidate": None,
        "iteration": 0,
        "converged": False,
        "previous_best_utility": None
    }
    base.update(overrides)
    return base


@patch("agent.tools.cluster_api.call_cluster_model", mock_cluster.call_cluster_model)
def test_cluster_node_high_confidence():
    from agent.nodes import cluster
    state = make_state()
    result = cluster.run(state)
    assert result["cluster_confidence"] >= 0.70
    assert result["cluster_profile"] is not None


@patch("agent.tools.simulator_api.call_simulator", mock_simulator.call_simulator)
def test_utility_node_selects_best_candidate():
    from agent.nodes import utility
    import asyncio

    state = make_state(simulated_results={
        "pill_a": mock_simulator.FAKE_SIMULATION_RESULTS["pill_a"],
        "pill_b": mock_simulator.FAKE_SIMULATION_RESULTS["pill_b"],
        "pill_c": mock_simulator.FAKE_SIMULATION_RESULTS["pill_c"],
    })
    result = utility.run(state)
    # pill_a has the best simulated profile, should always win
    assert result["best_candidate"] == "pill_a"
    assert result["utility_scores"]["pill_a"] > result["utility_scores"]["pill_b"]


def test_convergence_triggers_after_max_iterations():
    from agent.nodes import convergence
    state = make_state(
        iteration=5,  # MAX_ITERATIONS reached
        utility_scores={"pill_a": 0.85},
        best_candidate="pill_a",
        simulated_results={"pill_a": mock_simulator.FAKE_SIMULATION_RESULTS["pill_a"]}
    )
    result = convergence.run(state)
    assert result["converged"] is True
```

---

## Level 2 Implementation — Mock API Server

The mock server is a real FastAPI application that runs locally and responds to HTTP requests exactly as the real OpenShift models would. Your agent makes genuine HTTP calls to it — which means you're testing the full network path, request serialization, response parsing, and error handling.

### `tests/mock_server.py`

```python
"""
Lightweight mock server mimicking the Cluster Model and Simulator Model APIs.

Runs locally on port 8001. The agent hits this server via environment variables:
  CLUSTER_API_URL=http://localhost:8001/cluster/predict
  SIMULATOR_API_URL=http://localhost:8001/simulator/simulate

This server is a development and testing tool ONLY.
It is never deployed and must not be included in the production build.

REMOVE WHEN: Real models are deployed on Red Hat OpenShift.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import random

app = FastAPI(title="BC Agent Mock Model Server", version="0.1.0-mock")


# ── Request schemas matching what the agent sends ──────────────────────────

class ClusterRequest(BaseModel):
    patient: dict

class SimulatorRequest(BaseModel):
    candidate_pill: str
    patient: dict


# ── Cluster Model endpoint ─────────────────────────────────────────────────

@app.post("/cluster/predict")
def cluster_predict(body: ClusterRequest):
    """
    Mimics the Cluster Model API response.
    Assigns cluster based on patient age and pathology count to produce
    varied, somewhat realistic cluster assignments.
    """
    age = body.patient.get("age", 25)
    pathology_count = len(body.patient.get("pathologies", []))

    if age < 25 and pathology_count == 0:
        profile, confidence = "cluster_0", 0.93
    elif age < 35:
        profile, confidence = "cluster_2", round(random.uniform(0.78, 0.92), 2)
    elif age < 45:
        profile, confidence = "cluster_3", round(random.uniform(0.68, 0.82), 2)
    else:
        # Older patients with more pathologies → lower confidence
        profile = "cluster_4"
        confidence = round(max(0.50, 0.80 - (pathology_count * 0.05)), 2)

    return {"cluster_profile": profile, "cluster_confidence": confidence}


# ── Simulator Model endpoint ───────────────────────────────────────────────

# Deterministic per-pill results so the optimizer always produces the same ranking
PILL_PROFILES = {
    "pill_a": (0.10, 0.005, 0.20, 0.98),
    "pill_b": (0.18, 0.009, 0.35, 0.97),
    "pill_c": (0.27, 0.014, 0.55, 0.95),
    "pill_d": (0.33, 0.011, 0.45, 0.96),
    "pill_e": (0.22, 0.007, 0.30, 0.97),
}

@app.post("/simulator/simulate")
def simulator_simulate(body: SimulatorRequest):
    """
    Mimics the Simulator Model API response.
    Returns deterministic results for known pills, random for unknown.
    """
    pill = body.candidate_pill

    if pill in PILL_PROFILES:
        disc, severe, mild, effectiveness = PILL_PROFILES[pill]
    else:
        seed = sum(ord(c) for c in pill)
        rng = random.Random(seed)
        disc = round(rng.uniform(0.05, 0.40), 3)
        severe = round(rng.uniform(0.001, 0.025), 4)
        mild = round(rng.uniform(0.10, 0.80), 2)
        effectiveness = round(rng.uniform(0.90, 0.99), 3)

    return {
        "discontinuation_probability": disc,
        "severe_event_probability": severe,
        "mild_side_effect_score": mild,
        "contraceptive_effectiveness": effectiveness
    }


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "mode": "MOCK — not for production"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
```

### Running the Mock Server

```bash
# Terminal 1 — start the mock server
python tests/mock_server.py

# Terminal 2 — start your FastAPI backend pointing at the mock
CLUSTER_API_URL=http://localhost:8001/cluster/predict \
SIMULATOR_API_URL=http://localhost:8001/simulator/simulate \
LLM_PROVIDER=groq \
uvicorn main:app --reload
```

Or add these to your `.env.local`:

```bash
CLUSTER_API_URL=http://localhost:8001/cluster/predict
SIMULATOR_API_URL=http://localhost:8001/simulator/simulate
```

---

## Test Scenarios

Build test profiles that cover the important edge cases of the system. These go in `tests/fixtures/patients.json` and are reused across both unit and integration tests.

```json
{
  "standard": {
    "description": "Healthy young adult, should produce high-confidence cluster and clean recommendation",
    "age": 28,
    "pathologies": [],
    "habits": ["smoking"],
    "medical_history": []
  },
  "high_risk": {
    "description": "Multiple pathologies, should trigger aggressive relative risk rules",
    "age": 42,
    "pathologies": ["hypertension", "migraines"],
    "habits": [],
    "medical_history": ["dvt"]
  },
  "low_confidence_cluster": {
    "description": "Edge-case age profile, confidence below 0.70 threshold — tests LLM weight adjustment",
    "age": 48,
    "pathologies": ["diabetes", "hypertension", "obesity"],
    "habits": ["smoking"],
    "medical_history": []
  },
  "minimal_candidates": {
    "description": "Many contraindications — tests behavior when Safe Gate leaves very few candidates",
    "age": 38,
    "pathologies": ["lupus", "hypertension", "migraines"],
    "habits": ["smoking"],
    "medical_history": ["stroke", "dvt"]
  }
}
```

The `minimal_candidates` case is the most important one to test. It simulates the situation where the Safe Gate Engine filters heavily and the agent receives only 1-2 candidates. The loop should still converge correctly and the output should still be valid.

---

## Integration Test with Mock Server

```python
# tests/test_graph.py
import pytest
import httpx

# Assumes mock server is already running on port 8001
# In CI, start it as a pytest fixture

@pytest.fixture(scope="session", autouse=True)
def mock_server():
    """Start mock server for the test session."""
    import subprocess, time
    proc = subprocess.Popen(["python", "tests/mock_server.py"])
    time.sleep(1.5)  # Give it time to start
    yield
    proc.terminate()

def test_full_recommendation_standard_patient():
    patient = {
        "age": 28,
        "pathologies": [],
        "habits": ["smoking"],
        "medical_history": []
    }
    response = httpx.post("http://localhost:8000/recommend", json=patient)
    assert response.status_code == 200

    data = response.json()
    assert "selected_pill" in data
    assert "predicted_discontinuation" in data
    assert "severe_risk" in data
    assert isinstance(data["reason_codes"], list)
    assert len(data["reason_codes"]) >= 2

def test_recommendation_converges_within_max_iterations():
    patient = {
        "age": 42,
        "pathologies": ["hypertension"],
        "habits": [],
        "medical_history": []
    }
    response = httpx.post("http://localhost:8000/recommend", json=patient)
    assert response.status_code == 200
    # If it times out or 500s, the loop likely didn't converge
```

---

## What This Lets You Fully Validate

With the mock layer in place, you can test and verify everything that matters in the agent before any real model exists:

- The full graph executes all nodes in the correct order
- The loop converges within `MAX_ITERATIONS` for all patient profiles
- Low-confidence cluster assignments trigger the LLM weight adjustment correctly
- The utility function consistently selects the candidate with the best simulated profile
- The output structure is always valid and complete
- The Safe Gate boundary is respected — the agent never sees excluded pills
- LangSmith traces are clean and readable for every scenario
- The LLM generates valid parseable JSON for reason codes across different LLM providers (Groq locally, Claude in staging)

---

## Switching to Real Models — Checklist

When the ML team deploys the real models on Red Hat OpenShift, follow this checklist:

**1. Get the real endpoint URLs from the ML team**
```bash
# They will provide something like:
CLUSTER_API_URL=https://cluster-model.your-openshift-cluster.com/predict
SIMULATOR_API_URL=https://simulator-model.your-openshift-cluster.com/simulate
```

**2. Update environment variables**

Replace the localhost URLs in `.env.local` and `.env.production`. No code changes.

**3. Verify the response schemas match**

Run the integration tests against the real endpoints. If they pass, the schemas are compatible. If not, the only files to update are `cluster_api.py` and `simulator_api.py` in `agent/tools/`.

**4. Delete the mock files**

```bash
# Remove all mock infrastructure — it has no place in production
rm tests/mocks/cluster_api.py
rm tests/mocks/simulator_api.py
rm tests/mock_server.py
```

**5. Update the integration tests**

Replace the `mock_server` pytest fixture with direct calls to the real endpoints (or a staging environment). The test scenarios and assertions remain the same.

**6. Re-run the full test suite**

All unit tests that used Level 1 mocks should still pass — just without the patches. All integration tests should pass against the real endpoints.

---

## Summary

| | Level 1 (Static Mocks) | Level 2 (Mock Server) |
|---|---|---|
| **Used for** | Unit tests, node isolation | Integration tests, full stack |
| **How it works** | `patch` decorator replaces API clients | Real HTTP calls to localhost |
| **Speed** | Instant | Fast (local network) |
| **Files** | `tests/mocks/*.py` | `tests/mock_server.py` |
| **Removed when** | Real models on OpenShift | Real models on OpenShift |

The mock layer is a development scaffold, not a permanent fixture. It is intentionally kept small, clearly labeled, and isolated in the `tests/` directory so it can be deleted cleanly in a single step when the ML team is ready.