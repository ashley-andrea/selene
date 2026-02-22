# Birth Control Recommendation Agent — Implementation Guide

## Overview

This document is the definitive guide for building the LLM agent layer of the Birth Control Recommendation System. It covers framework selection rationale, project setup, code architecture, testing strategy, and deployment considerations.

The agent's role is to orchestrate a structured optimization loop: it takes patient data, drives a sequence of tool calls (cluster model, pill database, simulator), iterates until convergence, and produces a ranked recommendation with reason codes. The intelligence lives in the loop, not in LLM creativity — prompts are tight, outputs are structured JSON, and all medical safety is handled deterministically outside the LLM's visibility.

---

## Framework: LangGraph

### Why LangGraph

LangGraph is the correct framework for this project. The architecture described in `doc.md` maps directly to a graph of nodes with conditional edges — which is precisely what LangGraph provides.

Key reasons:

- **Iterative loop support**: The simulate → score → check convergence → (repeat or exit) loop is a first-class pattern in LangGraph via conditional edges. This would require custom boilerplate in any other framework.
- **Persistent state**: LangGraph passes a typed state object (`SystemState`) between nodes automatically. No manual threading of state through function calls.
- **Model-agnostic**: Swap LLM providers with a one-line environment variable change. Critical for local testing vs. Claude production deployment.
- **Production-grade**: Used in production AI systems. LangSmith tracing (free tier) integrates natively, which is invaluable for debugging loops during a hackathon.
- **Clean separation**: Nodes are plain Python functions. Tool calls (HTTP requests to model APIs) are separate modules. The graph definition is a thin wiring layer.
---

## LLM Provider Strategy

The project needs to run on free/local LLMs during development and Claude in production. The solution is a **factory pattern** driven by a single environment variable.

### The Factory

```python
# agent/llm.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Returns the appropriate LLM client based on LLM_PROVIDER env var.
    Default is 'claude' for production safety.
    """
    provider = os.getenv("LLM_PROVIDER", "claude")

    if provider == "claude":
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024
        )

    elif provider == "ollama":
        # For local development — requires Ollama running locally
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=0
        )

    elif provider == "crusoe":
        # Crusoe Managed Inference — OpenAI-compatible, dev/staging
        return ChatOpenAI(
            base_url="https://api.crusoe.ai/v1",
            api_key=os.getenv("CRUSOE_API_KEY"),
            model=os.getenv("CRUSOE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            temperature=0
        )

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
```
### Claude for Production

Use `claude-sonnet-4-6` (not Opus) for deployment. The agent's reasoning tasks are structured and well-defined — Sonnet is faster, cheaper, and more than capable. Set `LLM_PROVIDER=claude` and `ANTHROPIC_API_KEY` in the deployment environment.

---

## Environment Configuration

All configuration is environment-variable driven. Create `.env` files for each environment — never commit secrets.

### `.env.local` (development with Crusoe)

```bash
# Agent
LLM_PROVIDER=crusoe
CRUSOE_API_KEY=your_crusoe_api_key
CRUSOE_MODEL=meta-llama/Llama-3.3-70B-Instruct

# Model APIs (Red Hat OpenShift)
CLUSTER_API_URL=http://localhost:8001/predict   # or staging URL
SIMULATOR_API_URL=http://localhost:8002/simulate

# LangSmith Tracing (free tier - highly recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=bc-agent-dev
```

### `.env.local` (development with Ollama)

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
CLUSTER_API_URL=http://localhost:8001/predict
SIMULATOR_API_URL=http://localhost:8002/simulate
LANGCHAIN_TRACING_V2=false
```

### `.env.production`

```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-sonnet-4-6

CLUSTER_API_URL=https://your-openshift-cluster.example.com/cluster/predict
SIMULATOR_API_URL=https://your-openshift-cluster.example.com/simulator/simulate

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=bc-agent-prod
```

---

## Project Structure

```
agent/
├── __init__.py
├── graph.py              # LangGraph graph assembly — the wiring layer
├── llm.py                # LLM factory (described above)
├── state.py              # SystemState TypedDict
│
├── nodes/                # One file per graph node
│   ├── __init__.py
│   ├── validator.py      # Input validation and normalization
│   ├── cluster.py        # Calls Cluster Model API, evaluates confidence
│   ├── candidate_gen.py  # Queries pill DB, applies relative risk rules
│   ├── simulator.py      # Calls Simulator API for each candidate
│   ├── utility.py        # Utility score computation
│   └── convergence.py    # Loop control and output formatting
│
├── tools/                # HTTP clients for external APIs
│   ├── __init__.py
│   ├── cluster_api.py
│   └── simulator_api.py
│
└── prompts/              # Prompt templates (keep them here, not inline)
    ├── cluster_eval.txt
    ├── candidate_rank.txt
    └── reason_codes.txt

tests/
├── test_nodes.py
├── test_graph.py
└── fixtures/
    └── sample_patient.json
```

---

## State Definition

```python
# agent/state.py
from typing import TypedDict, Optional

class SystemState(TypedDict):
    # Patient input
    patient_data: dict                  # Validated and normalized

    # Cluster assignment
    cluster_profile: Optional[str]      # e.g. "cluster_3"
    cluster_confidence: Optional[float] # 0.0 - 1.0

    # Safety layer outputs (hard_constraints are NEVER stored here)
    relative_risk_rules: list           # Soft rules for candidate ranking

    # Candidate pool (already filtered by Safe Gate Engine in FastAPI backend)
    candidate_pool: list                # List of pill IDs from allowed set only

    # Simulation results
    simulated_results: dict             # {pill_id: {discontinuation, severe_event, mild_side_effect}}

    # Utility scores
    utility_scores: dict                # {pill_id: float}
    best_candidate: Optional[str]       # Pill ID with highest utility

    # Loop control
    iteration: int
    converged: bool
    previous_best_utility: Optional[float]  # For convergence comparison
```

**Important**: `hard_constraints` are never part of `SystemState`. The Safe Gate Engine (implemented as a deterministic module in the FastAPI backend) filters the candidate pool before the agent ever sees it. The agent operates only on the filtered pool and has no visibility into what was excluded.

---

## Node Implementations

### validator.py

```python
# agent/nodes/validator.py
from agent.state import SystemState

REQUIRED_FIELDS = ["age", "medical_history", "habits", "pathologies"]

def run(state: SystemState) -> dict:
    """Validates and normalizes patient_data."""
    patient = state.get("patient_data", {})

    for field in REQUIRED_FIELDS:
        if field not in patient:
            raise ValueError(f"Missing required patient field: {field}")

    # Normalize age to int
    patient["age"] = int(patient["age"])

    # Normalize string lists to lowercase
    patient["pathologies"] = [p.lower().strip() for p in patient.get("pathologies", [])]

    return {
        "patient_data": patient,
        "iteration": 0,
        "converged": False,
        "relative_risk_rules": [],
        "candidate_pool": [],
        "simulated_results": {},
        "utility_scores": {},
        "best_candidate": None,
        "previous_best_utility": None
    }
```

### cluster.py

```python
# agent/nodes/cluster.py
from agent.state import SystemState
from agent.tools.cluster_api import call_cluster_model
from agent.llm import get_llm
from langchain_core.messages import HumanMessage

def run(state: SystemState) -> dict:
    """
    Calls the Cluster Model API, then uses the LLM to evaluate
    confidence and flag low-confidence cases for weight adjustment.
    """
    result = call_cluster_model(state["patient_data"])

    cluster_profile = result["cluster_profile"]
    cluster_confidence = result["cluster_confidence"]

    # LLM evaluates cluster confidence — structured output only
    if cluster_confidence < 0.70:
        llm = get_llm()
        prompt = f"""
Patient cluster assignment: {cluster_profile}
Confidence score: {cluster_confidence}

Return a JSON object with:
- weight_adjustment: float (multiplier to apply to relative risk rules, 1.0 = no change)
- rationale: string (one sentence explanation)

Return ONLY the JSON object, no other text.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        import json
        adjustment = json.loads(response.content)
        # Store for use in candidate_gen
        cluster_profile = {
            "profile": cluster_profile,
            "weight_adjustment": adjustment["weight_adjustment"],
            "low_confidence": True
        }

    return {
        "cluster_profile": cluster_profile,
        "cluster_confidence": cluster_confidence
    }
```

### simulator.py

```python
# agent/nodes/simulator.py
import asyncio
from agent.state import SystemState
from agent.tools.simulator_api import call_simulator

def run(state: SystemState) -> dict:
    """
    Calls the Simulator API for each candidate in the pool.
    Runs calls concurrently for speed.
    """
    candidates = state["candidate_pool"]
    patient_data = state["patient_data"]

    async def run_all():
        tasks = [
            call_simulator(candidate_id, patient_data)
            for candidate_id in candidates
        ]
        return await asyncio.gather(*tasks)

    results_list = asyncio.run(run_all())
    simulated_results = {
        candidate_id: result
        for candidate_id, result in zip(candidates, results_list)
    }

    return {"simulated_results": simulated_results}
```

### utility.py

```python
# agent/nodes/utility.py
from agent.state import SystemState

# Utility weights — tune these for the hackathon
ALPHA = 2.0   # Weight for severe adverse event probability
BETA = 1.5    # Weight for discontinuation probability
GAMMA = 0.5   # Weight for mild side effect score
DELTA = 1.0   # Weight for contraceptive effectiveness

def compute_utility(sim_result: dict) -> float:
    return (
        - ALPHA * sim_result["severe_event_probability"]
        - BETA  * sim_result["discontinuation_probability"]
        - GAMMA * sim_result["mild_side_effect_score"]
        + DELTA * sim_result["contraceptive_effectiveness"]
    )

def run(state: SystemState) -> dict:
    utility_scores = {
        pill_id: compute_utility(result)
        for pill_id, result in state["simulated_results"].items()
    }

    best_candidate = max(utility_scores, key=utility_scores.get)

    return {
        "utility_scores": utility_scores,
        "best_candidate": best_candidate
    }
```

### convergence.py

```python
# agent/nodes/convergence.py
from agent.state import SystemState
from agent.llm import get_llm
from langchain_core.messages import HumanMessage
import json

MAX_ITERATIONS = 5
EPSILON = 0.01  # Minimum utility improvement to continue

def run(state: SystemState) -> dict:
    """
    Checks if the loop has converged. If yes, generates reason codes via LLM.
    If no, increments iteration counter for the next loop.
    """
    iteration = state["iteration"]
    best_utility = state["utility_scores"].get(state["best_candidate"], 0.0)
    previous_utility = state.get("previous_best_utility")

    # Convergence conditions
    improvement = abs(best_utility - (previous_utility or 0.0))
    converged = (
        iteration >= MAX_ITERATIONS
        or (previous_utility is not None and improvement < EPSILON)
    )

    if converged:
        reason_codes = generate_reason_codes(state)
        return {
            "converged": True,
            "iteration": iteration + 1,
            "previous_best_utility": best_utility,
            "reason_codes": reason_codes  # Extend SystemState if needed
        }

    return {
        "converged": False,
        "iteration": iteration + 1,
        "previous_best_utility": best_utility
    }

def generate_reason_codes(state: SystemState) -> list:
    """Uses LLM to generate human-readable reason codes for the top recommendation."""
    llm = get_llm()
    best = state["best_candidate"]
    sim = state["simulated_results"].get(best, {})
    cluster = state["cluster_profile"]

    prompt = f"""
You are generating reason codes for a contraceptive recommendation.

Selected pill: {best}
Cluster profile: {cluster}
Simulation results: {json.dumps(sim, indent=2)}

Return a JSON array of 2-4 short reason strings explaining why this pill was selected.
Example: ["lowest thrombotic risk in cluster", "high predicted adherence", "matches patient age profile"]

Return ONLY the JSON array, no other text.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return json.loads(response.content)
```

---

## HTTP Tool Clients

```python
# agent/tools/cluster_api.py
import httpx
import os

def call_cluster_model(patient_data: dict) -> dict:
    """Synchronous call to the Cluster Model API on Red Hat OpenShift."""
    url = os.getenv("CLUSTER_API_URL")
    if not url:
        raise ValueError("CLUSTER_API_URL environment variable not set")

    with httpx.Client(timeout=15.0) as client:
        response = client.post(url, json={"patient": patient_data})
        response.raise_for_status()
        return response.json()
        # Expected: {"cluster_profile": str, "cluster_confidence": float}
```

```python
# agent/tools/simulator_api.py
import httpx
import os

async def call_simulator(candidate_id: str, patient_data: dict) -> dict:
    """Async call to the Simulator Model API on Red Hat OpenShift."""
    url = os.getenv("SIMULATOR_API_URL")
    if not url:
        raise ValueError("SIMULATOR_API_URL environment variable not set")

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(url, json={
            "candidate_pill": candidate_id,
            "patient": patient_data
        })
        response.raise_for_status()
        return response.json()
        # Expected: {
        #   "discontinuation_probability": float,
        #   "severe_event_probability": float,
        #   "mild_side_effect_score": float,
        #   "contraceptive_effectiveness": float
        # }
```

Note: Tool clients are **plain functions**, not LangChain tools. In this architecture the LLM does not autonomously decide when to call model APIs — the graph structure dictates this. Plain functions keep the code simpler and the latency lower.

---

## Graph Assembly

```python
# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import SystemState
from agent.nodes import validator, cluster, candidate_gen, simulator, utility, convergence

def build_graph():
    g = StateGraph(SystemState)

    # Register nodes
    g.add_node("validate",           validator.run)
    g.add_node("assign_cluster",     cluster.run)
    g.add_node("generate_candidates", candidate_gen.run)
    g.add_node("simulate",           simulator.run)
    g.add_node("score_utility",      utility.run)
    g.add_node("check_convergence",  convergence.run)

    # Linear edges
    g.set_entry_point("validate")
    g.add_edge("validate",            "assign_cluster")
    g.add_edge("assign_cluster",      "generate_candidates")
    g.add_edge("generate_candidates", "simulate")
    g.add_edge("simulate",            "score_utility")
    g.add_edge("score_utility",       "check_convergence")

    # Conditional loop edge
    g.add_conditional_edges(
        "check_convergence",
        lambda state: END if state["converged"] else "simulate"
    )

    return g.compile()


# Singleton for use by FastAPI
agent_graph = build_graph()
```

### FastAPI Integration

```python
# In your FastAPI backend (main.py or router)
from agent.graph import agent_graph

@app.post("/recommend")
async def recommend(patient_input: PatientInput):
    initial_state = {
        "patient_data": patient_input.dict(),
        "iteration": 0,
        "converged": False,
        "cluster_profile": None,
        "cluster_confidence": None,
        "relative_risk_rules": [],
        "candidate_pool": [],
        "simulated_results": {},
        "utility_scores": {},
        "best_candidate": None,
        "previous_best_utility": None
    }

    final_state = await agent_graph.ainvoke(initial_state)

    return build_recommendation_output(final_state)
```

---

## Prompt Engineering Guidelines

Since the LLM's role is narrow and structured, prompts should enforce JSON output strictly.

**Do:**
- Always end prompts with "Return ONLY the JSON object/array, no other text."
- Keep context minimal — only what the LLM needs to reason
- Use `temperature=0` for all LLM calls (deterministic output)

**Don't:**
- Pass the full `SystemState` to the LLM — it doesn't need most of it
- Let the LLM make free-form medical decisions — keep it to reasoning about scores, confidence, and reason codes

**Output parsing:** Use `json.loads(response.content)` with a try/except. If parsing fails, log the raw output and either retry once or fall back to a default. Don't let a JSON parse failure crash a recommendation.

---

## Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = "^0.2"
langchain-core = "^0.3"
langchain-anthropic = "^0.2"     # Claude
langchain-ollama = "^0.1"        # Local Ollama
langchain-openai = "^0.2"        # Crusoe / OpenAI-compat
httpx = "^0.27"                  # Async HTTP for model API calls
fastapi = "^0.115"
uvicorn = "^0.30"
pydantic = "^2.0"

[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"
```

Install:

```bash
pip install langgraph langchain-core langchain-anthropic langchain-ollama langchain-openai httpx fastapi uvicorn
```

---

## LangSmith Tracing

Enable from day one — it's free and saves significant debugging time. Every node execution, LLM call, and state transition is logged and visualizable in the LangSmith UI.

```bash
# Add to .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_from_smith.langchain.com
LANGCHAIN_PROJECT=bc-agent-hackathon
```

No code changes needed. LangGraph picks this up automatically.

---

## Summary

| Concern | Solution |
|---|---|
| Iterative loop | LangGraph conditional edges |
| Persistent state | LangGraph `StateGraph` with `SystemState` TypedDict |
| LLM switching (local → Claude) | Factory pattern in `llm.py` driven by `LLM_PROVIDER` env var |
| Dev/staging LLM | Crusoe Managed Inference (`meta-llama/Llama-3.3-70B-Instruct`) via OpenAI-compat endpoint |
| Production LLM | `claude-sonnet-4-6` via `langchain-anthropic` |
| HTTP model API calls | `httpx` async clients in `agent/tools/` |
| Concurrent simulation | `asyncio.gather` across candidate pool |
| Safe gate enforcement | Handled by FastAPI backend before agent sees candidates |
| Debugging | LangSmith tracing (free tier) |
| Prompt outputs | Always JSON, `temperature=0`, `json.loads` with fallback |