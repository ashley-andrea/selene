"""
Strategic Planning Node — The agent decides WHAT to simulate and WHY.

This is the ORCHESTRATION core: The LLM looks at the candidates and patient context,
then strategically decides which pills to simulate and in what priority.

This makes the agent PROACTIVE rather than reactive to ML outputs.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.state import SystemState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "simulation_strategy.txt"


def _load_prompt_template() -> str:
    """Load the simulation strategy prompt template from file."""
    return _PROMPT_PATH.read_text()


def run(state: SystemState) -> dict:
    """
    The LLM strategically plans which candidates to simulate based on:
    - Patient medical profile
    - Cluster assignment
    - Available candidates
    - Previous iteration results (if any)
    
    This makes the agent the PROTAGONIST who orchestrates ML API calls.
    """
    iteration = state.get("iteration", 0)
    patient_data = state.get("patient_data", {})
    candidate_pool = state.get("candidate_pool", [])
    cluster_info = state.get("cluster_profile", {})
    cluster_confidence = state.get("cluster_confidence", 0.0)
    previous_results = state.get("simulated_results", {})
    
    # Resolve cluster profile
    if isinstance(cluster_info, dict):
        cluster_profile = cluster_info.get("profile", "unknown")
    else:
        cluster_profile = cluster_info
    
    # Make strategic decision
    strategy = _plan_simulation_strategy(
        iteration=iteration,
        patient_data=patient_data,
        candidate_pool=candidate_pool,
        cluster_profile=cluster_profile,
        cluster_confidence=cluster_confidence,
        previous_results=previous_results,
    )
    
    simulation_strategy = strategy.get("simulation_strategy", "EXPLORE_ALL")
    candidates_to_simulate = strategy.get("candidates_to_simulate", "all")
    medical_rationale = strategy.get("medical_rationale", "")
    
    # Log the strategic decision
    if candidates_to_simulate == "all":
        logger.info(
            "🧠 LLM STRATEGY [iteration %d]: %s | Simulating all %d candidates | Rationale: %s",
            iteration + 1,
            simulation_strategy,
            len(candidate_pool),
            medical_rationale[:100],
        )
        candidates_list = candidate_pool
    else:
        logger.info(
            "🧠 LLM STRATEGY [iteration %d]: %s | Simulating %d targeted candidates | Rationale: %s",
            iteration + 1,
            simulation_strategy,
            len(candidates_to_simulate) if isinstance(candidates_to_simulate, list) else 0,
            medical_rationale[:100],
        )
        # Ensure requested candidates are in the pool
        candidates_list = [c for c in candidates_to_simulate if c in candidate_pool]
        
        if not candidates_list:
            logger.warning("LLM requested candidates not in pool — falling back to all")
            candidates_list = candidate_pool
    
    return {
        "simulation_strategy": simulation_strategy,
        "simulation_candidates": candidates_list,  # New field: what the agent decided to simulate
        "simulation_priority": strategy.get("simulation_priority", {}),
    }


def _plan_simulation_strategy(
    iteration: int,
    patient_data: dict,
    candidate_pool: list,
    cluster_profile: str,
    cluster_confidence: float,
    previous_results: dict,
) -> dict:
    """
    Invokes the LLM to strategically plan the simulation approach.
    Falls back to exploring all candidates if LLM fails.
    """
    try:
        llm = get_llm()
        template = _load_prompt_template()
        
        # Format previous results for context
        if previous_results:
            prev_results_str = json.dumps(
                {
                    pill_id: {
                        "discontinuation": round(res.get("discontinuation_probability", 0), 3),
                        "severe_risk": round(res.get("severe_event_probability", 0), 4),
                        "effectiveness": round(res.get("contraceptive_effectiveness", 0), 2),
                    }
                    for pill_id, res in list(previous_results.items())[:5]  # Top 5
                },
                indent=2,
            )
        else:
            prev_results_str = "None (first iteration)"
        
        prompt = template.format(
            patient_age=patient_data.get("age", "unknown"),
            patient_pathologies=", ".join(patient_data.get("pathologies", [])) or "none",
            patient_habits=", ".join(patient_data.get("habits", [])) or "none",
            patient_medical_history=", ".join(patient_data.get("medical_history", [])) or "none",
            cluster_profile=cluster_profile,
            cluster_confidence=cluster_confidence,
            candidate_pool="\n".join([f"  - {pill}" for pill in candidate_pool]),
            iteration=iteration + 1,
            max_iterations=10,  # TODO: import from convergence.py
            previous_results=prev_results_str,
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        strategy = json.loads(response.content)
        
        logger.info(
            "✓ LLM planned simulation strategy: %s (expected outcome: %s)",
            strategy.get("simulation_strategy", "UNKNOWN"),
            strategy.get("expected_outcome", "not specified")[:80],
        )
        
        return strategy
        
    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.error("Failed to get LLM simulation strategy: %s — exploring all candidates", exc)
        
        # Fallback: explore all
        return {
            "simulation_strategy": "EXPLORE_ALL",
            "candidates_to_simulate": "all",
            "simulation_priority": {
                "prioritize_safety": 0.7,
                "prioritize_effectiveness": 0.5,
                "prioritize_low_side_effects": 0.5,
            },
            "medical_rationale": "Fallback strategy: exploring all candidates due to LLM failure",
            "expected_outcome": "Broad exploration of available options",
        }
