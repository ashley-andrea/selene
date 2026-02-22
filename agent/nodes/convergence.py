"""
Convergence Node — LLM-driven optimization controller.

The agent analyzes simulation results and utility scores, then decides:
1. Should we STOP (converge) or CONTINUE iterating?
2. If continuing: what new utility weights to use next iteration?
3. If continuing: which pills to reconsider / re-simulate?
4. If converging: generate reason codes for the top recommendation.

This is the core agentic loop — the LLM controls the entire optimization.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.state import SystemState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "convergence_decision.txt"

DEFAULT_WEIGHTS = {"alpha": 2.0, "beta": 1.5, "gamma": 0.5, "delta": 1.0}


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text()


def run(state: SystemState) -> dict:
    """
    LLM convergence controller.  Returns state updates:
    - converged, iteration, previous_best_utility  (always)
    - utility_weights, pills_to_simulate            (when continuing)
    - reason_codes                                   (when converging)
    """
    iteration = state.get("iteration", 0)
    best_candidate = state.get("best_candidate")
    utility_scores = state.get("utility_scores", {})
    best_utility = utility_scores.get(best_candidate, 0.0) if best_candidate else 0.0
    previous_utility = state.get("previous_best_utility")
    patient_data = state.get("patient_data", {})
    cluster_info = state.get("cluster_profile", {})
    cluster_confidence = state.get("cluster_confidence", 0.0)
    current_weights = state.get("utility_weights") or DEFAULT_WEIGHTS

    # Resolve cluster profile string
    cluster_profile = (
        cluster_info.get("profile", "unknown") if isinstance(cluster_info, dict) else cluster_info
    )

    # Best candidate's simulation data
    best_sim = state.get("simulated_results", {}).get(best_candidate, {})

    decision = _ask_llm(
        iteration=iteration,
        best_candidate=best_candidate,
        best_utility=best_utility,
        previous_utility=previous_utility,
        all_utilities=utility_scores,
        best_simulation=best_sim,
        cluster_profile=cluster_profile,
        cluster_confidence=cluster_confidence,
        patient_data=patient_data,
        current_weights=current_weights,
    )

    converged = decision.get("converged", True)
    medical_rationale = decision.get("medical_rationale", "")

    if not converged and iteration < MAX_ITERATIONS:
        new_weights = decision.get("new_weights", current_weights)
        pills_to_reconsider = decision.get("pills_to_reconsider", [])

        logger.info(
            "🧠 AGENT: Continue (iter %d) | weights=%s | reconsider=%s | %s",
            iteration + 1, new_weights, pills_to_reconsider, medical_rationale,
        )
        return {
            "converged": False,
            "iteration": iteration + 1,
            "previous_best_utility": best_utility,
            "utility_weights": new_weights,
            "pills_to_simulate": pills_to_reconsider if pills_to_reconsider else None,
        }

    # ── Converge ────────────────────────────────────────────────────────
    reason_codes = decision.get("reason_codes", [])
    logger.info(
        "🧠 AGENT: Converge at iter %d | best=%s (%.4f) | %s",
        iteration + 1, best_candidate, best_utility, medical_rationale,
    )
    return {
        "converged": True,
        "iteration": iteration + 1,
        "previous_best_utility": best_utility,
        "reason_codes": reason_codes,
    }


# ── LLM invocation ─────────────────────────────────────────────────────────

def _ask_llm(
    iteration: int,
    best_candidate: str,
    best_utility: float,
    previous_utility: float | None,
    all_utilities: dict,
    best_simulation: dict,
    cluster_profile: str,
    cluster_confidence: float,
    patient_data: dict,
    current_weights: dict,
) -> dict:
    """
    Format the convergence prompt, call the LLM, parse JSON response.
    Falls back to rule-based logic on failure.
    """
    try:
        llm = get_llm()
        template = _load_prompt_template()

        prev_str = (
            f"{previous_utility:.4f}" if isinstance(previous_utility, (int, float)) else "None (first iteration)"
        )
        all_utils_str = json.dumps(
            {k: round(v, 4) for k, v in sorted(all_utilities.items(), key=lambda x: x[1], reverse=True)},
            indent=2,
        )

        habits_parts = []
        if patient_data.get("smoker"):
            habits_parts.append("smoker")
        if patient_data.get("bmi"):
            habits_parts.append(f"BMI {patient_data['bmi']}")
        habits_str = ", ".join(habits_parts) or "none reported"

        prompt = template.format(
            patient_age=patient_data.get("age", "unknown"),
            patient_pathologies=", ".join(patient_data.get("pathologies", [])) or "none",
            patient_habits=habits_str,
            cluster_profile=cluster_profile,
            cluster_confidence=cluster_confidence,
            iteration=iteration + 1,
            max_iterations=MAX_ITERATIONS,
            best_candidate=best_candidate,
            best_utility=round(best_utility, 4),
            best_simulation=json.dumps(best_simulation, indent=2),
            previous_utility=prev_str,
            all_utility_scores=all_utils_str,
            alpha=current_weights.get("alpha", DEFAULT_WEIGHTS["alpha"]),
            beta=current_weights.get("beta", DEFAULT_WEIGHTS["beta"]),
            gamma=current_weights.get("gamma", DEFAULT_WEIGHTS["gamma"]),
            delta=current_weights.get("delta", DEFAULT_WEIGHTS["delta"]),
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = json.loads(response.content)

        logger.info(
            "✓ LLM convergence: converged=%s confidence=%.2f",
            decision.get("converged"), decision.get("confidence", 0.0),
        )
        return decision

    except Exception as exc:
        logger.error("LLM convergence failed: %s — fallback", exc)
        return _fallback_decision(
            iteration, best_utility, previous_utility, best_candidate, cluster_profile,
        )


def _fallback_decision(
    iteration: int,
    best_utility: float,
    previous_utility: float | None,
    best_candidate: str,
    cluster_profile: str,
) -> dict:
    """Simple rule-based convergence when LLM is unavailable."""
    improvement = (
        abs(best_utility - previous_utility) if isinstance(previous_utility, (int, float)) else float("inf")
    )
    should_continue = iteration < MAX_ITERATIONS and improvement > 0.01

    if should_continue:
        return {
            "converged": False,
            "confidence": 0.5,
            "medical_rationale": "Fallback: utility still improving, continue.",
            "new_weights": DEFAULT_WEIGHTS,
            "pills_to_reconsider": [],
            "reason_codes": [],
        }

    return {
        "converged": True,
        "confidence": 0.5,
        "medical_rationale": "Fallback: converged by rule (plateau or max iter).",
        "new_weights": DEFAULT_WEIGHTS,
        "pills_to_reconsider": [],
        "reason_codes": [
            f"utility {round(best_utility, 3)} for {best_candidate}",
            f"cluster: {cluster_profile}",
            "passes all safety constraints",
        ],
    }
