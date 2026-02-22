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
import re
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

    # Build top-3 pill data so the LLM can generate per-pill explanations and
    # make a fully-informed convergence decision (not just the #1 pill).
    all_simulated = state.get("simulated_results", {})
    top3_items = sorted(utility_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    all_simulations_data = {
        pid: {"utility_score": round(score, 4), **all_simulated.get(pid, {})}
        for pid, score in top3_items
    }

    decision = _ask_llm(
        iteration=iteration,
        best_candidate=best_candidate,
        best_utility=best_utility,
        previous_utility=previous_utility,
        all_utilities=utility_scores,
        all_simulations=all_simulations_data,
        cluster_profile=cluster_profile,
        cluster_confidence=cluster_confidence,
        patient_data=patient_data,
        current_weights=current_weights,
    )

    # Default to False — a parse failure should NOT prematurely end the loop.
    # Only converge when explicitly requested by the LLM or the fallback.
    converged = decision.get("converged", False)
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
    top3_reason_codes = decision.get("top3_reason_codes", {})
    logger.info(
        "🧠 AGENT: Converge at iter %d | best=%s (%.4f) | %s",
        iteration + 1, best_candidate, best_utility, medical_rationale,
    )
    return {
        "converged": True,
        "iteration": iteration + 1,
        "previous_best_utility": best_utility,
        "reason_codes": reason_codes,
        "top3_reason_codes": top3_reason_codes,
    }


# ── LLM invocation ─────────────────────────────────────────────────────────

def _strip_markdown_json(text: str) -> str:
    """Extract the first JSON object/array from an LLM response.

    Handles:
    - Bare JSON
    - JSON inside ```json ... ``` fences (with or without preamble text)
    - JSON inside ```plaintext ... ``` or any other fence language
    - Fences that have trailing newlines before the closing ```
    """
    text = text.strip()

    # 1. Try to extract from a code fence anywhere in the text
    fence_match = re.search(r"```(?:[\w+\-]*)\s*\n([\s\S]*?)\n?```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Try to find a raw JSON object / array anywhere in the text
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_match:
        return json_match.group(1).strip()

    # 3. Return stripped original and let json.loads raise a clear error
    return text


def _ask_llm(
    iteration: int,
    best_candidate: str,
    best_utility: float,
    previous_utility: float | None,
    all_utilities: dict,
    all_simulations: dict,
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
            all_simulations=json.dumps(all_simulations, indent=2),
            previous_utility=prev_str,
            all_utility_scores=all_utils_str,
            alpha=current_weights.get("alpha", DEFAULT_WEIGHTS["alpha"]),
            beta=current_weights.get("beta", DEFAULT_WEIGHTS["beta"]),
            gamma=current_weights.get("gamma", DEFAULT_WEIGHTS["gamma"]),
            delta=current_weights.get("delta", DEFAULT_WEIGHTS["delta"]),
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        decision = json.loads(_strip_markdown_json(response.content))

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
    """Simple rule-based convergence when LLM is unavailable.

    Conservative policy: keep iterating until we hit MAX_ITERATIONS.
    Only converge early if utility has truly plateaued AND we are past
    the midpoint of allowed iterations.
    """
    improvement = (
        abs(best_utility - previous_utility) if isinstance(previous_utility, (int, float)) else float("inf")
    )
    # Give the agent at least half the budget before allowing a fallback-converge.
    past_midpoint = iteration >= MAX_ITERATIONS // 2
    plateau = improvement < 0.01
    should_converge = (iteration >= MAX_ITERATIONS) or (past_midpoint and plateau)

    if not should_converge:
        logger.warning(
            "Fallback (iter %d): LLM parse failed — continuing with default weights.", iteration
        )
        return {
            "converged": False,
            "confidence": 0.4,
            "medical_rationale": "Fallback: LLM unavailable, continuing iteration.",
            "new_weights": DEFAULT_WEIGHTS,
            "pills_to_reconsider": [],
            "reason_codes": [],
            "top3_reason_codes": {},
        }

    logger.warning(
        "Fallback (iter %d): converging by rule (plateau=%s, max_iter=%s).",
        iteration, plateau, iteration >= MAX_ITERATIONS,
    )
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
        "top3_reason_codes": {},
    }
