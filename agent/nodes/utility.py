"""
Utility Node — computes a structured utility score for each simulated pill.

Formula:
    Utility = -α * severe_event_probability
              -β * discontinuation_probability
              -γ * mild_side_effect_score
              +δ * contraceptive_effectiveness

The agent CHOOSES the weights (α, β, γ, δ) stored in state['utility_weights'].
If no weights are set yet, use sensible defaults. The convergence node updates
these weights each iteration based on the agent's medical reasoning.
"""

import logging

from agent.state import SystemState

logger = logging.getLogger(__name__)

# ── Default utility weights (used only if agent hasn't set any yet) ────────
DEFAULT_WEIGHTS = {"alpha": 2.0, "beta": 1.5, "gamma": 0.5, "delta": 1.0}


def compute_utility(sim_result: dict, weights: dict) -> float:
    """Compute utility score for a single simulation result."""
    return (
        -weights["alpha"] * sim_result.get("severe_event_probability", 0.0)
        - weights["beta"] * sim_result.get("discontinuation_probability", 0.0)
        - weights["gamma"] * sim_result.get("mild_side_effect_score", 0.0)
        + weights["delta"] * sim_result.get("contraceptive_effectiveness", 0.0)
    )


def run(state: SystemState) -> dict:
    """
    Computes utility scores for ALL simulated pills (accumulated across
    iterations) using the agent's chosen weights.
    """
    simulated_results = state.get("simulated_results", {})

    if not simulated_results:
        logger.warning("No simulation results to score")
        return {"utility_scores": {}, "best_candidate": None}

    # Use agent-chosen weights, or defaults on first iteration
    weights = dict(state.get("utility_weights") or DEFAULT_WEIGHTS)

    # Apply low-confidence cluster weight adjustment if present.
    # When the cluster model is uncertain, we scale up the risk penalty terms
    # (alpha, beta) to be more conservative — i.e. penalise risky pills harder.
    cluster_info = state.get("cluster_profile")
    if isinstance(cluster_info, dict) and cluster_info.get("low_confidence"):
        adj = float(cluster_info.get("weight_adjustment", 1.0))
        if adj != 1.0:
            weights["alpha"] = weights["alpha"] * adj
            weights["beta"] = weights["beta"] * adj
            logger.info(
                "Low-confidence cluster adjustment (%.2f) applied → α=%.2f β=%.2f",
                adj, weights["alpha"], weights["beta"],
            )

    logger.info(
        "Scoring with weights: α=%.2f β=%.2f γ=%.2f δ=%.2f",
        weights["alpha"], weights["beta"], weights["gamma"], weights["delta"],
    )

    utility_scores = {
        pill_id: compute_utility(result, weights)
        for pill_id, result in simulated_results.items()
    }

    best_candidate = max(utility_scores, key=utility_scores.get)

    # Log top 5
    top5 = sorted(utility_scores.items(), key=lambda x: -x[1])[:5]
    logger.info("Top utilities: %s", {k: round(v, 4) for k, v in top5})
    logger.info("Best candidate: %s (utility=%.4f)", best_candidate, utility_scores[best_candidate])

    return {
        "utility_scores": utility_scores,
        "best_candidate": best_candidate,
    }
