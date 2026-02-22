"""
SystemState — the typed state object that flows through the LangGraph graph.

Every node reads from and writes to this state. LangGraph manages persistence
and passing between nodes automatically.

IMPORTANT: hard_constraints are NEVER stored here. The Safe Gate Engine filters
the candidate pool before the agent sees it. The agent operates only on the
filtered pool and has no visibility into what was excluded.
"""

from typing import TypedDict, Optional


class SystemState(TypedDict):
    """Persistent state flowing through the agent graph."""

    # ── Patient input ───────────────────────────────────────────────────────
    patient_data: dict  # Validated and normalized patient record

    # ── Cluster assignment ──────────────────────────────────────────────────
    cluster_profile: Optional[str]       # e.g. "cluster_3" or enriched dict when low-confidence
    cluster_confidence: Optional[float]  # 0.0 – 1.0

    # ── Safety layer outputs ────────────────────────────────────────────────
    relative_risk_rules: list  # Soft rules for candidate ranking

    # ── Candidate pool (already filtered by Safe Gate Engine) ───────────────
    candidate_pool: list  # List of pill IDs from allowed set only

    # ── Risk assessment (agent-computed) ────────────────────────────────────
    risk_scores: Optional[dict]       # {pill_id: {risk_score, risk_factors, ...}}
    pills_to_simulate: Optional[list]  # Ordered by risk — agent-selected subset

    # ── Simulation results ──────────────────────────────────────────────────
    simulated_results: dict  # {pill_id: {disc_prob, severe_prob, mild_score, effectiveness, temporal_data}}

    # ── Utility scores ──────────────────────────────────────────────────────
    utility_scores: dict           # {pill_id: float}
    utility_weights: Optional[dict]  # Agent-chosen: {alpha, beta, gamma, delta}
    best_candidate: Optional[str]  # Pill ID with highest utility

    # ── Loop control ────────────────────────────────────────────────────────
    iteration: int
    converged: bool
    previous_best_utility: Optional[float]  # For convergence comparison

    # ── Output ──────────────────────────────────────────────────────────────
    reason_codes: list  # Human-readable explanations for the #1 recommendation
    top3_reason_codes: Optional[dict]  # {pill_id: [reasons]} for top-3 pills (LLM-generated)

