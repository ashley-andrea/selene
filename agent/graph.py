"""
Graph Assembly — wires all agent nodes into a LangGraph StateGraph.

Canonical pipeline:
    validate → assign_cluster → generate_candidates → assess_risk
    → simulate → score_utility → check_convergence
    → (loop back to assess_risk OR END)

The agent (LLM) is the protagonist at three decision points:
  1. assess_risk  – computes risk per pill, selects which to simulate
  2. score_utility – applies agent-chosen weights
  3. check_convergence – decides stop/loop, sets new weights & pills to reconsider
"""

import logging

from langgraph.graph import END, StateGraph

from agent.nodes import candidate_gen, cluster, convergence, risk_assessor, simulator, utility, validator
from agent.state import SystemState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph agent graph.

    Returns a compiled graph ready to be invoked with an initial SystemState.
    """
    g = StateGraph(SystemState)

    # ── Register nodes ──────────────────────────────────────────────────────
    g.add_node("validate", validator.run)
    g.add_node("assign_cluster", cluster.run)
    g.add_node("generate_candidates", candidate_gen.run)
    g.add_node("assess_risk", risk_assessor.run)  # Agent: risk-score each pill, select subset
    g.add_node("simulate", simulator.run)          # Call simulator API for selected pills
    g.add_node("score_utility", utility.run)        # Apply agent-chosen weights
    g.add_node("check_convergence", convergence.run)  # Agent: stop or loop?

    # ── Linear edges ────────────────────────────────────────────────────────
    g.set_entry_point("validate")
    g.add_edge("validate", "assign_cluster")
    g.add_edge("assign_cluster", "generate_candidates")
    g.add_edge("generate_candidates", "assess_risk")
    g.add_edge("assess_risk", "simulate")
    g.add_edge("simulate", "score_utility")
    g.add_edge("score_utility", "check_convergence")

    # ── Conditional loop edge ───────────────────────────────────────────────
    # Converged → END, otherwise loop back to assess_risk so the agent can
    # re-evaluate which pills to simulate with new context.
    g.add_conditional_edges(
        "check_convergence",
        lambda state: "end" if state.get("converged") else "assess_risk",
        {
            "end": END,
            "assess_risk": "assess_risk",
        },
    )

    logger.info("Agent graph compiled successfully")
    return g.compile()


# Singleton instance for use by FastAPI
agent_graph = build_graph()
