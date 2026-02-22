"""
Risk Assessment & Pill Selection Node — The agent evaluates each candidate
pill's risk for this specific patient and decides which pills to simulate.

This is a KEY agentic step: instead of blindly simulating everything,
the agent uses its medical reasoning to:
1. Compute a patient-specific risk score for each candidate
2. Select the most promising (lowest risk) pills for simulation
3. Provide medical reasoning for why these were selected

This replaces the old "planner" node with a more principled approach:
risk assessment THEN selection, not just arbitrary strategic planning.
"""

import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.pill_database import get_pill_by_id
from agent.state import SystemState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "risk_assessment.txt"


def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text()


def run(state: SystemState) -> dict:
    """
    The agent assesses the risk of each candidate pill for this patient
    and selects which pills to send to the simulator.

    On iteration 0 (first pass): Full risk assessment of all candidates.
    On later iterations: Re-evaluate based on simulation results + agent adjustments.
    """
    candidate_pool = state.get("candidate_pool", [])
    patient_data = state.get("patient_data", {})
    cluster_info = state.get("cluster_profile", {})
    cluster_confidence = state.get("cluster_confidence", 0.0)
    relative_risk_rules = state.get("relative_risk_rules", [])
    iteration = state.get("iteration", 0)
    previous_sim_results = state.get("simulated_results", {})

    # Resolve cluster profile
    if isinstance(cluster_info, dict):
        cluster_profile = cluster_info.get("profile", "unknown")
    else:
        cluster_profile = cluster_info or "unknown"

    # Build detailed pill info for each candidate using real pills.csv fields
    candidate_details = []
    for pill_id in candidate_pool:
        pill = get_pill_by_id(pill_id)
        if pill:
            substance = pill.get('substance_name', 'unknown')
            is_combined = 'ETHINYL ESTRADIOL' in str(substance).upper()
            pill_type = 'combined (estrogen+progestin)' if is_combined else 'progestin-only'
            
            # Include a short excerpt from warnings/contraindications for LLM context
            warnings_text = str(pill.get('warnings', '') or '').strip()[:300]
            contraind_text = str(pill.get('contraindications', '') or '').strip()[:200]
            boxed_text = str(pill.get('boxed_warning', '') or '').strip()[:200]
            
            candidate_details.append(
                f"  - {pill_id}: {pill.get('brand_name', '?')} / {pill.get('generic_name', '?')}\n"
                f"    type={pill_type}, substance={substance}\n"
                f"    boxed_warning={boxed_text or 'none'}\n"
                f"    contraindications_excerpt={contraind_text or 'none'}\n"
                f"    warnings_excerpt={warnings_text or 'none'}"
            )
        else:
            candidate_details.append(f"  - {pill_id}: (details unavailable)")

    # On re-iterations, include previous simulation results as context
    prev_results_context = ""
    if iteration > 0 and previous_sim_results:
        prev_results_context = "\n\n## Previous Simulation Results\n"
        for pill_id, res in previous_sim_results.items():
            prev_results_context += (
                f"  - {pill_id}: disc={res.get('discontinuation_probability', '?')}, "
                f"severe={res.get('severe_event_probability', '?')}, "
                f"mild={res.get('mild_side_effect_score', '?')}, "
                f"eff={res.get('contraceptive_effectiveness', '?')}\n"
            )

    # Convergence may have requested specific pills to reconsider
    convergence_hint = state.get("pills_to_simulate") or []
    if convergence_hint:
        prev_results_context += (
            f"\n\n## Convergence Agent Request\n"
            f"The convergence agent specifically asked to re-simulate: {convergence_hint}\n"
            f"Include these pills in your selection.\n"
        )

    assessment = _assess_risks(
        patient_data=patient_data,
        cluster_profile=cluster_profile,
        cluster_confidence=cluster_confidence,
        relative_risk_rules=relative_risk_rules,
        candidate_details="\n".join(candidate_details) + prev_results_context,
        candidate_pool=candidate_pool,
    )

    risk_scores = assessment.get("risk_assessments", {})
    pills_to_simulate = assessment.get("pills_to_simulate", candidate_pool[:5])
    rationale = assessment.get("simulation_rationale", "")

    # Validate pills are in the pool
    pills_to_simulate = [p for p in pills_to_simulate if p in candidate_pool]

    # Ensure convergence-requested pills are always included
    for p in convergence_hint:
        if p in candidate_pool and p not in pills_to_simulate:
            pills_to_simulate.append(p)

    if not pills_to_simulate:
        logger.warning("LLM selected no valid pills — falling back to top 5 candidates")
        pills_to_simulate = candidate_pool[:5]

    logger.info(
        "🧠 RISK ASSESSMENT: Scored %d pills, selected %d for simulation — %s",
        len(risk_scores),
        len(pills_to_simulate),
        pills_to_simulate,
    )
    if rationale:
        logger.info("🧠 Selection rationale: %s", rationale[:120])

    return {
        "risk_scores": risk_scores,
        "pills_to_simulate": pills_to_simulate,
    }


def _assess_risks(
    patient_data: dict,
    cluster_profile: str,
    cluster_confidence: float,
    relative_risk_rules: list,
    candidate_details: str,
    candidate_pool: list,
) -> dict:
    """Invoke the LLM to assess risks and select pills for simulation."""
    try:
        llm = get_llm()
        template = _load_prompt_template()

        # Format relative risk rules
        if relative_risk_rules:
            rules_str = json.dumps(relative_risk_rules, indent=2)
        else:
            rules_str = "None (no additional risk modifiers)"

        prompt = template.format(
            patient_age=patient_data.get("age", "unknown"),
            patient_pathologies=", ".join(patient_data.get("pathologies", [])) or "none",
            patient_habits=", ".join(patient_data.get("habits", [])) or "none",
            patient_medical_history=", ".join(patient_data.get("medical_history", [])) or "none",
            cluster_profile=cluster_profile,
            cluster_confidence=cluster_confidence,
            relative_risk_rules=rules_str,
            candidate_pills_detail=candidate_details,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content)

        logger.info(
            "✓ Risk assessment complete: %d pills scored, %d selected for simulation",
            len(result.get("risk_assessments", {})),
            len(result.get("pills_to_simulate", [])),
        )

        return result

    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.error("Failed to get LLM risk assessment: %s — using fallback", exc)

        # Fallback: score by pill type (combined vs progestin-only) and patient risk factors
        risk_scores = {}
        patient_has_smoking = "smoking" in patient_data.get("habits", [])
        patient_has_vte = any(h in patient_data.get("medical_history", []) for h in ["dvt", "pe", "thrombosis"])
        patient_has_cv_risk = any(p in patient_data.get("pathologies", []) for p in ["hypertension", "migraines", "diabetes"])
        
        for pill_id in candidate_pool:
            pill = get_pill_by_id(pill_id)
            if pill:
                substance = str(pill.get("substance_name", "")).upper()
                is_combined = "ETHINYL ESTRADIOL" in substance
                # Higher base risk for combined pills if patient has CV/VTE risk factors
                if is_combined and (patient_has_vte or patient_has_smoking or patient_has_cv_risk):
                    base_risk = 0.6
                elif is_combined:
                    base_risk = 0.3
                else:
                    base_risk = 0.2  # progestin-only are generally safer
                risk_scores[pill_id] = {
                    "risk_score": base_risk,
                    "risk_factors": ["combined_pill" if is_combined else "progestin_only"],
                    "recommendation_priority": "low" if base_risk > 0.5 else "medium",
                }
            else:
                risk_scores[pill_id] = {
                    "risk_score": 0.5,
                    "risk_factors": ["unknown"],
                    "recommendation_priority": "low",
                }

        # Sort by risk, take top 5
        sorted_pills = sorted(risk_scores.items(), key=lambda x: x[1]["risk_score"])
        pills_to_simulate = [p[0] for p in sorted_pills[:5]]

        return {
            "risk_assessments": risk_scores,
            "pills_to_simulate": pills_to_simulate,
            "simulation_rationale": "Fallback: sorted by VTE risk level from pill database",
            "medical_reasoning": "LLM risk assessment failed; using pill database VTE risk as proxy",
        }
