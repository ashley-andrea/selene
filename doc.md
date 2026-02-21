## Birth Control Recommendation Agent: Architecture and Loop Documentation

### Overview

This document details the architecture and operational flow of a safe, agentic AI system designed to recommend oral contraceptives. The system operates fully under the hood—there is no chat interface for the user. The AI agent orchestrates input processing, medical safety enforcement, candidate evaluation, simulation, and iterative optimization to deliver medically reliable recommendations.

The core principles guiding this design are:

1. **Agentic AI**: The agent actively reasons, maintains state, adapts candidate selection, and iteratively optimizes outcomes.
2. **Medical Safety**: Hard constraints and safe gates prevent hallucinations and ensure recommendations align with validated medical guidelines.
3. **Simplicity of ML**: ML models are simple and interpretable, used primarily for abstraction and simulation, not creative reasoning.

---

### System Components

1. **Intake Validator**

   * **Function**: Validates and normalizes patient input data.
   * **Inputs**: Age, medical history, habits, pathologies.
   * **Outputs**: Structured `patient_data` object.

2. **Cluster Model**

   * **Function**: Abstracts the patient into a cluster.
   * **Tool**: Simple clustering ML model (avaible via API on red hat) .
   * **Outputs**: `cluster_profile`, `cluster_confidence`
   * **Agent Role**: Evaluate cluster assignment and flag low-confidence cases for additional weighting in later steps.

3. **Safe Gate Engine (Hard Constraints)**

   * **Function**: Applies deterministic medical rules to filter contraindicated pills.
   * **Inputs**: `patient_data`, `cluster_profile`
   * **Outputs**: `hard_constraints`, `relative_risk_rules`
   * **Agent Role**: Enforce safety rules; the agent must never see the list of excluded pills (`hard_constraints`) directly and should only operate on the filtered candidate pool without considering blocked pills.

4. **Pill Database**

   * **Function**: Provides structured data on available oral contraceptives.
   * **Agent Role**: Generate candidate pills only from the allowed set; the agent should not see or reason about pills blocked by `hard_constraints` and must treat the filtered list as the full universe of options.

5. **Simulator Model**

   * **Function**: Predicts patient-specific outcomes over time for each candidate pill.
   * **Model Type**: ML model avaible via API (hosted on red-hat)
   * **Outputs**:

     * X-day discontinuation probability
     * Severe adverse event probability
     * Mild side-effect score
   * **Agent Role**: Evaluate and compare candidates quantitatively.

6. **Utility Calculator**

   * **Function**: Computes a structured utility score for each candidate based on simulated outcomes.
   * **Formula**: `Utility = -α*severe_event - β*discontinuation - γ*mild_side_effect + δ*contraceptive_effectiveness`
   * **Agent Role**: Optimize pill selection within safety constraints.

7. **Convergence Checker**

   * **Function**: Determines if the optimization loop has reached an acceptable solution.
   * **Criteria**: Utility improvement below epsilon, maximum iterations reached, or risk thresholds satisfied.
   * **Agent Role**: Decide whether to continue looping or output final recommendation.

---

### Master State Object

The agent maintains a persistent structured state throughout the loop:

```python
SystemState = {
    'patient_data': {},
    'cluster_profile': None,
    'cluster_confidence': None,
    'relative_risk_rules': [],
    'candidate_pool': [],
    'simulated_results': {},
    'utility_scores': {},
    'best_candidate': None,
    'iteration': 0,
    'converged': False
}
```

The LLM never sees `hard_constraints` directly; it operates only on the `candidate_pool`, which already reflects all necessary exclusions. All reasoning is formalized on this filtered state, ensuring the agent cannot consider prohibited pills and cannot be influenced by them.

---

### Operational Loop

1. **Input Validation**

   * Validate and normalize `patient_data`.

2. **Cluster Assignment**

   * Apply cluster model to assign `cluster_profile`.
   * Record confidence score.

3. **Apply Hard Safe Gate**

   * Generate `hard_constraints` and `relative_risk_rules` based on cluster profile and patient data.
   * Filter prohibited pills.

4. **Candidate Generation**

   * Query pill database to generate initial `candidate_pool`.
   * Rank candidates using cluster-level priors and relative risk rules.

5. **Simulation**

   * Simulate individual outcomes for each candidate using `simulator_model`.
   * Store results in `simulated_results`.

6. **Utility Optimization**

   * Compute `utility_scores` for each candidate.
   * Select `best_candidate` based on highest utility within constraints.

7. **Convergence Check**

   * Evaluate improvement vs. previous iteration.
   * If converged: exit loop.
   * If not converged: refine candidate pool, increment iteration, repeat from simulation.

8. **Output**

* Structured recommendation showing the top 3 pills with simulation graphs over time and explainable reason codes for each choice

  * Structured recommendation:

```python
{
  'selected_pill': best_candidate_id,
  'predicted_discontinuation': simulated_results[best_candidate]['discontinuation_30d'],
  'severe_risk': simulated_results[best_candidate]['severe_event_probability'],
  'reason_codes': ['lowest thrombotic risk', 'fits patient cluster', ...]
}
```

---

### Safety and Compliance Measures

* **Hard constraints** are immutable and sourced from validated medical guidelines.
* **Simulator predictions** are probabilistic, not generative, and are fully structured.
* **Agent decisions** are numeric and structured; no text hallucination affects medical advice.
* **Iterative loop** ensures risk reduction is quantifiable and optimized before output.

---

### Agentic Characteristics

* Maintains persistent, structured state across iterations.
* Actively evaluates, adapts, and optimizes candidate selection.
* Balances cluster-level risk priors and patient-specific simulation outputs.
* Controls stopping criteria through explicit convergence logic.
* Adapts weights dynamically in low-confidence cluster cases.

The agent thus embodies true agency: reasoning, adapting, and optimizing, while strictly enforcing safety and reproducibility.

---

This design ensures a **medically safe, fully automated, agent-driven recommendation system** for contraceptive selection, where intelligence resides in looped reasoning and structured optimization rather than LLM creativity or overengineered ML.
