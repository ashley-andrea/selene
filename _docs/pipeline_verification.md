# Pipeline Implementation Verification

## Your Specified Pipeline → Implementation Mapping

### ✅ STEP 1: Base user profile
**Implementation:** 
- Node: [validator.py](../agent/nodes/validator.py)
- Flow: Entry point of graph
- Action: Validates patient input data (age, pathologies, habits, medical_history)
- Output: `patient_data` stored in state

---

### ✅ STEP 2: Agent calls the clustering model
**Implementation:**
- Node: [cluster.py](../agent/nodes/cluster.py)
- API: [cluster_api.py](../agent/tools/cluster_api.py) → calls external ML clustering model
- Flow: `validate` → `assign_cluster`
- Action: Sends patient data to clustering API
- Output: `cluster_profile` (e.g. "cluster_0") and `cluster_confidence` (0.0-1.0)

---

### ✅ STEP 3: User profile updated with cluster
**Implementation:**
- Node: [cluster.py](../agent/nodes/cluster.py)
- Flow: Updates state after API call
- Action: Stores cluster assignment in state
- Output: State now contains:
  - `cluster_profile`: cluster identifier
  - `cluster_confidence`: confidence score
  - Low-confidence assignments trigger LLM weight adjustment

---

### ✅ STEP 4: Each cluster has associated pills that are NOT good for that cluster (safety mechanism filters pills)
**Implementation:**
- **Configuration:** [cluster_exclusions.py](../agent/cluster_exclusions.py) ← **NEW**
  - `CLUSTER_EXCLUSIONS` dict maps cluster → excluded pill IDs
  - Easy to configure for each cluster
- **Enforcement:** [safe_gate.py](../agent/safe_gate.py) `apply_safe_gate()`
  - Layer 1: Hard constraints (patient-specific contraindications)
  - Layer 2: Cluster exclusions (population-level risk patterns) ← **NEW**
- **Node:** [candidate_gen.py](../agent/nodes/candidate_gen.py)
- **Flow:** `assign_cluster` → `generate_candidates`
- **Action:** 
  1. Applies hard constraints (age, pathologies, habits)
  2. Applies cluster-specific exclusions ← **NEW**
  3. Returns filtered `candidate_pool` (all pills - excluded pills)
- **Output:** Agent ONLY sees pills in `candidate_pool` (good pills for this patient+cluster)

---

### ✅ STEP 5: Agent calculates the risk for each pill
**Implementation:**
- Node: [risk_assessor.py](../agent/nodes/risk_assessor.py)
- Flow: `generate_candidates` → `assess_risk`
- Action: LLM evaluates each candidate pill considering:
  - Patient data (age, pathologies, habits, history)
  - Cluster profile
  - Pill characteristics (estrogen dose, progestin type, VTE risk, generation)
  - Relative risk rules from safe gate
- Output: `risk_scores` dict with detailed risk assessment per pill

---

### ✅ STEP 6: Agent chooses the pills to simulate (technically the ones with lower risks)
**Implementation:**
- Node: [risk_assessor.py](../agent/nodes/risk_assessor.py)
- Flow: Same node as risk calculation (combined decision)
- Action: LLM selects subset of lowest-risk pills to simulate
- Output: `pills_to_simulate` list (ordered by risk, typically 3-5 pills)
- Note: On iterations > 0, convergence node can request specific pills to re-simulate

---

### ✅ STEP 7: Agent calls the ML model to simulate effects in time (one pill at a time)
**Implementation:**
- Node: [simulator.py](../agent/nodes/simulator.py)
- API: [simulator_api.py](../agent/tools/simulator_api.py) → calls external ML simulator
- Flow: `assess_risk` → `simulate`
- Action: For each pill in `pills_to_simulate`:
  - Calls simulator API with pill_id + patient_data
  - Receives temporal simulation results
- Output: `simulated_results` dict with per-pill outcomes:
  - `discontinuation_probability`
  - `severe_event_probability`
  - `mild_side_effect_score`
  - `contraceptive_effectiveness`
  - `temporal_data` (time-series of effects)
- Note: Results accumulate across iterations

---

### ✅ STEP 8: Agent gets the result for each and chooses if he has to stop or loop again
**Implementation:**
- Node: [convergence.py](../agent/nodes/convergence.py)
- Flow: `score_utility` → `check_convergence` → (END or loop to `assess_risk`)
- Action: LLM analyzes utility scores and decides:
  - **CONVERGE:** Stop and return recommendation
  - **CONTINUE:** Loop back to assess_risk with:
    - New utility weights
    - Specific pills to reconsider
- Output: 
  - `converged` boolean
  - If continuing: `utility_weights` (new), `pills_to_simulate` (for next iteration)
  - If converged: `reason_codes` for the top recommendation
- Max iterations: 10

---

### ✅ STEP 9: Agent chooses the weights of the formula
**Implementation:**
- **Node:** [convergence.py](../agent/nodes/convergence.py) (weight decision)
- **Node:** [utility.py](../agent/nodes/utility.py) (weight application)
- **Flow:** `simulate` → `score_utility` → `check_convergence`
- **Formula:**
  ```
  Utility = -α × severe_event_probability
            -β × discontinuation_probability
            -γ × mild_side_effect_score
            +δ × contraceptive_effectiveness
  ```
- **Action:** 
  - First iteration: Uses default weights (α=2.0, β=1.5, γ=0.5, δ=1.0)
  - Each iteration: LLM analyzes results and adjusts weights based on:
    - Patient priorities
    - Observed trade-offs in simulation results
    - Medical best practices
- **Output:** `utility_weights` dict `{alpha, beta, gamma, delta}`

---

### ✅ STEP 10: When it converges, returns top 3 pills with: risk score, simulation plot, effects, explanation
**Implementation:**
- Function: [main.py](../main.py) `_build_output()`
- Flow: After convergence, transforms final state → API response
- Output: `RecommendationOutput` with:
  - **recommendations** (list of top 3 pills, each containing):
    - `pill_id`
    - `rank` (1, 2, 3)
    - `utility_score`
    - `predicted_discontinuation`
    - `severe_risk`
    - `mild_side_effect_score`
    - `contraceptive_effectiveness`
    - `reason_codes` (human-readable explanations)
  - **Metadata:**
    - `cluster_profile`
    - `cluster_confidence`
    - `iterations` (how many loops)
    - `total_candidates_evaluated`
  - **Note:** Simulation plots (temporal_data) are available in simulated_results for frontend rendering

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  POST /recommend (patient data)                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  1. VALIDATE         │  ← Normalize patient input
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  2. ASSIGN_CLUSTER   │  ← Call clustering ML model
          └──────────┬───────────┘     (cluster_profile + confidence)
                     │
                     ▼
          ┌──────────────────────┐
          │  3. GENERATE_         │  ← Safe Gate Engine:
          │     CANDIDATES        │     • Hard constraints (patient)
          └──────────┬───────────┘     • Cluster exclusions ← NEW
                     │                  • Relative risk rules
                     │                → candidate_pool (filtered)
                     │
                     ▼
          ┌──────────────────────┐
    ┌─────│  4. ASSESS_RISK      │  ← Agent calculates risk scores
    │     └──────────┬───────────┘     Selects pills to simulate
    │                │
    │                ▼
    │     ┌──────────────────────┐
    │     │  5. SIMULATE         │  ← Call simulator ML model
    │     └──────────┬───────────┘     (one pill at a time)
    │                │
    │                ▼
    │     ┌──────────────────────┐
    │     │  6. SCORE_UTILITY    │  ← Apply agent-chosen weights
    │     └──────────┬───────────┘     Compute utility scores
    │                │
    │                ▼
    │     ┌──────────────────────┐
    │     │  7. CHECK_           │  ← Agent decides:
    │     │     CONVERGENCE      │     STOP or CONTINUE?
    │     └──────────┬───────────┘     New weights? Reconsider pills?
    │                │
    │                ├─── converged = False ───┐
    └────────────────┘                         │
                     │                         │ (loop back)
                     │ converged = True        │
                     ▼                         │
          ┌──────────────────────┐            │
          │  RETURN TOP 3 PILLS  │            │
          │  + risk scores       │◄───────────┘
          │  + simulation data   │
          │  + explanations      │
          └──────────────────────┘
```

---

## Key Implementation Points

### ✅ Agent Control Points (LLM decisions)
1. **Risk Assessment** ([risk_assessor.py](../agent/nodes/risk_assessor.py))
   - Evaluates medical risk for each pill
   - Selects which pills to simulate
   
2. **Weight Selection** ([convergence.py](../agent/nodes/convergence.py))
   - Chooses utility function weights
   - Balances safety vs effectiveness based on patient context
   
3. **Convergence Decision** ([convergence.py](../agent/nodes/convergence.py))
   - Decides when to stop iterating
   - Generates reason codes for recommendation

### ✅ Deterministic Safety Layer
- **No LLM involvement** in safety filtering
- **Two stages:**
  1. Hard constraints (patient contraindications)
  2. Cluster exclusions (population risk patterns) ← **NEW**
- **Agent never sees excluded pills**

### ✅ ML Model Integration
- **Cluster Model:** Assigns patient to cluster
- **Simulator Model:** Predicts outcomes per pill
- **API clients:** [cluster_api.py](../agent/tools/cluster_api.py), [simulator_api.py](../agent/tools/simulator_api.py)
- **Mocks available:** [tests/mocks/](../tests/mocks/) for testing without ML APIs

### ✅ State Management
- **SystemState** ([state.py](../agent/state.py)): Typed state flowing through graph
- **Persistent across nodes:** LangGraph handles state passing
- **Accumulates results:** Simulation results persist across iterations

---

## Configuration

### To customize cluster-specific exclusions:

Edit [agent/cluster_exclusions.py](../agent/cluster_exclusions.py):

```python
CLUSTER_EXCLUSIONS = {
    "cluster_0": [
        "pill_cyproterone_35",    # High VTE risk
        "pill_norethisterone_35",  # High estrogen dose
    ],
    "cluster_1": [
        "pill_drospirenone_30",
        "pill_drospirenone_20",
    ],
    # ... add more clusters
}
```

The configuration is commented with examples for each cluster type.

---

## Summary

**Your pipeline is NOW fully implemented!** 

All 10 steps are present and working together:
1. ✅ Patient validation
2. ✅ Clustering model call
3. ✅ State update with cluster
4. ✅ Safety filtering (hard constraints + cluster exclusions) ← **NEWLY ADDED**
5. ✅ Risk calculation (agent)
6. ✅ Pill selection for simulation (agent)
7. ✅ Simulation model calls (one at a time)
8. ✅ Convergence decision (agent)
9. ✅ Weight optimization (agent)
10. ✅ Top 3 pills with full details

The main addition was **cluster-specific pill filtering** in the Safe Gate Engine, which is now fully integrated and configurable.
