# Selene - Every body reacts differently

AI-powered oral contraceptive recommendation engine. A medical-safety-first agentic loop selects the best pill for a patient, no LLM ever makes a safety decision.

Contraception is one of the most personal and consequential healthcare decisions a person can make. This system uses AI not to automate paperwork, but to give every patient access to the kind of careful, individualised clinical reasoning that is usually reserved for specialist consultations. **Claude (claude-sonnet-4-5)** is the agent powering this reasoning.


https://github.com/user-attachments/assets/ebff3b42-dfae-40a1-b527-e6b22a7c95c8


[![Live Demo](https://img.shields.io/badge/Live%20Demo-27selene.vercel.app-blue?style=for-the-badge)](https://27selene.vercel.app)

*Note: the pipeline will not work — we can't leave our Claude API key there forever, we are poor.*

---

## Agent

A **LangGraph agent** orchestrates the full pipeline as an iterative optimization loop:

```
validate → cluster → safe_gate → assess_risk → simulate → score_utility → converge
                                                    ↑_____________________|  (loop)
```

**Claude (claude-sonnet-4-5)** is the agent. It acts at three points only: selecting which pills to simulate based on the patient's clinical picture, adjusting the utility weight vector `{α, β, γ, δ}` to reflect what matters most for this specific patient, and deciding when the recommendation is ready. All medical safety filtering happens before Claude sees any candidates — deterministic WHO MEC Cat 3/4 rules filter contraindicated pills, and the agent works only on the safe remainder. The loop runs over and over, refining weights and re-simulating until the best candidate converges.

Unlike today’s agents focused on workflow efficiency, the goal here is simple: help real people navigate deeply personal decisions affecting their health, body, and daily quality of life.

---

## ML Model 1 — Patient Clustering

**Gaussian Mixture Model (GMM)**, k = 12 profiles, selected by BIC.  
Trained on 4,117 synthetic patients (80/20 split, stratified on contraindication status).

| Metric | Value |
|---|---|
| Safety Recall | **96.5%** |
| Mean assignment confidence | **99.9%** |
| Per-condition block rate | 94–100% |

The GMM assigns each patient a soft probability vector over 12 clinical archetypes (e.g. "PCOS + Hypertension", "Thrombophilia + Endometriosis", "Baseline / Low-Risk"). Profiles with high risk prevalence trigger population-level pill exclusions on top of the patient-specific hard rules.

---

## ML Model 2 — Pill Simulator

**HistGradientBoosting** (`MultiOutputClassifier` + `Regressor`), trained on **444,636 rows** (4,117 patients × 9 pills × 12 months).  
Predicts a full monthly trajectory of 18 symptom/event probabilities and a satisfaction score for any patient × pill combination.

| Target | AUROC |
|---|---|
| `sym_pcos_improvement` | **0.992** |
| `sym_cramps_relieved` | **0.987** |
| `sym_acne_improved` | **0.804** |
| `still_taking` (non-discontinuation) | **0.778** |
| Mean across all targets | **0.695** |

Satisfaction regression: RMSE 0.873, MAE 0.645. Native NaN handling covers PHQ-9 (64% missing) and testosterone (92% missing) — no imputation pipeline needed.

---

## Hosting — Hugging Face

During the hackathon, both models were deployed as **REST APIs on Red Hat OpenShift** — which earned us the **Best Use of Red Hat** award. We have since migrated the endpoints to **Hugging Face Spaces** to keep things running without the infrastructure overhead.

The current endpoint URLs are configured via environment variables — check your `.env` file:

| Tool | Endpoint | Purpose |
|---|---|---|
| `cluster_api.py` | `POST /cluster/predict` | Returns cluster profile + confidence |
| `simulator_api.py` | `POST /simulator/simulate` | Returns 12-month trajectory + summary metrics |

Simulator calls are fired **concurrently** (`asyncio.gather`) — all selected pills are simulated in parallel per iteration.

---

## Data

- **Synthea** synthetic patient records with custom modules for OCP-relevant conditions (PCOS, endometriosis, thrombophilia, migraine with aura)
- **FDA drug labels** (warnings, contraindications, boxed warnings) via OpenFDA API
- **FDA FAERS** adverse event rates per drug
- **Drugs.com** reviews for real-world tolerability signals
- Pill reference database: 9 formulations covering the full clinical spectrum of combined and progestin-only oral contraceptives

Full technical documentation: `docs/TechDocumentation.md`

---

## Market & Business Case

We studied the commercial viability of spinning this into a startup. The addressable market is large, over 150 million women use oral contraceptives globally, yet the prescribing process remains largely unchanged: a brief GP visit, a generic recommendation, and a trial-and-error period that can last months.

Our analysis covers market sizing, competitive landscape, go-to-market strategy, regulatory pathway considerations, and a monetisation model (B2B2C via gynaecology clinics and telehealth platforms). The full study is in `_docs/BusinessAnalysis.pdf`

