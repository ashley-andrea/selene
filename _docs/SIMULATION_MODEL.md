# Simulator Model — Reference Card

## What it does

Given one patient's clinical features and one pill identifier, the model predicts a **monthly trajectory** of symptom probabilities and satisfaction score over a configurable horizon (default 12 months). It is called once per pill by the agent; the agent compares trajectories across the safe pill candidates produced by the clustering model's safety gate.

---

## Model

Two HistGradientBoosting models trained on 444,636 rows (4,117 train patients × 9 pills × 12 months):

| Model                    | Type                             | Targets                                |
| ------------------------ | -------------------------------- | -------------------------------------- |
| `model_symptoms.pkl`     | `MultiOutputClassifier(HistGBM)` | 18 binary symptom / event flags        |
| `model_satisfaction.pkl` | `HistGBMRegressor`               | `satisfaction_score` (1–10 continuous) |

`month` (1…N) is passed as a plain numeric feature, so any horizon up to 12 months can be requested at inference without retraining.  
NaN features (PHQ-9: 64% missing, testosterone: 92% missing) are handled natively — no imputation pipeline required.

---

## Inputs

### Patient features (37 total)

| Group                        | Features                                                                                                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Continuous vitals**        | `age`, `obs_bmi`, `obs_systolic_bp`, `obs_diastolic_bp`, `obs_phq9_score`, `obs_testosterone`                                                                               |
| **Binary vitals**            | `obs_smoker`                                                                                                                                                                |
| **WHO MEC Cat 4 conditions** | `cond_migraine_with_aura`, `cond_stroke`, `cond_mi`, `cond_dvt`, `cond_breast_cancer`, `cond_lupus`, `cond_thrombophilia`, `cond_atrial_fibrillation`, `cond_liver_disease` |
| **WHO MEC Cat 3 conditions** | `cond_hypertension`, `cond_migraine`, `cond_gallstones`, `cond_diabetes`, `cond_prediabetes`, `cond_epilepsy`, `cond_chronic_kidney_disease`, `cond_sleep_apnea`            |
| **Indications**              | `cond_pcos`, `cond_endometriosis`                                                                                                                                           |
| **Comorbidities**            | `cond_depression`, `cond_hypothyroidism`, `cond_rheumatoid_arthritis`, `cond_fibromyalgia`, `cond_osteoporosis`, `cond_asthma`                                              |
| **OCP history**              | `med_ever_ocp`                                                                                                                                                              |

### Pill features (6, auto-derived from `pill_reference_db.csv`)

| Feature                | Encoding                                              |
| ---------------------- | ----------------------------------------------------- |
| `pill_type_binary`     | 1 = combined, 0 = progestin-only                      |
| `estrogen_dose_mcg`    | 0, 20, 25, 30, 35                                     |
| `progestin_dose_mg`    | numeric                                               |
| `progestin_generation` | 1–4 ordinal                                           |
| `androgenic_score`     | anti-androgenic = −1, low = 1, moderate = 2, high = 3 |
| `vte_risk_numeric`     | very_low = 1 … high = 5                               |

### Temporal feature

- `month` — integer 1…N_months (default 12, configurable via `--n-months`)

---

## Outputs

### `predict_trajectory(patient_row, combo_id, ..., n_months=12)` returns:

```json
{
  "combo_id": "EE30_DRSP3",
  "n_months": 12,
  "months": [1, 2, ..., 12],
  "symptom_probs": {
    "sym_nausea":            [0.021, 0.018, ...],
    "sym_mood_worsened":     [0.031, 0.028, ...],
    "sym_acne_improved":     [0.042, 0.071, ...],
    "sym_cramps_relieved":   [0.083, 0.091, ...],
    "still_taking":          [0.91,  0.88,  ...],
    "evt_dvt":               [0.000, 0.000, ...]
    // ... 18 targets total
  },
  "satisfaction": [6.1, 6.3, 6.5, ...]
}
```

All `symptom_probs` values are P(event = 1 | patient, pill, month) ∈ [0, 1].

---

## Test-set performance (1,030 held-out patients)

### Binary targets — AUROC highlights

| Target                          | AUROC     | Note                                                               |
| ------------------------------- | --------- | ------------------------------------------------------------------ |
| `sym_pcos_improvement`          | **0.992** | PCOS flag is a direct driver                                       |
| `sym_cramps_relieved`           | **0.987** | Endometriosis flag + progestin generation                          |
| `sym_acne_improved`             | **0.804** | Anti-androgenic score (DRSP)                                       |
| `sym_spotting`                  | **0.786** | Progestin-only flag + dose                                         |
| `still_taking`                  | **0.778** | Satisfaction + profile interaction                                 |
| Mood / MH signals               | 0.62–0.67 | Harder — PHQ-9 64% missing, multi-condition                        |
| `evt_dvt / evt_pe / evt_stroke` | n/a       | ~0 prevalence in test set — primary gate is WHO MEC blocking layer |

Mean AUROC across all meaningful targets: **0.695**

### Satisfaction regression

| RMSE  | MAE   | R²   |
| ----- | ----- | ---- |
| 0.873 | 0.645 | 0.35 |

The 65% unexplained variance reflects the inherent stochasticity of which symptoms realise in any given month (the model predicts expected probability, not random draws).
