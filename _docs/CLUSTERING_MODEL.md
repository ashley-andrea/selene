# Clustering Model — Reference Card

## Model

Gaussian Mixture Model (GMM, `covariance_type=diag`), **k = 12** profiles selected by BIC over k ∈ [3, 12].  
Inference threshold **θ = 0.40** — a pill is blocked if P(profile_k | patient) > 0.40 for any profile k that blocks it.  
Train/test split: **80 / 20**, stratified on `has_absolute_contraindication_combined_ocp`.

---

## Inputs

| Group             | Features                                                                                                                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Continuous**    | `age`, `obs_bmi`, `obs_systolic_bp`, `obs_diastolic_bp`, `obs_phq9_score`, `obs_testosterone`                                                                               |
| **WHO MEC Cat 4** | `cond_migraine_with_aura`, `cond_stroke`, `cond_mi`, `cond_dvt`, `cond_breast_cancer`, `cond_lupus`, `cond_thrombophilia`, `cond_atrial_fibrillation`, `cond_liver_disease` |
| **WHO MEC Cat 3** | `cond_hypertension`, `cond_migraine`, `cond_gallstones`, `cond_diabetes`, `cond_prediabetes`, `cond_epilepsy`, `cond_chronic_kidney_disease`, `cond_sleep_apnea`            |
| **Indications**   | `cond_pcos`, `cond_endometriosis`                                                                                                                                           |
| **Comorbidities** | `cond_depression`, `cond_hypothyroidism`, `cond_rheumatoid_arthritis`, `cond_fibromyalgia`, `cond_osteoporosis`, `cond_asthma`, `obs_smoker`                                |

Missing continuous values → median imputation (training medians). Missing binary values → 0 (conservative absence assumption).

---

## Outputs

| Output                    | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `P(profile_k \| patient)` | Soft membership vector, length 12, sums to 1          |
| `hard_profile`            | argmax of the above                                   |
| `n_profiles_triggered`    | Number of profiles with P > θ                         |
| `blocked_combos`          | Union of blocked pill IDs from all triggered profiles |
| `n_blocked`               | Count of blocked pills (0–9)                          |

---

## Blocking Rules per Profile

Policy: **Cat 3 and Cat 4 conditions both trigger hard blocks** (conservative).  
Centroid thresholds: Cat 4 triggers at ≥ 15% cluster prevalence; Cat 3 at ≥ 25%.

| #   | Profile                        | Train n (%)  | Dominant conditions                         | Blocked pills                                |
| --- | ------------------------------ | ------------ | ------------------------------------------- | -------------------------------------------- |
| 0   | Rheum. Arthritis + Sleep Apnea | 14 (0.3%)    | RA, Sleep Apnea, Hypertension               | All 9 (incl. minipill — liver disease rule)  |
| 1   | Migraine + Depression          | 661 (16.1%)  | Migraine, Depression, Endometriosis         | **None**                                     |
| 2   | Diabetes + Breast Cancer       | 66 (1.6%)    | Diabetes, Breast Cancer, Depression         | All 9 (breast cancer blocks minipill too)    |
| 3   | Hypertension + Diabetes        | 57 (1.4%)    | Hypertension, Diabetes, Endometriosis       | All combined OCPs (8); minipill available    |
| 4   | Thrombophilia + Endometriosis  | 138 (3.4%)   | Thrombophilia, Endometriosis, Migraine+Aura | All combined OCPs (8); minipill available    |
| 5   | Baseline / Low-Risk            | 2403 (58.4%) | —                                           | **None**                                     |
| 6   | PCOS + Thrombophilia           | 13 (0.3%)    | PCOS, Thrombophilia, Epilepsy               | All 9 (epilepsy + thrombophilia combination) |
| 7   | Epilepsy                       | 65 (1.6%)    | Epilepsy                                    | All combined OCPs (8); minipill available    |
| 8   | PCOS + Hypertension            | 72 (1.7%)    | PCOS, Hypertension, Migraine                | All combined OCPs (8); minipill available    |
| 9   | PCOS                           | 239 (5.8%)   | PCOS                                        | **None**                                     |
| 10  | Hypertension + Migraine+Aura   | 360 (8.7%)   | Hypertension, Migraine+Aura                 | All combined OCPs (8); minipill available    |
| 11  | Hypertension + Thrombophilia   | 29 (0.7%)    | Hypertension, Thrombophilia, Diabetes       | All combined OCPs (8); minipill available    |

> **Combined OCPs (8):** `EE20_LNG90`, `EE30_LNG150`, `EE35_NET500_1000`, `EE20_NET1000`, `EE25_35_NGM`, `EE30_DSG150`, `EE30_DRSP3`, `EE20_DRSP3`  
> **Minipill (1):** `NET_PO_350`

---

## Test-set performance

| Metric                     | Value     | Note                                                                      |
| -------------------------- | --------- | ------------------------------------------------------------------------- |
| Safety Recall              | **96.5%** | 3 FN out of 85 abs. contraindicated patients                              |
| Safety Precision           | 41.4%     | Conservative over-blocking is acceptable at this stage                    |
| Per-condition block rate   | 94–100%   | All 6 audited conditions ≥ 94%                                            |
| Mean assignment confidence | 99.9%     | Near-zero entropy — patients cleanly assigned                             |
| Silhouette score           | −0.02     | Expected near-zero on sparse binary feature space; not a validity concern |
