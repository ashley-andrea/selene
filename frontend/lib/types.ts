/* ── Patient input ──────────────────────────────────────────────────────── */
export interface PatientFormData {
  /* Required */
  age: number;

  /* Free-text arrays */
  pathologies: string[];
  habits: string[];
  medical_history: string[];

  /* Vitals — all optional */
  obs_bmi?: number | null;
  obs_systolic_bp?: number | null;
  obs_diastolic_bp?: number | null;
  obs_phq9_score?: number | null;
  obs_testosterone?: number | null;
  obs_smoker?: number;
  obs_pain_score?: number | null;

  /* WHO MEC Cat 4 conditions */
  cond_migraine_with_aura?: number;
  cond_stroke?: number;
  cond_mi?: number;
  cond_dvt?: number;
  cond_breast_cancer?: number;
  cond_lupus?: number;
  cond_thrombophilia?: number;
  cond_atrial_fibrillation?: number;
  cond_liver_disease?: number;

  /* WHO MEC Cat 3 conditions */
  cond_hypertension?: number;
  cond_migraine?: number;
  cond_gallstones?: number;
  cond_diabetes?: number;
  cond_prediabetes?: number;
  cond_epilepsy?: number;
  cond_chronic_kidney_disease?: number;
  cond_sleep_apnea?: number;

  /* Indications / comorbidities */
  cond_pcos?: number;
  cond_endometriosis?: number;
  cond_depression?: number;
  cond_hypothyroidism?: number;
  cond_rheumatoid_arthritis?: number;
  cond_fibromyalgia?: number;
  cond_osteoporosis?: number;
  cond_asthma?: number;

  /* OCP history */
  med_ever_ocp?: number;
  med_current_combined_ocp?: number;
  med_current_minipill?: number;
  has_absolute_contraindication_combined_oc?: number;
}

/* ── API response shapes ─────────────────────────────────────────────────── */
export interface SymptomProbs {
  still_taking: number[];
  sym_nausea: number[];
  sym_headache: number[];
  sym_breast_tenderness: number[];
  sym_spotting: number[];
  sym_mood_worsened: number[];
  sym_depression_episode: number[];
  sym_anxiety: number[];
  sym_libido_decreased: number[];
  sym_weight_gain: number[];
  sym_acne_worsened: number[];
  sym_hair_loss: number[];
  evt_dvt: number[];
  evt_pe: number[];
  evt_stroke: number[];
  sym_bloating: number[];
  sym_fatigue: number[];
  sym_irregular_bleeding: number[];
}

export interface PillRecommendation {
  pill_id: string;
  rank: number;
  utility_score: number;
  predicted_discontinuation: number;
  severe_risk: number;
  mild_side_effect_score: number;
  contraceptive_effectiveness: number;
  reason_codes: string[];
  months: number[];
  symptom_probs: SymptomProbs;
  satisfaction: number[];
}

export interface RecommendationResponse {
  recommendations: PillRecommendation[];
  cluster_profile: string;
  cluster_confidence: number;
  iterations: number;
  total_candidates_evaluated: number;
  selected_pill: string;
}

export interface PDFExtractionResult extends Partial<PatientFormData> {
  pathologies: string[];
  habits: string[];
  medical_history: string[];
  pages_parsed: number;
  parser_backend: string;
}

/* ── UI helpers ─────────────────────────────────────────────────────────── */
export type AppStep = "intake" | "loading" | "results";

/** Pill IDs → human-readable brand/INN names */
export const PILL_LABELS: Record<string, string> = {
  EE30_DRSP3:   "EE 30mcg + Drospirenone 3mg",
  EE30_LNG150:  "EE 30mcg + Levonorgestrel 150mcg",
  NET_PO_350:   "Norethisterone 350mcg (mini-pill)",
  EE20_DRSP3:   "EE 20mcg + Drospirenone 3mg",
  EE25_DSG150:  "EE 25mcg + Desogestrel 150mcg",
  EE30_DSG150:  "EE 30mcg + Desogestrel 150mcg",
  EE35_NET1000: "EE 35mcg + Norethisterone 1mg",
  EE35_NORG250: "EE 35mcg + Norgestimate 250mcg",
  LNG_IUD:      "Levonorgestrel IUS",
};

export const PILL_SHORT: Record<string, string> = {
  EE30_DRSP3:   "DRSP/EE30",
  EE30_LNG150:  "LNG/EE30",
  NET_PO_350:   "NET mini-pill",
  EE20_DRSP3:   "DRSP/EE20",
  EE25_DSG150:  "DSG/EE25",
  EE30_DSG150:  "DSG/EE30",
  EE35_NET1000: "NET/EE35",
  EE35_NORG250: "NORG/EE35",
  LNG_IUD:      "LNG-IUS",
};

/** Feature groups for the trajectory chart */
export interface FeatureGroup {
  id: string;
  label: string;
  color: string; // group accent for the selector tab
  features: Array<{ key: keyof SymptomProbs; label: string; color: string }>;
}

export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    id: "adherence",
    label: "Adherence",
    color: "#1A002E",
    features: [{ key: "still_taking", label: "Still taking", color: "#1A002E" }],
  },
  {
    id: "mood",
    label: "Mood",
    color: "#7767A4",
    features: [
      { key: "sym_mood_worsened",      label: "Mood worsened",      color: "#7767A4" },
      { key: "sym_depression_episode", label: "Depression episode",  color: "#35285A" },
      { key: "sym_anxiety",            label: "Anxiety",             color: "#9B8BC4" },
    ],
  },
  {
    id: "physical",
    label: "Physical",
    color: "#059669",
    features: [
      { key: "sym_nausea",            label: "Nausea",             color: "#059669" },
      { key: "sym_headache",          label: "Headache",           color: "#0891B2" },
      { key: "sym_bloating",          label: "Bloating",           color: "#10B981" },
      { key: "sym_fatigue",           label: "Fatigue",            color: "#64748B" },
      { key: "sym_weight_gain",       label: "Weight gain",        color: "#6B7280" },
      { key: "sym_breast_tenderness", label: "Breast tenderness",  color: "#0E7490" },
    ],
  },
  {
    id: "skin",
    label: "Skin & Hair",
    color: "#B45309",
    features: [
      { key: "sym_acne_worsened", label: "Acne worsened", color: "#B45309" },
      { key: "sym_hair_loss",     label: "Hair loss",     color: "#92400E" },
    ],
  },
  {
    id: "bleeding",
    label: "Bleeding",
    color: "#BE123C",
    features: [
      { key: "sym_spotting",           label: "Spotting",           color: "#BE123C" },
      { key: "sym_irregular_bleeding", label: "Irregular bleeding", color: "#E11D48" },
    ],
  },
  {
    id: "libido",
    label: "Libido",
    color: "#7C3AED",
    features: [{ key: "sym_libido_decreased", label: "Libido decreased", color: "#7C3AED" }],
  },
  {
    id: "risk",
    label: "Serious Events",
    color: "#DC2626",
    features: [
      { key: "evt_dvt",    label: "DVT",                 color: "#DC2626" },
      { key: "evt_pe",     label: "Pulmonary embolism",  color: "#991B1B" },
      { key: "evt_stroke", label: "Stroke",              color: "#7F1D1D" },
    ],
  },
];

/** Generic / INN names per pill ID */
export const PILL_GENERIC: Record<string, string> = {
  EE30_DRSP3:   "Drospirenone / Ethinylestradiol",
  EE30_LNG150:  "Levonorgestrel / Ethinylestradiol",
  NET_PO_350:   "Norethisterone (progestin-only)",
  EE20_DRSP3:   "Drospirenone / Ethinylestradiol (low dose)",
  EE25_DSG150:  "Desogestrel / Ethinylestradiol",
  EE30_DSG150:  "Desogestrel / Ethinylestradiol",
  EE35_NET1000: "Norethisterone / Ethinylestradiol",
  EE35_NORG250: "Norgestimate / Ethinylestradiol",
  LNG_IUD:      "Levonorgestrel intrauterine system",
};

/** Human-readable cluster profile descriptions */
export const CLUSTER_DESCRIPTIONS: Record<string, string> = {
  cluster_0:  "Profile 1 — Healthy baseline, no significant risk factors",
  cluster_1:  "Profile 2 — Mild mood sensitivity, good adherence history",
  cluster_2:  "Profile 3 — Metabolic concerns (BMI / blood pressure)",
  cluster_3:  "Profile 4 — Hormonal sensitivity pattern",
  cluster_4:  "Profile 5 — Active smoker or cardiovascular risk factors",
  cluster_5:  "Profile 6 — History of mood disorders",
  cluster_6:  "Profile 7 — Gynaecological indications (PCOS / endometriosis)",
  cluster_7:  "Profile 8 — Low adherence risk, high side-effect sensitivity",
  cluster_8:  "Profile 9 — Bleeding irregularity pattern",
  cluster_9:  "Profile 10 — Complex comorbidities, prior OCP use",
  cluster_10: "Profile 11 — Progestin-favourable hormonal profile",
  cluster_11: "Profile 12 — Adolescent or first-time contraceptive user",
};

/** Common side-effect highlights per pill (for patient-facing summary) */
export const PILL_SIDE_EFFECTS: Record<string, string[]> = {
  EE30_DRSP3:   ["Possible mood changes", "Blood pressure monitoring advised", "Mild fluid retention"],
  EE30_LNG150:  ["Androgenic effects possible (acne, oily skin)", "Reliable cycle control", "Weight-related concerns in some users"],
  NET_PO_350:   ["Irregular spotting, especially in first months", "Suitable during breastfeeding", "No oestrogen-related thrombotic risk"],
  EE20_DRSP3:   ["Lower oestrogen dose — less nausea", "Anti-androgenic benefits (acne, PCOS)", "Possible breakthrough bleeding"],
  EE25_DSG150:  ["Low androgenic profile", "Generally well-tolerated mood profile", "Possible mild headache"],
  EE30_DSG150:  ["Low androgenic activity", "Reliable cycle control", "Rare nausea / headache"],
  EE35_NET1000: ["Moderate androgenic activity", "Possible acne in sensitive users", "Well-established long-term profile"],
  EE35_NORG250: ["Low androgenic profile", "Possible breakthrough bleeding", "Rarely mood-related symptoms"],
  LNG_IUD:      ["Irregular bleeding especially in first 3–6 months", "Very low systemic hormone exposure", "Local progestin action only"],
};
