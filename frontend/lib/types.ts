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
  color: string;
  features: Array<{ key: keyof SymptomProbs; label: string }>;
}

export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    id: "adherence",
    label: "Adherence",
    color: "#DDD8C4",
    features: [{ key: "still_taking", label: "Still taking" }],
  },
  {
    id: "mood",
    label: "Mood",
    color: "#C084FC",
    features: [
      { key: "sym_mood_worsened",    label: "Mood worsened" },
      { key: "sym_depression_episode", label: "Depression episode" },
      { key: "sym_anxiety",          label: "Anxiety" },
    ],
  },
  {
    id: "physical",
    label: "Physical",
    color: "#6EE7B7",
    features: [
      { key: "sym_nausea",           label: "Nausea" },
      { key: "sym_headache",         label: "Headache" },
      { key: "sym_bloating",         label: "Bloating" },
      { key: "sym_fatigue",          label: "Fatigue" },
      { key: "sym_weight_gain",      label: "Weight gain" },
      { key: "sym_breast_tenderness", label: "Breast tenderness" },
    ],
  },
  {
    id: "skin",
    label: "Skin & Hair",
    color: "#FCD34D",
    features: [
      { key: "sym_acne_worsened", label: "Acne worsened" },
      { key: "sym_hair_loss",     label: "Hair loss" },
    ],
  },
  {
    id: "bleeding",
    label: "Bleeding",
    color: "#FB7185",
    features: [
      { key: "sym_spotting",           label: "Spotting" },
      { key: "sym_irregular_bleeding", label: "Irregular bleeding" },
    ],
  },
  {
    id: "libido",
    label: "Libido",
    color: "#A5B4FC",
    features: [{ key: "sym_libido_decreased", label: "Libido decreased" }],
  },
  {
    id: "risk",
    label: "Serious Events",
    color: "#F87171",
    features: [
      { key: "evt_dvt",    label: "DVT" },
      { key: "evt_pe",     label: "Pulmonary embolism" },
      { key: "evt_stroke", label: "Stroke" },
    ],
  },
];
