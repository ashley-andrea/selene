"use client";

import type { PatientFormData, RecommendationResponse } from "@/lib/types";

/* WHO MEC blocking rules — mirrors the clustering model's blocking logic */

const CAT4_CONDITIONS: Array<{ key: keyof PatientFormData; label: string }> = [
  { key: "cond_migraine_with_aura", label: "Migraine with aura" },
  { key: "cond_stroke",             label: "Stroke (history)" },
  { key: "cond_mi",                 label: "Heart attack (MI)" },
  { key: "cond_dvt",                label: "DVT / thromboembolism" },
  { key: "cond_breast_cancer",      label: "Breast cancer" },
  { key: "cond_lupus",              label: "Systemic lupus erythematosus" },
  { key: "cond_thrombophilia",      label: "Thrombophilia" },
  { key: "cond_atrial_fibrillation",label: "Atrial fibrillation" },
  { key: "cond_liver_disease",      label: "Liver disease" },
];

const CAT3_CONDITIONS: Array<{ key: keyof PatientFormData; label: string }> = [
  { key: "cond_hypertension",          label: "Hypertension" },
  { key: "cond_migraine",              label: "Migraine (without aura)" },
  { key: "cond_gallstones",            label: "Gallstones" },
  { key: "cond_diabetes",              label: "Diabetes" },
  { key: "cond_prediabetes",           label: "Pre-diabetes" },
  { key: "cond_epilepsy",              label: "Epilepsy" },
  { key: "cond_chronic_kidney_disease",label: "Chronic kidney disease" },
  { key: "cond_sleep_apnea",           label: "Sleep apnea" },
];

interface Props {
  patient: PatientFormData;
  response: RecommendationResponse;
}

export default function SafetyGate({ patient, response }: Props) {
  const cat4Present = CAT4_CONDITIONS.filter((c) => patient[c.key] === 1);
  const cat3Present = CAT3_CONDITIONS.filter((c) => patient[c.key] === 1);

  const blocked   = response.total_candidates_evaluated - response.recommendations.length;
  const hasSmoker = patient.obs_smoker === 1;

  return (
    <div className="space-y-4">
      {/* Summary row */}
      <div
        className="flex flex-wrap items-center gap-3 rounded-2xl px-6 py-5"
        style={{
          background: "rgba(255,255,255,0.85)",
          border: "1px solid rgba(53,40,90,0.1)",
        }}
      >
        {/* Check */}
        <div
          className="flex items-center justify-center w-9 h-9 rounded-full flex-shrink-0"
          style={{ background: "rgba(110,231,183,0.12)", border: "1px solid rgba(110,231,183,0.3)" }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path
              d="M3 8l3.5 3.5 6.5-6.5"
              stroke="#6EE7B7"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-body font-semibold text-sm" style={{ color: "#1A002E" }}>
            Safety screening complete
          </p>
          <p className="font-body text-xs" style={{ color: "rgba(26,0,46,0.55)" }}>
            {response.total_candidates_evaluated} candidates evaluated ·{" "}
            <span style={{ color: "#6EE7B7" }}>
              {response.recommendations.length} passed
            </span>
            {blocked > 0 && (
              <span style={{ color: "#F87171" }}> · {blocked} excluded</span>
            )}
          </p>
        </div>

        {/* Cluster badge */}
        <div
          className="text-xs font-body px-3 py-1 rounded-full"
          style={{
            background: "rgba(119,103,164,0.1)",
            color: "#1A002E",
            border: "1px solid rgba(119,103,164,0.25)",
          }}
        >
          {response.cluster_profile} ·{" "}
          {(response.cluster_confidence * 100).toFixed(0)}% conf.
        </div>
      </div>

      {/* Cat 4 banner */}
      {cat4Present.length > 0 && (
        <div
          className="rounded-2xl px-6 py-5"
          style={{
            background: "rgba(248,113,113,0.08)",
            border: "1px solid rgba(248,113,113,0.25)",
          }}
        >
          <p className="font-body font-semibold text-sm mb-3" style={{ color: "#F87171" }}>
            ⚠ WHO MEC Category 4 — combined pills excluded
          </p>
          <div className="flex flex-wrap gap-2">
            {cat4Present.map((c) => (
              <span
                key={c.key}
                className="text-xs font-body px-2.5 py-1 rounded-full"
                style={{
                  background: "rgba(248,113,113,0.12)",
                  color: "#FCA5A5",
                  border: "1px solid rgba(248,113,113,0.2)",
                }}
              >
                {c.label}
              </span>
            ))}
          </div>
          <p className="font-body text-xs mt-3" style={{ color: "rgba(26,0,46,0.55)" }}>
            All combined estrogen–progestin pills are contraindicated for these
            conditions. Progestin-only options remain available.
          </p>
        </div>
      )}

      {/* Cat 3 banner */}
      {cat3Present.length > 0 && (
        <div
          className="rounded-2xl px-6 py-5"
          style={{
            background: "rgba(252,211,77,0.07)",
            border: "1px solid rgba(252,211,77,0.2)",
          }}
        >
          <p className="font-body font-semibold text-sm mb-3" style={{ color: "#FCD34D" }}>
            ⚡ WHO MEC Category 3 — use with monitoring
          </p>
          <div className="flex flex-wrap gap-2">
            {cat3Present.map((c) => (
              <span
                key={c.key}
                className="text-xs font-body px-2.5 py-1 rounded-full"
                style={{
                  background: "rgba(252,211,77,0.1)",
                  color: "#78350F",
                  border: "1px solid rgba(252,211,77,0.3)",
                }}
              >
                {c.label}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Smoker note for combined pills age-related risk */}
      {hasSmoker && (patient.age ?? 0) >= 35 && (
        <div
          className="rounded-2xl px-6 py-4"
          style={{
            background: "rgba(251,113,133,0.07)",
            border: "1px solid rgba(251,113,133,0.2)",
          }}
        >
          <p className="font-body text-sm" style={{ color: "#FDA4AF" }}>
            ⚠ Smoker aged ≥ 35: combined estrogen pills carry elevated
            cardiovascular risk (WHO MEC Cat 4). Progestin-only options
            are prioritised.
          </p>
        </div>
      )}
    </div>
  );
}
