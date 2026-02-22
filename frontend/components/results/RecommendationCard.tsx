"use client";

import type { PillRecommendation, RecommendationResponse } from "@/lib/types";
import { PILL_LABELS, PILL_SHORT } from "@/lib/types";

interface Props {
  top: PillRecommendation;
  response: RecommendationResponse;
}

export default function RecommendationCard({ top, response }: Props) {
  const adherence12m = top.symptom_probs.still_taking.at(-1) ?? 0;
  const sat12m = top.satisfaction.at(-1) ?? 0;

  return (
    <div
      className="relative rounded-3xl overflow-hidden p-8 md:p-10"
      style={{
        background:
          "radial-gradient(ellipse 120% 80% at 50% 0%, rgba(119,103,164,0.18) 0%, rgba(53,40,90,0.35) 60%)",
        border: "1px solid rgba(221,216,196,0.25)",
      }}
    >
      {/* Decorative glow */}
      <div
        className="absolute top-0 right-0 w-64 h-64 pointer-events-none"
        style={{
          background:
            "radial-gradient(circle, rgba(221,216,196,0.06) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10">
        {/* Label */}
        <div className="flex items-center gap-3 mb-6">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{
              background: "rgba(221,216,196,0.1)",
              border: "1px solid rgba(221,216,196,0.25)",
            }}
          >
            {/* Moon icon */}
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M13 8.5A5 5 0 1 1 7.5 3 3.5 3.5 0 0 0 13 8.5z"
                stroke="#DDD8C4"
                strokeWidth="1.2"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <span className="section-label">Selene's recommendation</span>
        </div>

        {/* Pill name */}
        <h2
          className="font-display mb-1"
          style={{
            fontFamily: "Catchye, 'Cormorant Garamond', Georgia, serif",
            fontSize: "clamp(1.6rem, 4vw, 2.4rem)",
            color: "#DDD8C4",
            lineHeight: 1.1,
          }}
        >
          {PILL_SHORT[top.pill_id] ?? top.pill_id}
        </h2>
        <p
          className="font-body text-sm mb-8"
          style={{ color: "rgba(221,216,196,0.45)" }}
        >
          {PILL_LABELS[top.pill_id] ?? top.pill_id}
        </p>

        {/* Stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            {
              label: "Utility score",
              value: `${(top.utility_score * 100).toFixed(0)}/100`,
              color: "#DDD8C4",
            },
            {
              label: "12-month adherence",
              value: `${(adherence12m * 100).toFixed(0)}%`,
              color: "#6EE7B7",
            },
            {
              label: "Severe risk",
              value:
                top.severe_risk < 0.005
                  ? "<0.5%"
                  : `${(top.severe_risk * 100).toFixed(1)}%`,
              color: top.severe_risk < 0.01 ? "#6EE7B7" : "#F87171",
            },
            {
              label: "Satisfaction (mo. 12)",
              value: `${sat12m.toFixed(1)} / 10`,
              color: "#C084FC",
            },
          ].map((s) => (
            <div key={s.label}>
              <p
                className="font-body text-xs mb-1"
                style={{ color: "rgba(221,216,196,0.45)" }}
              >
                {s.label}
              </p>
              <p
                className="font-body font-semibold text-xl"
                style={{ color: s.color }}
              >
                {s.value}
              </p>
            </div>
          ))}
        </div>

        {/* Reason codes */}
        {top.reason_codes.length > 0 && (
          <div className="mb-8">
            <p className="section-label mb-3">Why Selene chose this</p>
            <ul className="space-y-2">
              {top.reason_codes.map((r) => (
                <li key={r} className="flex items-start gap-2">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 14 14"
                    fill="none"
                    className="mt-0.5 flex-shrink-0"
                  >
                    <path
                      d="M2.5 7l3 3 6-6"
                      stroke="#7767A4"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <span
                    className="font-body text-sm"
                    style={{ color: "rgba(221,216,196,0.75)" }}
                  >
                    {r}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Meta row */}
        <div
          className="flex flex-wrap gap-3 pt-6"
          style={{ borderTop: "1px solid rgba(119,103,164,0.18)" }}
        >
          {[
            { k: "Cluster", v: response.cluster_profile },
            {
              k: "Confidence",
              v: `${(response.cluster_confidence * 100).toFixed(0)}%`,
            },
            { k: "Agent iterations", v: String(response.iterations) },
            {
              k: "Candidates evaluated",
              v: String(response.total_candidates_evaluated),
            },
          ].map((m) => (
            <div
              key={m.k}
              className="text-xs font-body px-3 py-1.5 rounded-full"
              style={{
                background: "rgba(53,40,90,0.6)",
                border: "1px solid rgba(119,103,164,0.2)",
              }}
            >
              <span style={{ color: "rgba(221,216,196,0.4)" }}>{m.k}: </span>
              <span style={{ color: "#DDD8C4" }}>{m.v}</span>
            </div>
          ))}
        </div>

        {/* Disclaimer */}
        <p
          className="font-body text-xs mt-6"
          style={{ color: "rgba(221,216,196,0.3)" }}
        >
          This recommendation is generated by a clinical decision-support
          model. Always consult a healthcare provider before making changes to
          your contraception.
        </p>
      </div>
    </div>
  );
}
