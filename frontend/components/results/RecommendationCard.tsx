"use client";

import type { PillRecommendation, RecommendationResponse } from "@/lib/types";
import {
  PILL_LABELS,
  PILL_SHORT,
  PILL_GENERIC,
  CLUSTER_DESCRIPTIONS,
  PILL_SIDE_EFFECTS,
} from "@/lib/types";

interface Props {
  top: PillRecommendation;
  response: RecommendationResponse;
}

function SideEffectTag({ text }: { text: string }) {
  return (
    <span
      className="inline-block text-xs font-body px-2.5 py-1 rounded-full"
      style={{
        background: "rgba(53,40,90,0.06)",
        border: "1px solid rgba(53,40,90,0.12)",
        color: "rgba(26,0,46,0.65)",
      }}
    >
      {text}
    </span>
  );
}

function PillCard({
  pill,
  isTop,
}: {
  pill: PillRecommendation;
  isTop: boolean;
}) {
  const adherence12m = pill.symptom_probs.still_taking.at(-1) ?? 0;
  const sat = pill.satisfaction[pill.satisfaction.length - 1] ?? 0;
  const sideEffects = PILL_SIDE_EFFECTS[pill.pill_id] ?? [];

  return (
    <div
      className="rounded-2xl p-6"
      style={{
        background: isTop ? "rgba(119,103,164,0.07)" : "rgba(255,255,255,0.7)",
        border: isTop
          ? "1px solid rgba(119,103,164,0.35)"
          : "1px solid rgba(53,40,90,0.1)",
        boxShadow: isTop ? "0 4px 24px rgba(119,103,164,0.12)" : "none",
      }}
    >
      {/* Header row */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="section-label">Rank {pill.rank}</span>
            {isTop && (
              <span
                className="text-xs font-body px-2 py-0.5 rounded-full"
                style={{
                  background: "rgba(119,103,164,0.12)",
                  color: "#7767A4",
                  border: "1px solid rgba(119,103,164,0.3)",
                }}
              >
                Best match
              </span>
            )}
          </div>
          <h3
            className="font-display font-semibold"
            style={{
              fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "1.5rem",
              lineHeight: 1.1,
              color: "#1A002E",
            }}
          >
            {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
          </h3>
          <p
            className="font-body text-xs mt-0.5"
            style={{ color: "rgba(26,0,46,0.45)" }}
          >
            {PILL_GENERIC[pill.pill_id] ?? PILL_LABELS[pill.pill_id] ?? pill.pill_id}
          </p>
        </div>

        {/* Suitability score */}
        <div className="text-right ml-4 flex-shrink-0">
          <p
            className="font-display font-semibold leading-none"
            style={{
              fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "2.4rem",
              color: isTop ? "#7767A4" : "#35285A",
            }}
          >
            {(pill.utility_score * 100).toFixed(0)}
          </p>
          <p
            className="font-body text-xs"
            style={{ color: "rgba(26,0,46,0.35)" }}
          >
            / 100 suitability
          </p>
        </div>
      </div>

      {/* Key stats */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        {[
          {
            label: "12-mo. adherence",
            value: `${(adherence12m * 100).toFixed(0)}%`,
            color: "#059669",
          },
          {
            label: "Severe risk",
            value:
              pill.severe_risk < 0.005
                ? "<0.5%"
                : `${(pill.severe_risk * 100).toFixed(1)}%`,
            color: pill.severe_risk < 0.01 ? "#059669" : "#DC2626",
          },
          {
            label: "Satisfaction",
            value: `${sat.toFixed(1)} / 10`,
            color: "#7767A4",
          },
        ].map((s) => (
          <div
            key={s.label}
            className="rounded-xl p-3 text-center"
            style={{
              background: "rgba(255,255,255,0.6)",
              border: "1px solid rgba(53,40,90,0.08)",
            }}
          >
            <p
              className="font-body text-xs mb-1"
              style={{ color: "rgba(26,0,46,0.4)" }}
            >
              {s.label}
            </p>
            <p
              className="font-body font-semibold text-base"
              style={{ color: s.color }}
            >
              {s.value}
            </p>
          </div>
        ))}
      </div>

      {/* Side effects */}
      {sideEffects.length > 0 && (
        <div className="mb-4">
          <p className="section-label mb-2">Potential side effects</p>
          <div className="flex flex-wrap gap-2">
            {sideEffects.map((se) => (
              <SideEffectTag key={se} text={se} />
            ))}
          </div>
        </div>
      )}

      {/* Reason codes */}
      {pill.reason_codes.length > 0 && (
        <div>
          <p className="section-label mb-2">Why Selene selected this</p>
          <ul className="space-y-1.5">
            {pill.reason_codes.map((r) => (
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
                  className="font-body text-xs"
                  style={{ color: "rgba(26,0,46,0.65)" }}
                >
                  {r}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function RecommendationCard({ response }: Props) {
  const pills = response.recommendations;
  const clusterLabel =
    CLUSTER_DESCRIPTIONS[response.cluster_profile] ?? response.cluster_profile;

  return (
    <div className="space-y-4">
      {/* ── Meta info bar ────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap gap-3 px-5 py-4 rounded-2xl"
        style={{
          background: "rgba(255,255,255,0.75)",
          border: "1px solid rgba(53,40,90,0.1)",
        }}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className="section-label flex-shrink-0">Your profile</span>
          <span
            className="font-body text-xs truncate"
            style={{ color: "#1A002E" }}
          >
            {clusterLabel}
          </span>
        </div>
        <div className="flex gap-4 flex-shrink-0">
          {[
            {
              k: "Confidence",
              v: `${(response.cluster_confidence * 100).toFixed(0)}%`,
            },
            { k: "Iterations", v: String(response.iterations) },
            {
              k: "Candidates",
              v: String(response.total_candidates_evaluated),
            },
          ].map((m) => (
            <div key={m.k} className="text-xs font-body text-right">
              <span style={{ color: "rgba(26,0,46,0.4)" }}>{m.k} </span>
              <span style={{ color: "#1A002E", fontWeight: 600 }}>{m.v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Intro text ────────────────────────────────────────────────── */}
      <p
        className="font-body text-sm px-1"
        style={{ color: "rgba(26,0,46,0.55)" }}
      >
        Review all three options below. Each shows its suitability score, key
        statistics, potential side effects, and the reasons behind Selene's
        ranking — so you can discuss the best choice with your healthcare provider.
      </p>

      {/* ── Three pill cards ──────────────────────────────────────────── */}
      <div className="space-y-4">
        {pills.map((pill) => (
          <PillCard
            key={pill.pill_id}
            pill={pill}
            isTop={pill.rank === 1}
          />
        ))}
      </div>

      {/* ── Disclaimer ────────────────────────────────────────────────── */}
      <p
        className="font-body text-xs px-1 pt-2"
        style={{
          color: "rgba(26,0,46,0.3)",
          borderTop: "1px solid rgba(53,40,90,0.08)",
          paddingTop: "1rem",
        }}
      >
        This analysis is generated by a clinical decision-support model. Always
        consult a healthcare provider before starting, stopping, or switching
        contraception.
      </p>
    </div>
  );
}

