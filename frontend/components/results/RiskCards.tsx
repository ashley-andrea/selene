"use client";

import type { PillRecommendation } from "@/lib/types";
import { PILL_SHORT } from "@/lib/types";

function RiskBar({
  value,
  color,
  label,
}: {
  value: number;
  color: string;
  label: string;
}) {
  const pct = Math.min(value * 100, 100);
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span
          className="font-body text-xs"
          style={{ color: "rgba(221,216,196,0.55)" }}
        >
          {label}
        </span>
        <span className="font-body text-xs font-medium" style={{ color }}>
          {pct < 0.5 ? "<0.5%" : `${pct.toFixed(1)}%`}
        </span>
      </div>
      <div
        className="h-1.5 rounded-full overflow-hidden"
        style={{ background: "rgba(53,40,90,0.8)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{
            width: `${Math.max(pct, 0.5)}%`,
            background: color,
            boxShadow: `0 0 6px ${color}88`,
          }}
        />
      </div>
    </div>
  );
}

const RANK_COLORS = ["#DDD8C4", "#7767A4", "#9B8BC4"];

interface Props {
  recommendations: PillRecommendation[];
}

export default function RiskCards({ recommendations }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {recommendations.map((pill, i) => {
        const color  = RANK_COLORS[i] ?? "#9B8BC4";
        const isTop  = pill.rank === 1;

        return (
          <div
            key={pill.pill_id}
            className="rounded-2xl p-6 space-y-5 relative overflow-hidden"
            style={{
              background: isTop
                ? "rgba(119,103,164,0.12)"
                : "rgba(53,40,90,0.35)",
              border: isTop
                ? "1px solid rgba(221,216,196,0.3)"
                : "1px solid rgba(119,103,164,0.18)",
            }}
          >
            {/* Rank badge */}
            <div className="flex items-start justify-between">
              <div>
                <span
                  className="section-label block mb-1"
                  style={{ color: "rgba(221,216,196,0.4)" }}
                >
                  Rank {pill.rank}
                </span>
                <h3
                  className="font-body font-semibold text-base leading-tight"
                  style={{ color }}
                >
                  {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
                </h3>
              </div>
              {isTop && (
                <span
                  className="text-xs font-body px-2 py-0.5 rounded-full"
                  style={{
                    background: "rgba(110,231,183,0.15)",
                    color: "#6EE7B7",
                    border: "1px solid rgba(110,231,183,0.25)",
                  }}
                >
                  Best match
                </span>
              )}
            </div>

            {/* Utility score */}
            <div className="flex items-end gap-1">
              <span
                className="font-display"
                style={{
                  fontFamily: "Catchye, 'Cormorant Garamond', serif",
                  fontSize: "2.2rem",
                  lineHeight: 1,
                  color,
                }}
              >
                {(pill.utility_score * 100).toFixed(0)}
              </span>
              <span
                className="font-body text-xs mb-1"
                style={{ color: "rgba(221,216,196,0.45)" }}
              >
                / 100 utility
              </span>
            </div>

            {/* Risk bars */}
            <div className="space-y-3">
              <RiskBar
                label="Severe adverse event"
                value={pill.severe_risk}
                color="#F87171"
              />
              <RiskBar
                label="Discontinuation risk"
                value={pill.predicted_discontinuation}
                color="#FCD34D"
              />
              <RiskBar
                label="Mild side-effect score"
                value={pill.mild_side_effect_score}
                color="#C084FC"
              />
              <RiskBar
                label="Contraceptive efficacy"
                value={pill.contraceptive_effectiveness}
                color="#6EE7B7"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
