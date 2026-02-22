"use client";

import type { PillRecommendation } from "@/lib/types";
import { PILL_SHORT, PILL_GENERIC } from "@/lib/types";

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
          style={{ color: "rgba(26,0,46,0.5)" }}
        >
          {label}
        </span>
        <span className="font-body text-xs font-medium" style={{ color }}>
          {pct < 0.5 ? "<0.5%" : `${pct.toFixed(1)}%`}
        </span>
      </div>
      <div
        className="h-1.5 rounded-full overflow-hidden"
        style={{ background: "rgba(53,40,90,0.08)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{
            width: `${Math.max(pct, 0.5)}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

const RANK_BORDER_COLORS = ["#7767A4", "rgba(53,40,90,0.18)", "rgba(53,40,90,0.12)"];

interface Props {
  recommendations: PillRecommendation[];
}

export default function RiskCards({ recommendations }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {recommendations.map((pill, i) => {
        const isTop  = pill.rank === 1;
        const border = RANK_BORDER_COLORS[i] ?? "rgba(53,40,90,0.12)";

        return (
          <div
            key={pill.pill_id}
            className="rounded-2xl p-6 space-y-5 relative overflow-hidden"
            style={{
              background: isTop
                ? "rgba(119,103,164,0.06)"
                : "rgba(255,255,255,0.75)",
              border: `1px solid ${border}`,
              boxShadow: isTop ? "0 2px 16px rgba(119,103,164,0.12)" : "none",
            }}
          >
            {/* Rank badge */}
            <div className="flex items-start justify-between">
              <div>
                <span
                  className="section-label block mb-1"
                  style={{ color: "rgba(26,0,46,0.4)" }}
                >
                  Rank {pill.rank}
                </span>
                <h3
                  className="font-body font-semibold text-base leading-tight"
                  style={{ color: "#1A002E" }}
                >
                  {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
                </h3>
                <p
                  className="font-body text-xs mt-0.5"
                  style={{ color: "rgba(26,0,46,0.45)" }}
                >
                  {PILL_GENERIC[pill.pill_id] ?? ""}
                </p>
              </div>
              {isTop && (
                <span
                  className="text-xs font-body px-2 py-0.5 rounded-full inline-flex items-center justify-center"
                  style={{
                    background: "rgba(119,103,164,0.1)",
                    color: "#7767A4",
                    border: "1px solid rgba(119,103,164,0.3)",
                  }}
                >
                  Best match
                </span>
              )}
            </div>

            {/* Suitability score */}
            <div className="flex items-end gap-1">
              <span
                className="font-display font-semibold"
                style={{
                  fontFamily: "'Cormorant Garamond', Georgia, serif",
                  fontSize: "2.2rem",
                  lineHeight: 1,
                  color: isTop ? "#7767A4" : "#35285A",
                }}
              >
                {(pill.utility_score * 100).toFixed(0)}
              </span>
              <span
                className="font-body text-xs mb-1"
                style={{ color: "rgba(26,0,46,0.4)" }}
              >
                / 100 suitability
              </span>
            </div>

            {/* Risk bars */}
            <div className="space-y-3">
              <RiskBar
                label="Severe adverse event"
                value={pill.severe_risk}
                color="#DC2626"
              />
              <RiskBar
                label="Discontinuation risk"
                value={pill.predicted_discontinuation}
                color="#B45309"
              />
              <RiskBar
                label="Mild side-effect score"
                value={pill.mild_side_effect_score}
                color="#7767A4"
              />
              <RiskBar
                label="Contraceptive efficacy"
                value={pill.contraceptive_effectiveness}
                color="#059669"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
