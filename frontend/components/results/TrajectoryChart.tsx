"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PillRecommendation, FeatureGroup } from "@/lib/types";
import { FEATURE_GROUPS, PILL_SHORT } from "@/lib/types";

/* ── Pill line styles (dash only — colour comes from feature, not pill) ── */
const PILL_DASHES  = ["0", "6 3", "2 2"];
const PILL_OPAQUE  = [1, 0.8, 0.65];
const PILL_LEGEND_COLORS = ["#1A002E", "#7767A4", "#9B8BC4"];

/* ── Custom tooltip ─────────────────────────────────────────────────────── */
function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-xl px-4 py-3 space-y-1.5 text-xs font-body"
      style={{
        background: "rgba(255,255,255,0.98)",
        border: "1px solid rgba(53,40,90,0.15)",
        backdropFilter: "blur(12px)",
        boxShadow: "0 4px 24px rgba(26,0,46,0.1)",
      }}
    >
      <p
        className="font-semibold mb-2"
        style={{ color: "rgba(26,0,46,0.55)" }}
      >
        Month {label}
      </p>
      {payload.map((p) => {
        const pct = p.value * 100;
        const formatted = pct < 0.1 ? `${pct.toFixed(3)}%` : `${pct.toFixed(1)}%`;
        return (
          <div key={p.name} className="flex items-center gap-2">
            <span
              className="inline-block w-3 h-0.5"
              style={{ background: p.color }}
            />
            <span style={{ color: "rgba(26,0,46,0.65)" }}>{p.name}</span>
            <span className="font-semibold ml-auto pl-4" style={{ color: p.color }}>
              {formatted}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────────────────── */
interface Props {
  recommendations: PillRecommendation[];
}

export default function TrajectoryChart({ recommendations }: Props) {
  const [activeGroupId, setActiveGroupId] = useState<string>("adherence");
  const [activeFeatures, setActiveFeatures] = useState<Set<string>>(
    new Set(["still_taking"])
  );
  const [revealPct, setRevealPct] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const rafRef = useRef<number | null>(null);
  const startRef = useRef<number>(0);
  const ANIM_DURATION = 3600;

  /* Switch group — reset feature selection to all-in-group */
  const switchGroup = useCallback((g: FeatureGroup) => {
    setActiveGroupId(g.id);
    setActiveFeatures(new Set(g.features.map((f) => f.key)));
    setRevealPct(0);
    setIsPlaying(false);
  }, []);

  const toggleFeature = (key: string) => {
    setActiveFeatures((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        if (next.size === 1) return prev;
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  /* Animation loop */
  const play = useCallback(() => {
    if (revealPct >= 1) setRevealPct(0);
    startRef.current = performance.now() - revealPct * ANIM_DURATION;
    setIsPlaying(true);
  }, [revealPct]);

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      return;
    }
    const tick = (now: number) => {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / ANIM_DURATION, 1);
      setRevealPct(progress);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        setIsPlaying(false);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying]);

  const group = FEATURE_GROUPS.find((g) => g.id === activeGroupId)!;
  const months = recommendations[0]?.months ?? [1,2,3,4,5,6,7,8,9,10,11,12];

  const chartData = useMemo(() => {
    return months.map((m, mi) =>
      Object.fromEntries([
        ["month", m],
        ...recommendations.flatMap((pill, pi) =>
          group.features
            .filter((f) => activeFeatures.has(f.key))
            .map((f) => [
              `${f.key}__${pi}`,
              pill.symptom_probs[f.key]?.[mi] ?? 0,
            ])
        ),
      ])
    );
  }, [months, recommendations, group, activeFeatures]);

  const displayMonth = Math.round(revealPct * 12);

  return (
    <div className="space-y-6">
      {/* ── Feature group selector ──────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2">
        {FEATURE_GROUPS.map((g) => {
          const active = g.id === activeGroupId;
          return (
            <button
              key={g.id}
              type="button"
              onClick={() => switchGroup(g)}
              className="text-xs font-body font-medium px-3 py-1.5 rounded-full transition-all duration-200"
              style={{
                background: active ? g.color + "18" : "rgba(53,40,90,0.06)",
                color: active ? g.color : "rgba(26,0,46,0.45)",
                border: `1px solid ${active ? g.color + "55" : "rgba(53,40,90,0.15)"}`,
              }}
            >
              {g.label}
            </button>
          );
        })}
      </div>

      {/* ── Feature chips (within group) ──────────────────────────────── */}
      {group.features.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {group.features.map((f) => {
            const on = activeFeatures.has(f.key);
            return (
              <button
                key={f.key}
                type="button"
                onClick={() => toggleFeature(f.key)}
                className="text-xs font-body px-2.5 py-1 rounded-full transition-all"
                style={{
                  background: on ? f.color + "18" : "transparent",
                  color: on ? f.color : "rgba(26,0,46,0.35)",
                  border: `1px solid ${on ? f.color + "55" : "rgba(26,0,46,0.15)"}`,
                }}
              >
                {f.label}
              </button>
            );
          })}
        </div>
      )}

      {/* ── Main chart ─────────────────────────────────────────────────── */}
      <div className="relative rounded-2xl overflow-hidden" style={{ height: 340 }}>
        {/* Progressive reveal mask (matches light page background) */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `linear-gradient(to right, transparent ${revealPct * 100}%, #F5F3EC ${Math.min(revealPct * 100 + 3, 100)}%)`,
            zIndex: 2,
          }}
        />

        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 8, right: 20, bottom: 0, left: -10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(53,40,90,0.1)"
              vertical={false}
            />
            <XAxis
              dataKey="month"
              tick={{ fill: "rgba(26,0,46,0.4)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "rgba(53,40,90,0.15)" }}
              label={{
                value: "Month",
                position: "insideBottom",
                offset: -2,
                fill: "rgba(26,0,46,0.3)",
                fontSize: 10,
              }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fill: "rgba(26,0,46,0.4)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />

            {recommendations.flatMap((pill, pi) =>
              group.features
                .filter((f) => activeFeatures.has(f.key))
                .map((f) => (
                  <Line
                    key={`${f.key}__${pi}`}
                    type="monotone"
                    dataKey={`${f.key}__${pi}`}
                    name={`${f.label} — ${PILL_SHORT[pill.pill_id] ?? pill.pill_id}`}
                    stroke={f.color}
                    strokeWidth={pi === 0 ? 2 : 1.5}
                    strokeDasharray={PILL_DASHES[pi]}
                    strokeOpacity={PILL_OPAQUE[pi]}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Pill line-style legend ───────────────────────────────────────── */}
      <div className="flex flex-wrap gap-5 justify-center">
        {recommendations.map((pill, i) => (
          <div key={pill.pill_id} className="flex items-center gap-2">
            <svg width="28" height="8">
              <line
                x1="0"
                y1="4"
                x2="28"
                y2="4"
                stroke={PILL_LEGEND_COLORS[i]}
                strokeWidth={i === 0 ? 2 : 1.5}
                strokeDasharray={PILL_DASHES[i]}
                strokeOpacity={1}
              />
            </svg>
            <span
              className="font-body text-xs"
              style={{ color: PILL_LEGEND_COLORS[i] }}
            >
              {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
            </span>
          </div>
        ))}
        <span
          className="font-body text-xs self-center"
          style={{ color: "rgba(26,0,46,0.35)" }}
        >
          · line style = pill, colour = symptom
        </span>
      </div>

      {/* ── Play controls ───────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={() => (isPlaying ? setIsPlaying(false) : play())}
          className="flex items-center gap-2 rounded-full px-5 py-2 text-sm font-body font-medium transition-all"
          style={{
            background: isPlaying ? "rgba(53,40,90,0.08)" : "#7767A4",
            color: isPlaying ? "#1A002E" : "#F5F3EC",
            border: "1px solid rgba(53,40,90,0.2)",
          }}
        >
          {isPlaying ? (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12">
                <rect x="1" y="1" width="4" height="10" rx="1" fill="currentColor" />
                <rect x="7" y="1" width="4" height="10" rx="1" fill="currentColor" />
              </svg>
              Pause
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 12 12">
                <path d="M2 1l9 5-9 5V1z" fill="currentColor" />
              </svg>
              {revealPct >= 1 ? "Replay" : "Play"}
            </>
          )}
        </button>

        {/* Scrubber */}
        <div className="flex-1 flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={revealPct}
            onChange={(e) => {
              setIsPlaying(false);
              setRevealPct(parseFloat(e.target.value));
            }}
            className="flex-1 h-1"
            style={{ accentColor: "#7767A4" }}
          />
          <span
            className="text-xs font-body w-16 text-right"
            style={{ color: "rgba(26,0,46,0.45)" }}
          >
            {displayMonth < 1 ? "Month —" : `Month ${displayMonth}`}
          </span>
        </div>
      </div>

      {/* ── Satisfaction scores (static per pill) ───────────────────────── */}
      <div>
        <p className="section-label mb-3">Predicted satisfaction score (1–10)</p>
        <div className="grid grid-cols-3 gap-3">
          {recommendations.map((pill, i) => {
            const sat = pill.satisfaction[pill.satisfaction.length - 1] ?? 0;
            const color = PILL_LEGEND_COLORS[i];
            return (
              <div
                key={pill.pill_id}
                className="rounded-xl p-4 text-center"
                style={{
                  background: "rgba(255,255,255,0.7)",
                  border: `1px solid ${color}33`,
                }}
              >
                <p
                  className="font-body text-xs mb-1"
                  style={{ color: "rgba(26,0,46,0.45)" }}
                >
                  {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
                </p>
                <p
                  className="font-display font-semibold"
                  style={{
                    fontFamily: "'Cormorant Garamond', Georgia, serif",
                    fontSize: "2rem",
                    lineHeight: 1,
                    color,
                  }}
                >
                  {sat.toFixed(1)}
                </p>
                <p
                  className="font-body text-xs mt-1"
                  style={{ color: "rgba(26,0,46,0.3)" }}
                >
                  / 10
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

