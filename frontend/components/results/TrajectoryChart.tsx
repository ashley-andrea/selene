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
  Legend,
} from "recharts";
import type { PillRecommendation, FeatureGroup } from "@/lib/types";
import { FEATURE_GROUPS, PILL_SHORT } from "@/lib/types";

/* ── Colour per pill rank ───────────────────────────────────────────────── */
const PILL_COLORS  = ["#DDD8C4", "#7767A4", "#A78BFA"];
const PILL_DASHES  = ["0", "6 3", "2 2"];
const PILL_OPAQUE  = [1, 0.85, 0.7];

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
        background: "rgba(26,0,46,0.95)",
        border: "1px solid rgba(119,103,164,0.35)",
        backdropFilter: "blur(12px)",
      }}
    >
      <p
        className="font-semibold mb-2"
        style={{ color: "rgba(221,216,196,0.6)" }}
      >
        Month {label}
      </p>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span
            className="inline-block w-3 h-0.5"
            style={{ background: p.color }}
          />
          <span style={{ color: "rgba(221,216,196,0.7)" }}>{p.name}</span>
          <span className="font-semibold ml-auto pl-4" style={{ color: p.color }}>
            {(p.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
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
  const ANIM_DURATION = 3600; // ms for full 12-month reveal

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
        if (next.size === 1) return prev; // keep at least one
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

  /* Build chart data — full 12 months; mask handles reveal */
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

  /* Satisfaction data */
  const satisfactionData = useMemo(
    () =>
      months.map((m, mi) =>
        Object.fromEntries([
          ["month", m],
          ...recommendations.map((p, pi) => [`sat__${pi}`, p.satisfaction[mi] ?? 0]),
        ])
      ),
    [months, recommendations]
  );

  const displayMonth = Math.round(revealPct * 12);
  const isRisk = activeGroupId === "risk";

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
                background: active ? g.color + "25" : "rgba(53,40,90,0.6)",
                color: active ? g.color : "rgba(221,216,196,0.45)",
                border: `1px solid ${active ? g.color + "55" : "rgba(119,103,164,0.2)"}`,
                boxShadow: active ? `0 0 12px ${g.color}33` : "none",
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
                  background: on ? "rgba(221,216,196,0.12)" : "transparent",
                  color: on ? "#DDD8C4" : "rgba(221,216,196,0.3)",
                  border: "1px solid rgba(221,216,196,0.18)",
                }}
              >
                {f.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Serious events scale note */}
      {isRisk && (
        <p className="text-xs font-body" style={{ color: "rgba(221,216,196,0.4)" }}>
          Note: serious event probabilities are very small (&lt;1%). The
          y-axis is automatically scaled to make differences visible.
        </p>
      )}

      {/* ── Main chart ─────────────────────────────────────────────────── */}
      <div className="relative rounded-2xl overflow-hidden" style={{ height: 340 }}>
        {/* Mask for progressive reveal */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `linear-gradient(to right, transparent ${revealPct * 100}%, rgba(26,0,46,0.96) ${Math.min(revealPct * 100 + 3, 100)}%)`,
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
              stroke="rgba(53,40,90,0.9)"
              vertical={false}
            />
            <XAxis
              dataKey="month"
              tick={{ fill: "rgba(221,216,196,0.4)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "rgba(119,103,164,0.2)" }}
              label={{
                value: "Month",
                position: "insideBottom",
                offset: -2,
                fill: "rgba(221,216,196,0.3)",
                fontSize: 10,
              }}
            />
            <YAxis
              tickFormatter={(v: number) =>
                isRisk ? `${(v * 100).toFixed(2)}%` : `${(v * 100).toFixed(0)}%`
              }
              tick={{ fill: "rgba(221,216,196,0.4)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: 12 }}
              formatter={(value: string) => (
                <span className="font-body text-xs" style={{ color: "rgba(221,216,196,0.6)" }}>
                  {value}
                </span>
              )}
            />

            {recommendations.flatMap((pill, pi) =>
              group.features
                .filter((f) => activeFeatures.has(f.key))
                .map((f) => (
                  <Line
                    key={`${f.key}__${pi}`}
                    type="monotone"
                    dataKey={`${f.key}__${pi}`}
                    name={`${f.label} — ${PILL_SHORT[pill.pill_id] ?? pill.pill_id}`}
                    stroke={group.color}
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

      {/* ── Pill legend ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-4 justify-center">
        {recommendations.map((pill, i) => (
          <div key={pill.pill_id} className="flex items-center gap-2">
            <svg width="24" height="6">
              <line
                x1="0"
                y1="3"
                x2="24"
                y2="3"
                stroke={PILL_COLORS[i]}
                strokeWidth={i === 0 ? 2 : 1.5}
                strokeDasharray={PILL_DASHES[i]}
                strokeOpacity={PILL_OPAQUE[i]}
              />
            </svg>
            <span
              className="font-body text-xs"
              style={{ color: PILL_COLORS[i] }}
            >
              {PILL_SHORT[pill.pill_id] ?? pill.pill_id}
            </span>
          </div>
        ))}
      </div>

      {/* ── Play controls ───────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={() => (isPlaying ? setIsPlaying(false) : play())}
          className="flex items-center gap-2 rounded-full px-5 py-2 text-sm font-body font-medium transition-all"
          style={{
            background: isPlaying ? "rgba(119,103,164,0.2)" : "#7767A4",
            color: "#DDD8C4",
            border: "1px solid rgba(119,103,164,0.4)",
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
            className="flex-1 accent-[#7767A4] h-1"
            style={{ accentColor: "#7767A4" }}
          />
          <span
            className="text-xs font-body w-16 text-right"
            style={{ color: "rgba(221,216,196,0.5)" }}
          >
            {displayMonth < 1 ? "Month —" : `Month ${displayMonth}`}
          </span>
        </div>
      </div>

      {/* ── Satisfaction chart ──────────────────────────────────────────── */}
      <div>
        <p className="section-label mb-4">Predicted satisfaction score (1–10)</p>
        <div className="relative rounded-2xl overflow-hidden" style={{ height: 200 }}>
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: `linear-gradient(to right, transparent ${revealPct * 100}%, rgba(26,0,46,0.96) ${Math.min(revealPct * 100 + 3, 100)}%)`,
              zIndex: 2,
            }}
          />
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={satisfactionData}
              margin={{ top: 8, right: 20, bottom: 0, left: -10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(53,40,90,0.9)"
                vertical={false}
              />
              <XAxis
                dataKey="month"
                tick={{ fill: "rgba(221,216,196,0.4)", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "rgba(119,103,164,0.2)" }}
              />
              <YAxis
                domain={[1, 10]}
                tickFormatter={(v: number) => `${v}`}
                tick={{ fill: "rgba(221,216,196,0.4)", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                formatter={(
                  v: number | string | undefined,
                  name: string | undefined
                ) => [
                  typeof v === "number" ? v.toFixed(1) : String(v ?? ""),
                  (name ?? "").replace("sat__", "Pill "),
                ]}
                contentStyle={{
                  background: "rgba(26,0,46,0.95)",
                  border: "1px solid rgba(119,103,164,0.35)",
                  borderRadius: 12,
                  fontFamily: "Epilogue, sans-serif",
                  fontSize: 12,
                  color: "#DDD8C4",
                }}
              />
              {recommendations.map((pill, pi) => (
                <Line
                  key={`sat__${pi}`}
                  type="monotone"
                  dataKey={`sat__${pi}`}
                  name={PILL_SHORT[pill.pill_id] ?? pill.pill_id}
                  stroke={PILL_COLORS[pi]}
                  strokeWidth={pi === 0 ? 2 : 1.5}
                  strokeDasharray={PILL_DASHES[pi]}
                  strokeOpacity={PILL_OPAQUE[pi]}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
