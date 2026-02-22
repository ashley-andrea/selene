"use client";

import { useEffect, useState } from "react";

const STEPS = [
  "Clustering patient profile",
  "Running WHO safety screening",
  "Simulating 12-month trajectories",
  "Optimising recommendation",
];

export default function LoadingMoon() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const t = setInterval(
      () => setStep((s) => Math.min(s + 1, STEPS.length - 1)),
      3800
    );
    return () => clearInterval(t);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-6">
      {/* ── Orbital rig ────────────────────────────────────────────────── */}
      <div className="relative mb-14" style={{ width: 160, height: 160 }}>
        {/* Outer ring */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            border: "1px solid rgba(119,103,164,0.18)",
          }}
        />
        {/* Mid ring */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 20,
            border: "1px solid rgba(119,103,164,0.28)",
          }}
        />
        {/* Inner ring */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 44,
            border: "1.5px solid rgba(119,103,164,0.45)",
          }}
        />

        {/* Glow core */}
        <div
          className="absolute"
          style={{
            top: "50%",
            left: "50%",
            width: 36,
            height: 36,
            marginLeft: -18,
            marginTop: -18,
            borderRadius: "50%",
            background: "radial-gradient(circle, #7767A4 0%, transparent 70%)",
            opacity: 0.45,
            animation: "pulse-glow 2.5s ease-in-out infinite",
          }}
        />

        {/* Centre moon */}
        <div
          className="absolute"
          style={{
            top: "50%",
            left: "50%",
            width: 20,
            height: 20,
            marginLeft: -10,
            marginTop: -10,
          }}
        >
          <svg viewBox="0 0 20 20" fill="none" width={20} height={20}>
            <path
              d="M16 10.5A6 6 0 1 1 9.5 4 4.5 4.5 0 0 0 16 10.5z"
              fill="#7767A4"
              opacity="0.85"
            />
          </svg>
        </div>

        {/* Orbiting dot — outer orbit */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            width: 8,
            height: 8,
            marginLeft: -4,
            marginTop: -4,
            animation: "orbit-outer 2.4s linear infinite",
          }}
        >
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#7767A4",
              boxShadow: "0 0 6px #7767A4, 0 0 14px rgba(119,103,164,0.4)",
            }}
          />
        </div>

        {/* Second orbiting dot — mid orbit, opposite phase */}
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            width: 5,
            height: 5,
            marginLeft: -2.5,
            marginTop: -2.5,
            animation: "orbit-mid 3.6s linear infinite",
          }}
        >
          <div
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "#7767A4",
              boxShadow: "0 0 8px #7767A4",
            }}
          />
        </div>
      </div>

      {/* ── Title ───────────────────────────────────────────────────────── */}
      <h2
        className="font-display mb-1 glow-text"
        style={{
          fontFamily: "'Cormorant Garamond', Georgia, serif",
          fontSize: "1.8rem",
          color: "#1A002E",
          textAlign: "center",
        }}
      >
        Selene is thinking
      </h2>
      <p
        className="font-body text-sm mb-10"
        style={{ color: "rgba(26,0,46,0.5)", textAlign: "center" }}
      >
        This usually takes 10–30 seconds
      </p>

      {/* ── Step list ───────────────────────────────────────────────────── */}
      <div className="space-y-4 w-full max-w-xs">
        {STEPS.map((s, i) => {
          const done    = i < step;
          const current = i === step;
          return (
            <div
              key={s}
              className="flex items-center gap-4 transition-all duration-700"
              style={{ opacity: done ? 0.35 : current ? 1 : 0.18 }}
            >
              {/* indicator */}
              <div
                style={{
                  width: 22,
                  height: 22,
                  borderRadius: "50%",
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: done
                    ? "rgba(119,103,164,0.3)"
                    : current
                    ? "#7767A4"
                    : "rgba(53,40,90,0.6)",
                  border: current ? "none" : "1px solid rgba(119,103,164,0.3)",
                  transition: "all 0.5s ease",
                }}
              >
                {done ? (
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path
                      d="M2 6l3 3 5-5"
                      stroke="white"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : current ? (
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                    background: "white",
                      animation: "step-pulse 1.2s ease-in-out infinite",
                    }}
                  />
                ) : (
                  <div
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: "50%",
                      background: "rgba(26,0,46,0.15)",
                    }}
                  />
                )}
              </div>

              <span
                className="font-body text-sm"
                style={{ color: "#1A002E" }}
              >
                {s}
              </span>
            </div>
          );
        })}
      </div>

      {/* Keyframes */}
      <style>{`
        @keyframes orbit-outer {
          from { transform: rotate(0deg) translateX(72px) rotate(0deg); }
          to   { transform: rotate(360deg) translateX(72px) rotate(-360deg); }
        }
        @keyframes orbit-mid {
          from { transform: rotate(180deg) translateX(50px) rotate(-180deg); }
          to   { transform: rotate(540deg) translateX(50px) rotate(-540deg); }
        }
        @keyframes pulse-glow {
          0%, 100% { transform: scale(1);   opacity: 0.35; }
          50%       { transform: scale(1.4); opacity: 0.6; }
        }
        @keyframes step-pulse {
          0%, 100% { transform: scale(1);   opacity: 0.7; }
          50%       { transform: scale(1.3); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
