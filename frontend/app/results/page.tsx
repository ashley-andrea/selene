"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Stars from "@/components/home/Stars";
import Navbar from "@/components/layout/Navbar";
import LoadingMoon from "@/components/results/LoadingMoon";
import SafetyGate from "@/components/results/SafetyGate";
import RiskCards from "@/components/results/RiskCards";
import TrajectoryChart from "@/components/results/TrajectoryChart";
import RecommendationCard from "@/components/results/RecommendationCard";
import { getRecommendation } from "@/lib/api";
import type { PatientFormData, RecommendationResponse } from "@/lib/types";

/* Animated section wrapper */
function Section({
  title,
  label,
  children,
  delay = 0,
}: {
  title: string;
  label: string;
  children: React.ReactNode;
  delay?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.08 }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="mb-16 transition-all duration-700"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(28px)",
        transitionDelay: `${delay}ms`,
      }}
    >
      <p className="section-label mb-1">{label}</p>
      <h2
        className="font-display mb-6"
        style={{
          fontFamily: "'Cormorant Garamond', Georgia, serif",
          fontSize: "clamp(1.4rem, 3vw, 2rem)",
          color: "#1A002E",
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

export default function ResultsPage() {
  const router = useRouter();
  const [patient, setPatient] = useState<PatientFormData | null>(null);
  const [results, setResults] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const raw = sessionStorage.getItem("selene_patient");
    if (!raw) {
      router.replace("/intake");
      return;
    }
    const parsed: PatientFormData = JSON.parse(raw);
    setPatient(parsed);

    getRecommendation(parsed)
      .then(setResults)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "An unexpected error occurred.")
      );
  }, [router]);

  /* ── Error state ──────────────────────────────────────────────────────── */
  if (error) {
    return (
      <main
        className="relative min-h-screen flex flex-col items-center justify-center px-6 text-center"
        style={{ background: "#F5F3EC" }}
      >
        <Stars />
        <div
          className="relative z-10 max-w-md card p-10"
        >
          <p
            className="font-body text-4xl mb-4"
          >
            🌑
          </p>
          <h2
            className="font-display mb-3"
            style={{
              fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "1.6rem",
              color: "#1A002E",
            }}
          >
            Something went wrong
          </h2>
          <p className="font-body text-sm mb-6" style={{ color: "rgba(26,0,46,0.55)" }}>
            {error}
          </p>
          <button
            onClick={() => router.push("/intake")}
            className="btn-primary"
          >
            Try again
          </button>
        </div>
      </main>
    );
  }

  /* ── Loading state ────────────────────────────────────────────────────── */
  if (!results) {
    return (
      <main className="relative min-h-screen" style={{ background: "#F5F3EC" }}>
        <Stars />
        <div className="relative z-10">
          <LoadingMoon />
        </div>
      </main>
    );
  }

  const top = results.recommendations[0];

  /* ── Results ──────────────────────────────────────────────────────────── */
  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(53,40,90,0.06) 0%, #F5F3EC 55%)",
          zIndex: 0,
        }}
      />
      <Stars />
      <Navbar />

      <div
        className="relative max-w-4xl mx-auto px-6 pt-28 pb-24"
        style={{ zIndex: 1 }}
      >
        {/* Page header */}
        <div className="text-center mb-16">
          <p className="section-label mb-2">Your results</p>
          <h1
            className="font-display glow-text"
            style={{
              fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "clamp(2rem, 6vw, 3.5rem)",
              color: "#1A002E",
              lineHeight: 1,
            }}
          >
            Analysis complete
          </h1>
        </div>

        {/* ── 1. Safety screening ─────────────────────────────────────── */}
        <Section label="Step 1" title="Safety screening" delay={0}>
          {patient && (
            <SafetyGate patient={patient} response={results} />
          )}
        </Section>

        {/* ── 2. Risk forecast ────────────────────────────────────────── */}
        <Section label="Step 2" title="Immediate risk overview" delay={100}>
          <RiskCards recommendations={results.recommendations} />
        </Section>

        {/* ── 3. Trajectory simulation ────────────────────────────────── */}
        <Section label="Step 3" title="12-month treatment trajectory" delay={200}>
          <div
            className="rounded-2xl p-6 md:p-8"
            style={{
              background: "rgba(255,255,255,0.75)",
              border: "1px solid rgba(53,40,90,0.1)",
            }}
          >
            <p
              className="font-body text-sm mb-6"
              style={{ color: "rgba(26,0,46,0.5)" }}
            >
              Select a feature group, choose individual factors, then press{" "}
              <strong style={{ color: "#1A002E" }}>Play</strong> to animate how
              each metric evolves over 12 months for all three pill options.
            </p>
            <TrajectoryChart recommendations={results.recommendations} />
          </div>
        </Section>

        {/* ── 4. Final recommendation ─────────────────────────────────── */}
        <Section label="Selene's choice" title="Recommended option" delay={300}>
          <RecommendationCard top={top} response={results} />
        </Section>

        {/* CTA — start over */}
        <div className="text-center mt-8">
          <button
            onClick={() => {
              sessionStorage.removeItem("selene_patient");
              router.push("/intake");
            }}
            className="btn-ghost text-sm"
          >
            ← Start a new assessment
          </button>
        </div>
      </div>
    </main>
  );
}
