"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/layout/Navbar";
import Stars from "@/components/home/Stars";
import WakeupPing from "@/components/home/WakeupPing";
import UploadStep from "@/components/intake/UploadStep";
import ManualForm from "@/components/intake/ManualForm";
import type { PatientFormData, PDFExtractionResult } from "@/lib/types";

type Tab = "pdf" | "manual";

export default function IntakePage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("manual");
  const [prefill, setPrefill] = useState<Partial<PatientFormData> | undefined>();
  const [pdfMeta, setPdfMeta] = useState<{ pages: number; backend: string } | null>(null);

  const handlePDFExtracted = (data: PDFExtractionResult) => {
    const { pages_parsed, parser_backend, ...rest } = data;
    setPdfMeta({ pages: pages_parsed, backend: parser_backend });
    setPrefill(rest);
    setTab("manual"); // switch to manual tab pre-filled
  };

  const handleSubmit = (data: PatientFormData) => {
    sessionStorage.setItem("selene_patient", JSON.stringify(data));
    router.push("/results");
  };

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 100% 70% at 50% -10%, rgba(53,40,90,0.07) 0%, #F5F3EC 60%)",
          zIndex: 0,
        }}
      />
      <Stars />
      <WakeupPing />
      <Navbar />

      <div
        className="relative max-w-2xl mx-auto px-6 pt-28 pb-20"
        style={{ zIndex: 1 }}
      >
        {/* Header */}
        <div className="text-center mb-10">
          <p className="section-label mb-2">Assessment</p>
          <h1
            className="font-display mb-2"
            style={{
              fontFamily: "'Cormorant Garamond', Georgia, serif",
              fontSize: "clamp(2rem, 6vw, 3rem)",
              color: "#1A002E",
            }}
          >
            Your profile
          </h1>
          <p className="font-body text-sm" style={{ color: "rgba(26,0,46,0.5)" }}>
            Only your age is required. Provide as much detail as you have.
          </p>
        </div>

        {/* Tab switcher */}
        <div
          className="flex rounded-xl p-1 mb-8"
          style={{ background: "rgba(53,40,90,0.08)" }}
        >
          {(["pdf", "manual"] as Tab[]).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className="flex-1 py-2.5 rounded-lg text-sm font-body font-medium transition-all"
              style={{
                background: tab === t ? "#7767A4" : "transparent",
                color: tab === t ? "#F5F3EC" : "rgba(26,0,46,0.45)",
              }}
            >
              {t === "pdf" ? "Upload medical record" : "Enter manually"}
            </button>
          ))}
        </div>

        {/* PDF extraction banner */}
        {pdfMeta && tab === "manual" && (
          <div
            className="flex items-start gap-3 rounded-xl px-5 py-4 mb-6"
            style={{
              background: "rgba(119,103,164,0.12)",
              border: "1px solid rgba(119,103,164,0.3)",
            }}
          >
            <span style={{ fontSize: "1.2rem" }}>✓</span>
            <div>
              <p className="font-body text-sm font-semibold" style={{ color: "#DDD8C4" }}>
                PDF extracted — {pdfMeta.pages} pages parsed
              </p>
              <p className="font-body text-xs" style={{ color: "rgba(221,216,196,0.5)" }}>
                Fields below are pre-filled from your document. Review and
                correct anything before submitting.
              </p>
            </div>
          </div>
        )}

        {/* Tab content */}
        {tab === "pdf" ? (
          <UploadStep onExtracted={handlePDFExtracted} />
        ) : (
          <ManualForm initial={prefill} onSubmit={handleSubmit} />
        )}
      </div>
    </main>
  );
}
