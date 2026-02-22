import type { PatientFormData, PDFExtractionResult, RecommendationResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "https://bc-recommendation-agent.onrender.com";

/* ── PDF extraction ─────────────────────────────────────────────────────── */
export async function uploadPDF(file: File): Promise<PDFExtractionResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/api/v1/patient/upload-pdf`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `Upload failed (${res.status})`);
  }
  return res.json() as Promise<PDFExtractionResult>;
}

/* ── Recommendation pipeline ────────────────────────────────────────────── */
export async function getRecommendation(data: PatientFormData): Promise<RecommendationResponse> {
  const res = await fetch(`${BASE}/api/v1/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? `Recommendation failed (${res.status})`);
  }
  return res.json() as Promise<RecommendationResponse>;
}

/* ── Health check ───────────────────────────────────────────────────────── */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { method: "GET" });
    return res.ok;
  } catch {
    return false;
  }
}
