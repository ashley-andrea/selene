import type { PatientFormData, PDFExtractionResult, RecommendationResponse } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "https://bc-recommendation-agent.onrender.com";

/**
 * Sanitize the patient payload before sending to the backend.
 *
 * Rules:
 *  - Fields declared as `float | None` on the backend (obs_bmi, obs_*_bp,
 *    obs_phq9_score, obs_testosterone) → keep null, drop if undefined.
 *  - `obs_pain_score` → backend is `float` (no None), so null/undefined → 0.
 *  - All `cond_*` / `med_*` / `obs_smoker` → backend is `int` with default 0,
 *    so undefined → omit (backend uses default), null → 0.
 *  - Remove any key whose value is `undefined` so JSON.stringify doesn't
 *    silently drop it in ways Pydantic finds surprising.
 */
function sanitizePayload(data: PatientFormData): Record<string, unknown> {
  const nullable_float_fields = new Set([
    "obs_bmi",
    "obs_systolic_bp",
    "obs_diastolic_bp",
    "obs_phq9_score",
    "obs_testosterone",
  ]);

  const out: Record<string, unknown> = {};

  for (const [k, v] of Object.entries(data)) {
    if (v === undefined) continue; // omit — backend has defaults for everything

    if (k === "obs_pain_score") {
      // Backend: float (not Optional) — null becomes 0
      out[k] = v === null ? 0 : v;
    } else if (nullable_float_fields.has(k)) {
      // Backend: float | None — null is valid, pass through
      out[k] = v;
    } else {
      out[k] = v;
    }
  }

  return out;
}

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
    body: JSON.stringify(sanitizePayload(data)),
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
