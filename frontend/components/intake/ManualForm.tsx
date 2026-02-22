"use client";

import { useState } from "react";
import type { PatientFormData } from "@/lib/types";

/* ── Tiny building-block components ──────────────────────────────────────── */

function Toggle({
  value,
  onChange,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      className="relative inline-flex h-5 w-9 flex-shrink-0 items-center rounded-full transition-colors focus:outline-none"
      style={{ background: value ? "#7767A4" : "rgba(119,103,164,0.15)" }}
    >
      <span
        className="inline-block h-3.5 w-3.5 transform rounded-full transition-transform"
        style={{
          background: "white",
          transform: value ? "translateX(19px)" : "translateX(3px)",
        }}
      />
    </button>
  );
}

function NumberInput({
  label,
  placeholder,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
}: {
  label: string;
  placeholder?: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
}) {
  return (
    <div>
      <label
        className="block font-body text-xs font-medium mb-1"
        style={{ color: "rgba(26,0,46,0.65)" }}
      >
        {label}
        {unit && (
          <span style={{ color: "rgba(26,0,46,0.4)" }}> ({unit})</span>
        )}
      </label>
      <input
        type="number"
        className="input-field"
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        value={value ?? ""}
        onChange={(e) =>
          onChange(e.target.value === "" ? null : parseFloat(e.target.value))
        }
      />
    </div>
  );
}

function Section({
  title,
  badge,
  children,
  defaultOpen = false,
  badgeColor = "#7767A4",
}: {
  title: string;
  badge?: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badgeColor?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{ border: "1px solid rgba(119,103,164,0.2)" }}
    >
      <button
        type="button"
        className="w-full flex items-center justify-between px-6 py-4 transition-colors hover:bg-white/5 focus:outline-none"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="flex items-center gap-3">
          <span className="font-body font-semibold text-sm" style={{ color: "#1A002E" }}>
            {title}
          </span>
          {badge && (
            <span
              className="text-xs px-2 py-0.5 rounded-full font-body font-medium"
              style={{ background: `${badgeColor}20`, color: badgeColor }}
            >
              {badge}
            </span>
          )}
        </div>
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          className="transition-transform"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        >
          <path
            d="M4 6l4 4 4-4"
            stroke="#7767A4"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>

      {open && (
        <div
          className="px-6 pb-6 pt-2 space-y-4"
          style={{ background: "rgba(255,255,255,0.6)" }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

function ConditionRow({
  label,
  sublabel,
  value,
  onChange,
}: {
  label: string;
  sublabel?: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-1.5">
      <div>
        <span className="font-body text-sm" style={{ color: "#1A002E" }}>
          {label}
        </span>
        {sublabel && (
          <p className="font-body text-xs" style={{ color: "rgba(26,0,46,0.5)" }}>
            {sublabel}
          </p>
        )}
      </div>
      <Toggle value={value} onChange={onChange} />
    </div>
  );
}

/* ── Main form component ─────────────────────────────────────────────────── */

interface Props {
  initial?: Partial<PatientFormData>;
  onSubmit: (data: PatientFormData) => void;
}

const defaults: PatientFormData = {
  age: 0,
  pathologies: [],
  habits: [],
  medical_history: [],
  obs_bmi: null,
  obs_systolic_bp: null,
  obs_diastolic_bp: null,
  obs_phq9_score: null,
  obs_testosterone: null,
  obs_smoker: 0,
  obs_pain_score: null,
  cond_migraine_with_aura: 0,
  cond_stroke: 0,
  cond_mi: 0,
  cond_dvt: 0,
  cond_breast_cancer: 0,
  cond_lupus: 0,
  cond_thrombophilia: 0,
  cond_atrial_fibrillation: 0,
  cond_liver_disease: 0,
  cond_hypertension: 0,
  cond_migraine: 0,
  cond_gallstones: 0,
  cond_diabetes: 0,
  cond_prediabetes: 0,
  cond_epilepsy: 0,
  cond_chronic_kidney_disease: 0,
  cond_sleep_apnea: 0,
  cond_pcos: 0,
  cond_endometriosis: 0,
  cond_depression: 0,
  cond_hypothyroidism: 0,
  cond_rheumatoid_arthritis: 0,
  cond_fibromyalgia: 0,
  cond_osteoporosis: 0,
  cond_asthma: 0,
  med_ever_ocp: 0,
  med_current_combined_ocp: 0,
  med_current_minipill: 0,
  has_absolute_contraindication_combined_oc: 0,
};

export default function ManualForm({ initial, onSubmit }: Props) {
  const [form, setForm] = useState<PatientFormData>({ ...defaults, ...initial });
  const [ageErr, setAgeErr] = useState("");

  const set = <K extends keyof PatientFormData>(key: K, val: PatientFormData[K]) =>
    setForm((f) => ({ ...f, [key]: val }));

  const flag = (key: keyof PatientFormData) => ({
    value: !!form[key],
    onChange: (v: boolean) => set(key, v ? 1 : 0),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.age || form.age < 15 || form.age > 60) {
      setAgeErr("Please enter an age between 15 and 60.");
      return;
    }
    setAgeErr("");
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* ── Basic info ─────────────────────────────────────────────────── */}
      <Section title="Basic info" defaultOpen>
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2 sm:col-span-1">
            <label
              className="block font-body text-xs font-medium mb-1"
              style={{ color: "rgba(26,0,46,0.65)" }}
            >
              Age <span style={{ color: "#F87171" }}>*</span>
            </label>
            <input
              type="number"
              required
              min={15}
              max={60}
              className="input-field"
              placeholder="e.g. 28"
              value={form.age || ""}
              onChange={(e) => {
                setAgeErr("");
                set("age", parseInt(e.target.value) || 0);
              }}
            />
            {ageErr && (
              <p className="text-xs mt-1 font-body" style={{ color: "#F87171" }}>
                {ageErr}
              </p>
            )}
          </div>
          <NumberInput
            label="BMI"
            placeholder="e.g. 22.5"
            value={form.obs_bmi}
            onChange={(v) => set("obs_bmi", v)}
            min={10}
            max={60}
            step={0.1}
            unit="kg/m²"
          />
          <NumberInput
            label="Systolic BP"
            placeholder="e.g. 118"
            value={form.obs_systolic_bp}
            onChange={(v) => set("obs_systolic_bp", v)}
            min={70}
            max={220}
            unit="mmHg"
          />
          <NumberInput
            label="Diastolic BP"
            placeholder="e.g. 76"
            value={form.obs_diastolic_bp}
            onChange={(v) => set("obs_diastolic_bp", v)}
            min={40}
            max={140}
            unit="mmHg"
          />
          <NumberInput
            label="PHQ-9 score"
            placeholder="0 – 27"
            value={form.obs_phq9_score}
            onChange={(v) => set("obs_phq9_score", v)}
            min={0}
            max={27}
          />
          <NumberInput
            label="Testosterone"
            placeholder="ng/dL"
            value={form.obs_testosterone}
            onChange={(v) => set("obs_testosterone", v)}
            min={0}
            max={300}
            unit="ng/dL"
          />
          <NumberInput
            label="Pain score"
            placeholder="0 – 10"
            value={form.obs_pain_score}
            onChange={(v) => set("obs_pain_score", v)}
            min={0}
            max={10}
            step={0.5}
            unit="/10"
          />
        </div>
        <ConditionRow label="Current smoker" {...flag("obs_smoker")} />
      </Section>

      {/* ── Cat 4 conditions ───────────────────────────────────────────── */}
      <Section
        title="Serious conditions"
        badge="WHO MEC Category 4"
        badgeColor="#F87171"
      >
        <p className="font-body text-xs mb-3" style={{ color: "rgba(26,0,46,0.55)" }}>
          Category 4 = absolute contraindication for combined oral contraceptives.
        </p>
        {[
          { key: "cond_migraine_with_aura" as const, label: "Migraine with aura" },
          { key: "cond_stroke" as const, label: "Stroke (history)" },
          { key: "cond_mi" as const, label: "Heart attack (MI)" },
          { key: "cond_dvt" as const, label: "Deep vein thrombosis (DVT)" },
          { key: "cond_breast_cancer" as const, label: "Breast cancer" },
          { key: "cond_lupus" as const, label: "Lupus" },
          { key: "cond_thrombophilia" as const, label: "Thrombophilia" },
          { key: "cond_atrial_fibrillation" as const, label: "Atrial fibrillation" },
          { key: "cond_liver_disease" as const, label: "Liver disease" },
        ].map(({ key, label }) => (
          <ConditionRow key={key} label={label} {...flag(key)} />
        ))}
      </Section>

      {/* ── Cat 3 conditions ───────────────────────────────────────────── */}
      <Section
        title="Monitored conditions"
        badge="WHO MEC Category 3"
        badgeColor="#FCD34D"
      >
        <p className="font-body text-xs mb-3" style={{ color: "rgba(26,0,46,0.55)" }}>
          Category 3 = use with caution; some combined pill options remain.
        </p>
        {[
          { key: "cond_hypertension" as const, label: "Hypertension" },
          { key: "cond_migraine" as const, label: "Migraine (without aura)" },
          { key: "cond_gallstones" as const, label: "Gallstones" },
          { key: "cond_diabetes" as const, label: "Diabetes" },
          { key: "cond_prediabetes" as const, label: "Pre-diabetes" },
          { key: "cond_epilepsy" as const, label: "Epilepsy" },
          {
            key: "cond_chronic_kidney_disease" as const,
            label: "Chronic kidney disease",
          },
          { key: "cond_sleep_apnea" as const, label: "Sleep apnea" },
        ].map(({ key, label }) => (
          <ConditionRow key={key} label={label} {...flag(key)} />
        ))}
      </Section>

      {/* ── Indications / other conditions ─────────────────────────────── */}
      <Section title="Other conditions">
        {[
          { key: "cond_pcos" as const, label: "PCOS", sublabel: "Polycystic ovary syndrome" },
          { key: "cond_endometriosis" as const, label: "Endometriosis" },
          { key: "cond_depression" as const, label: "Depression" },
          { key: "cond_hypothyroidism" as const, label: "Hypothyroidism" },
          { key: "cond_rheumatoid_arthritis" as const, label: "Rheumatoid arthritis" },
          { key: "cond_fibromyalgia" as const, label: "Fibromyalgia" },
          { key: "cond_osteoporosis" as const, label: "Osteoporosis" },
          { key: "cond_asthma" as const, label: "Asthma" },
        ].map(({ key, label, sublabel }) => (
          <ConditionRow
            key={key}
            label={label}
            sublabel={sublabel}
            {...flag(key)}
          />
        ))}
      </Section>

      {/* ── OCP history ─────────────────────────────────────────────────── */}
      <Section title="Contraceptive history">
        <ConditionRow
          label="Ever taken a combined OCP"
          {...flag("med_ever_ocp")}
        />
        <ConditionRow
          label="Currently on a combined OCP"
          {...flag("med_current_combined_ocp")}
        />
        <ConditionRow
          label="Currently on the mini-pill"
          {...flag("med_current_minipill")}
        />
      </Section>

      {/* ── Submit ──────────────────────────────────────────────────────── */}
      <button type="submit" className="btn-primary w-full py-4 text-base mt-2">
        Run analysis →
      </button>
    </form>
  );
}
