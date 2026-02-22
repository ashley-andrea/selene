"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import type { PDFExtractionResult } from "@/lib/types";
import { uploadPDF } from "@/lib/api";

interface Props {
  onExtracted: (data: PDFExtractionResult) => void;
}

export default function UploadStep({ onExtracted }: Props) {
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return;
      setStatus("loading");
      setErrorMsg("");
      try {
        const result = await uploadPDF(accepted[0]);
        onExtracted(result);
      } catch (e: unknown) {
        setStatus("error");
        setErrorMsg(e instanceof Error ? e.message : "Upload failed. Please try again.");
      }
    },
    [onExtracted]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled: status === "loading",
  });

  return (
    <div className="space-y-6">
      <div
        {...getRootProps()}
        className="relative cursor-pointer rounded-2xl p-12 text-center transition-all duration-200"
        style={{
          border: `2px dashed ${isDragActive ? "#7767A4" : "rgba(119,103,164,0.35)"}`,
          background: isDragActive
            ? "rgba(119,103,164,0.08)"
            : "rgba(53,40,90,0.25)",
        }}
      >
        <input {...getInputProps()} />

        {status === "loading" ? (
          <div className="flex flex-col items-center gap-4">
            {/* Spinner */}
            <div
              className="w-10 h-10 rounded-full border-2"
              style={{
                borderColor: "rgba(119,103,164,0.3)",
                borderTopColor: "#7767A4",
                animation: "spin 0.8s linear infinite",
              }}
            />
            <p className="font-body text-sm" style={{ color: "rgba(221,216,196,0.6)" }}>
              Parsing your document…
            </p>
          </div>
        ) : (
          <>
            {/* Cloud / PDF icon */}
            <svg
              width="48"
              height="48"
              viewBox="0 0 48 48"
              fill="none"
              className="mx-auto mb-5"
            >
              <path
                d="M36 30c3.314 0 6-2.686 6-6 0-3.062-2.298-5.585-5.254-5.958A9.002 9.002 0 0 0 15.133 20H14c-3.314 0-6 2.686-6 6s2.686 6 6 6h22z"
                stroke="#7767A4"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M28 38l-4-4-4 4M24 34v-10"
                stroke="#7767A4"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>

            <p className="font-body font-semibold mb-1" style={{ color: "#DDD8C4" }}>
              {isDragActive ? "Drop your PDF here" : "Drag & drop your medical record"}
            </p>
            <p className="font-body text-sm" style={{ color: "rgba(221,216,196,0.45)" }}>
              or click to browse · PDF only
            </p>
          </>
        )}
      </div>

      {status === "error" && (
        <p
          className="text-sm text-center font-body"
          style={{ color: "#F87171" }}
        >
          {errorMsg}
        </p>
      )}

      <p
        className="text-xs text-center font-body"
        style={{ color: "rgba(221,216,196,0.35)" }}
      >
        Your document is processed server-side, never stored. Selene extracts
        clinical data only.
      </p>

      <style>{`@keyframes spin { to { transform:rotate(360deg); } }`}</style>
    </div>
  );
}
