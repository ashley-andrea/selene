"use client";

import { useEffect } from "react";
import { healthCheck } from "@/lib/api";

/**
 * Invisible component that pings the Render /health endpoint as soon as the
 * landing page loads. This wakes up the free-tier Render service ~20 seconds
 * before the user actually submits their intake form, eliminating cold-start
 * timeouts on the /recommend call.
 */
export default function WakeupPing() {
  useEffect(() => {
    // Fire and forget — we don't need the result, just want Render to wake up.
    healthCheck().catch(() => {
      // Silently ignore — this is just a best-effort warmup.
    });
  }, []);

  return null;
}
