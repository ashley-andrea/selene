import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Expose NEXT_PUBLIC_API_URL to the browser bundle at build time
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "https://bc-recommendation-agent.onrender.com",
  },
};

export default nextConfig;
