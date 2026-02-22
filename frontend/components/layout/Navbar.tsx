"use client";

import Link from "next/link";
import Image from "next/image";
import { useEffect, useState } from "react";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
      style={{
        background: scrolled
          ? "rgba(26, 0, 46, 0.92)"
          : "transparent",
        backdropFilter: scrolled ? "blur(16px)" : "none",
        borderBottom: scrolled ? "1px solid rgba(119,103,164,0.2)" : "none",
      }}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 relative flex-shrink-0">
            <Image
              src="/logo.png"
              alt="Selene logo"
              fill
              className="object-contain"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
            {/* Fallback moon SVG if logo not yet placed */}
            <svg
              viewBox="0 0 24 24"
              fill="none"
              className="absolute inset-0 w-full h-full"
              aria-hidden
            >
              <path
                d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"
                stroke="#DDD8C4"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <span
            className="font-display text-xl tracking-wide text-cream group-hover:glow-text transition-all"
            style={{ fontFamily: "Catchye, 'Cormorant Garamond', Georgia, serif" }}
          >
            Selene
          </span>
        </Link>

        {/* Links */}
        <div className="flex items-center gap-1">
          <Link href="#how-it-works" className="btn-ghost text-sm hidden sm:block">
            How it works
          </Link>
          <Link href="#about" className="btn-ghost text-sm hidden sm:block">
            About
          </Link>
          <Link href="/intake" className="btn-primary text-sm ml-2">
            Get started
          </Link>
        </div>
      </div>
    </nav>
  );
}
