import Link from "next/link";
import Navbar from "@/components/layout/Navbar";
import Stars from "@/components/home/Stars";

/* Orbital ring decorative element */
function OrbitalRing({
  size,
  opacity,
  duration,
  delay = "0s",
}: {
  size: number;
  opacity: number;
  duration: string;
  delay?: string;
}) {
  return (
    <div
      className="absolute rounded-full border"
      style={{
        width: size,
        height: size,
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        borderColor: `rgba(119, 103, 164, ${opacity})`,
        animation: `spin ${duration} linear infinite`,
        animationDelay: delay,
      }}
    />
  );
}

/* Moon crescent icon */
function MoonIcon({ size = 56 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 56 56"
      fill="none"
      aria-hidden
    >
      <path
        d="M42 29.12A16 16 0 1 1 26.88 14 12 12 0 0 0 42 29.12z"
        fill="#7767A4"
        opacity="0.3"
      />
      <path
        d="M42 29.12A16 16 0 1 1 26.88 14 12 12 0 0 0 42 29.12z"
        stroke="#DDD8C4"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Background gradient */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% 0%, #35285A 0%, #1A002E 55%, #0D0017 100%)",
          zIndex: 0,
        }}
      />
      <Stars />
      <Navbar />

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section
        className="relative flex flex-col items-center justify-center min-h-screen px-6 text-center"
        style={{ zIndex: 1 }}
      >
        {/* Orbital decoration */}
        <div className="relative w-64 h-64 mb-10 flex items-center justify-center">
          <OrbitalRing size={240} opacity={0.12} duration="28s" />
          <OrbitalRing size={180} opacity={0.18} duration="20s" delay="-5s" />
          <OrbitalRing size={120} opacity={0.25} duration="14s" delay="-8s" />

          {/* Orbiting dot */}
          <div
            className="absolute"
            style={{
              top: "50%",
              left: "50%",
              width: "10px",
              height: "10px",
              marginTop: "-5px",
              marginLeft: "-5px",
              animation: "animate-orbit 14s linear infinite",
            }}
          >
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: "#DDD8C4", boxShadow: "0 0 8px #DDD8C4" }}
            />
          </div>

          {/* Center moon */}
          <div style={{ animation: "float 6s ease-in-out infinite" }}>
            <MoonIcon size={64} />
          </div>
        </div>

        {/* Title */}
        <h1
          className="font-display glow-text mb-3"
          style={{
            fontFamily: "Catchye, 'Cormorant Garamond', Georgia, serif",
            fontSize: "clamp(3.5rem, 10vw, 7rem)",
            fontWeight: 400,
            lineHeight: 1,
            color: "#DDD8C4",
            letterSpacing: "0.04em",
          }}
        >
          Selene
        </h1>

        {/* Tagline */}
        <p
          className="font-body mb-12"
          style={{
            fontSize: "clamp(0.85rem, 2vw, 1.05rem)",
            color: "rgba(221, 216, 196, 0.55)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
          }}
        >
          every&nbsp;body works differently
        </p>

        {/* CTA */}
        <Link href="/intake" className="btn-primary text-base px-10 py-4">
          Begin your assessment
        </Link>

        {/* Scroll hint */}
        <div
          className="absolute bottom-10 flex flex-col items-center gap-1.5"
          style={{ color: "rgba(221,216,196,0.3)" }}
        >
          <span className="text-xs tracking-widest uppercase">scroll</span>
          <svg width="14" height="20" viewBox="0 0 14 20" fill="none">
            <rect
              x="1"
              y="1"
              width="12"
              height="18"
              rx="6"
              stroke="currentColor"
              strokeWidth="1.2"
            />
            <circle cx="7" cy="6" r="2" fill="currentColor">
              <animateTransform
                attributeName="transform"
                type="translate"
                values="0 0;0 7;0 0"
                dur="2s"
                repeatCount="indefinite"
              />
            </circle>
          </svg>
        </div>
      </section>

      {/* ── How it works ──────────────────────────────────────────────────── */}
      <section
        id="how-it-works"
        className="relative py-32 px-6"
        style={{ zIndex: 1 }}
      >
        <div className="max-w-5xl mx-auto">
          <p className="section-label text-center mb-3">How it works</p>
          <h2
            className="font-display text-center mb-16"
            style={{
              fontFamily: "Catchye, 'Cormorant Garamond', Georgia, serif",
              fontSize: "clamp(2rem, 5vw, 3.2rem)",
              color: "#DDD8C4",
            }}
          >
            From profile to personalised plan
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                num: "01",
                title: "Share your profile",
                body: "Upload a medical PDF or fill in a short form. Only your age is required — add as much or as little detail as you have.",
              },
              {
                num: "02",
                title: "Safety screening",
                body: "Selene applies WHO Medical Eligibility Criteria to block contraindicated options before any simulation runs.",
              },
              {
                num: "03",
                title: "12-month simulation",
                body: "A validated ML model simulates your long-term trajectory on each remaining pill, scoring adherence, side-effect risk, and satisfaction.",
              },
            ].map((step) => (
              <div key={step.num} className="card p-8">
                <span
                  className="font-display block mb-4"
                  style={{
                    fontFamily: "Catchye, 'Cormorant Garamond', Georgia, serif",
                    fontSize: "2.5rem",
                    color: "rgba(119,103,164,0.5)",
                  }}
                >
                  {step.num}
                </span>
                <h3
                  className="font-body font-semibold text-cream mb-2 text-lg"
                  style={{ color: "#DDD8C4" }}
                >
                  {step.title}
                </h3>
                <p className="font-body text-sm" style={{ color: "rgba(221,216,196,0.6)" }}>
                  {step.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── About / disclaimer ────────────────────────────────────────────── */}
      <section
        id="about"
        className="relative py-20 px-6"
        style={{ zIndex: 1 }}
      >
        <div
          className="max-w-2xl mx-auto text-center card p-12"
        >
          <p className="section-label mb-4">Transparency</p>
          <p
            className="font-body text-sm leading-relaxed"
            style={{ color: "rgba(221,216,196,0.65)" }}
          >
            Selene is a clinical decision-support tool, not a replacement for a
            healthcare provider. All recommendations should be reviewed with your
            doctor or pharmacist before starting, stopping, or switching
            contraception.
          </p>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer
        className="relative py-10 px-6 text-center"
        style={{
          borderTop: "1px solid rgba(119,103,164,0.12)",
          color: "rgba(221,216,196,0.3)",
          fontSize: "0.75rem",
          zIndex: 1,
        }}
      >
        <span
          className="font-display mr-2"
          style={{ fontFamily: "Catchye, 'Cormorant Garamond', serif" }}
        >
          Selene
        </span>
        · For informational purposes only · Not medical advice
      </footer>

      {/* Inline keyframe for the orbiting dot (supplement globals.css) */}
      <style>{`
        @keyframes spin {
          from { transform: translate(-50%,-50%) rotate(0deg); }
          to   { transform: translate(-50%,-50%) rotate(360deg); }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50%       { transform: translateY(-10px); }
        }
      `}</style>
    </main>
  );
}
