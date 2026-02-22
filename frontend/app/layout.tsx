import type { Metadata } from "next";
import { Epilogue } from "next/font/google";
import "./globals.css";

const epilogue = Epilogue({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-epilogue",
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Selene — every body works differently",
  description:
    "Personalised contraceptive recommendations powered by clinical AI. Selene analyses your profile and simulates long-term outcomes for each pill option.",
  keywords: ["contraception", "personalised medicine", "pill recommendation", "women's health"],
  openGraph: {
    title: "Selene",
    description: "every body works differently",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={epilogue.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
