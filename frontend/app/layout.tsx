import type { Metadata } from "next";
import { Archivo, Inter, JetBrains_Mono } from "next/font/google";
import TopBar from "@/components/TopBar";
import "./globals.css";

// Fonts are downloaded at build time and served locally, so the demo works offline
const archivo = Archivo({ subsets: ["latin"], weight: ["800"], variable: "--font-archivo" });
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], weight: ["500", "600", "700"], variable: "--font-jetbrains" });

export const metadata: Metadata = {
  title: "CaseTrail",
  description: "Precedent and statute retrieval for Indian court judgments (CSD358 IR project)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${archivo.variable} ${inter.variable} ${jetbrains.variable}`}>
      <body className="min-h-screen">
        <TopBar />
        <main>{children}</main>
      </body>
    </html>
  );
}
