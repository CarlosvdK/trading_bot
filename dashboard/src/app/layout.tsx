import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import "./globals.css";

const inter = Inter({ variable: "--font-sans", subsets: ["latin"] });
const jetbrainsMono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Trading Ops Dashboard",
  description: "Multi-agent trading operations dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-[family-name:var(--font-sans)] antialiased`}>
        <Sidebar />
        <TopBar />
        <main className="ml-60 mt-14 min-h-[calc(100vh-3.5rem)] bg-[var(--bg-base)] p-6">
          {children}
        </main>
      </body>
    </html>
  );
}
