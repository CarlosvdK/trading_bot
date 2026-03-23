import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Sidebar } from "@/components/layout/Sidebar";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "MulaMachina — Multi-Agent Trading",
  description: "121-agent algorithmic trading system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-[family-name:var(--font-sans)] antialiased`}>
        <Sidebar />
        <main className="ml-[220px] min-h-screen bg-[var(--bg-base)]">
          {children}
        </main>
      </body>
    </html>
  );
}
