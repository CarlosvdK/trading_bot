"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Brain, Wifi, WifiOff, Database, Clock } from "lucide-react";
import { useEffect, useState } from "react";
import { useApi } from "@/hooks/useApi";
import { fetchHealth } from "@/lib/api";

function useClock() {
  const [time, setTime] = useState("");
  useEffect(() => {
    const tick = () =>
      setTime(
        new Date().toLocaleTimeString("en-US", {
          hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
          timeZone: "America/New_York",
        })
      );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

function isMarketOpen(): boolean {
  const now = new Date();
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const day = et.getDay();
  const totalMinutes = et.getHours() * 60 + et.getMinutes();
  return day >= 1 && day <= 5 && totalMinutes >= 570 && totalMinutes < 960;
}

const nav = [
  { label: "Portfolio", href: "/", icon: BarChart3 },
  { label: "Agent Swarm", href: "/swarm", icon: Brain },
];

export function Sidebar() {
  const pathname = usePathname();
  const { data: health } = useApi(fetchHealth, 5000);
  const time = useClock();
  const [marketOpen, setMarketOpen] = useState(false);

  useEffect(() => {
    const check = () => setMarketOpen(isMarketOpen());
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  const ibkr = health?.ibkrConnected || health?.ibkr_connected || false;
  const db = health?.supabaseConnected || health?.supabase_connected || false;
  const online = health?.status === "ok";

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-[220px] flex-col bg-white border-r border-[var(--border)]">
      {/* Brand */}
      <div className="px-6 pt-8 pb-6">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-[var(--accent)] text-white text-xs font-black">
            T
          </div>
          <div>
            <h1 className="text-lg font-black tracking-tight text-[var(--text-primary)]">TradeOps</h1>
            <p className="text-[10px] text-[var(--accent)] font-medium">121-agent swarm</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3">
        {nav.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-xl px-4 py-3 mb-1 text-[13px] font-semibold transition-all duration-200 ${
                active
                  ? "bg-[var(--accent)] text-white shadow-md shadow-blue-200"
                  : "text-[var(--text-secondary)] hover:bg-[var(--accent-light)] hover:text-[var(--accent)]"
              }`}
            >
              <Icon size={17} strokeWidth={active ? 2.5 : 2} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Status Footer */}
      <div className="border-t border-[var(--border)] px-5 py-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-[var(--text-muted)]">Market</span>
          <div className="flex items-center gap-1.5">
            <span className={`h-1.5 w-1.5 rounded-full ${marketOpen ? "bg-[var(--positive)] animate-pulse-dot" : "bg-[var(--text-muted)]"}`} />
            <span className={`text-[11px] font-medium ${marketOpen ? "text-[var(--positive)]" : "text-[var(--text-muted)]"}`}>
              {marketOpen ? "Open" : "Closed"}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-[var(--text-muted)]">IBKR</span>
          <div className="flex items-center gap-1.5">
            {ibkr ? <Wifi size={10} className="text-[var(--positive)]" /> : <WifiOff size={10} className="text-[var(--text-muted)]" />}
            <span className={`text-[11px] font-medium ${ibkr ? "text-[var(--positive)]" : "text-[var(--text-muted)]"}`}>
              {ibkr ? "Connected" : "Offline"}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-[var(--text-muted)]">Database</span>
          <div className="flex items-center gap-1.5">
            <Database size={10} className={db ? "text-[var(--positive)]" : "text-[var(--text-muted)]"} />
            <span className={`text-[11px] font-medium ${db ? "text-[var(--positive)]" : "text-[var(--text-muted)]"}`}>
              {db ? "Connected" : "Offline"}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between pt-1 border-t border-[var(--border)]">
          <div className="flex items-center gap-1.5">
            <Clock size={10} className="text-[var(--text-muted)]" />
            <span className="text-[11px] text-[var(--text-muted)]">ET</span>
          </div>
          <span className="text-[11px] font-medium tabular-nums text-[var(--text-secondary)]">{time || "--:--:--"}</span>
        </div>
        {!online && (
          <div className="mt-2 rounded-lg bg-[var(--negative-light)] px-3 py-2 text-[10px] text-[var(--negative)] font-medium">
            API Offline
          </div>
        )}
      </div>
    </aside>
  );
}
