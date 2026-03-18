"use client";

import { useEffect, useState } from "react";
import { Bell, Wifi, Clock, Activity } from "lucide-react";

function useCurrentTime() {
  const [time, setTime] = useState<string>("");

  useEffect(() => {
    const update = () => {
      setTime(
        new Date().toLocaleTimeString("en-US", {
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          timeZone: "America/New_York",
        })
      );
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  return time;
}

function isMarketOpen(): boolean {
  const now = new Date();
  const et = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" })
  );
  const day = et.getDay();
  const hours = et.getHours();
  const minutes = et.getMinutes();
  const totalMinutes = hours * 60 + minutes;
  // M-F, 9:30 - 16:00 ET
  return day >= 1 && day <= 5 && totalMinutes >= 570 && totalMinutes < 960;
}

export function TopBar() {
  const time = useCurrentTime();
  const [marketOpen, setMarketOpen] = useState(false);

  useEffect(() => {
    const check = () => setMarketOpen(isMarketOpen());
    check();
    const interval = setInterval(check, 30_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="fixed left-60 right-0 top-0 z-30 flex h-14 items-center justify-between border-b border-[var(--border)] bg-[var(--bg-card)] px-5">
      {/* Left: Connection & Market Status */}
      <div className="flex items-center gap-5">
        {/* Broker connection */}
        <div className="flex items-center gap-2 text-xs">
          <Wifi size={13} className="text-[var(--positive)]" />
          <span className="text-[var(--text-secondary)]">IBKR Connected</span>
        </div>

        {/* Divider */}
        <div className="h-4 w-px bg-[var(--border)]" />

        {/* Market status */}
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${
              marketOpen ? "bg-[var(--positive)]" : "bg-[var(--text-muted)]"
            }`}
          />
          <span className="text-[var(--text-secondary)]">
            {marketOpen ? "Market Open" : "Market Closed"}
          </span>
        </div>

        {/* Divider */}
        <div className="h-4 w-px bg-[var(--border)]" />

        {/* Sync */}
        <div className="flex items-center gap-2 text-xs">
          <Activity size={13} className="text-[var(--text-muted)]" />
          <span className="text-[var(--text-muted)]">Synced</span>
        </div>
      </div>

      {/* Center: Active regime */}
      <div className="flex items-center">
        <span className="rounded-full bg-[var(--accent-light)] px-3 py-1 text-xs font-medium text-[var(--accent)]">
          LOW VOL TRENDING UP
        </span>
      </div>

      {/* Right: Time & Alerts */}
      <div className="flex items-center gap-5">
        {/* Last update */}
        <span className="text-[10px] text-[var(--text-muted)]">
          Updated just now
        </span>

        {/* Divider */}
        <div className="h-4 w-px bg-[var(--border)]" />

        {/* Alerts */}
        <button className="relative flex items-center gap-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:text-[var(--text-primary)]">
          <Bell size={14} />
          <span className="absolute -right-1.5 -top-1.5 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[var(--negative)] text-[8px] font-bold text-white">
            2
          </span>
        </button>

        {/* Divider */}
        <div className="h-4 w-px bg-[var(--border)]" />

        {/* Clock */}
        <div className="flex items-center gap-2 text-xs tabular-nums text-[var(--text-secondary)]">
          <Clock size={13} />
          <span>{time || "--:--:--"} ET</span>
        </div>
      </div>
    </header>
  );
}
