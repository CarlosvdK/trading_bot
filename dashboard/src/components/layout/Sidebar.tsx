"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, TrendingUp, ArrowUpDown, Brain, Users, Bot, Trophy, ShieldAlert, Settings,
} from "lucide-react";
import type { ComponentType } from "react";

interface NavItem { label: string; href: string; icon: ComponentType<{ size?: number; className?: string }> }
interface NavSection { title: string; items: NavItem[] }

const navSections: NavSection[] = [
  {
    title: "PORTFOLIO",
    items: [
      { label: "Dashboard", href: "/", icon: LayoutDashboard },
      { label: "Positions", href: "/positions", icon: TrendingUp },
      { label: "Orders", href: "/orders", icon: ArrowUpDown },
    ],
  },
  {
    title: "INTELLIGENCE",
    items: [
      { label: "Decisions", href: "/decisions", icon: Brain },
      { label: "Groups", href: "/groups", icon: Users },
      { label: "Agents", href: "/agents", icon: Bot },
      { label: "Leaderboard", href: "/leaderboard", icon: Trophy },
    ],
  },
  {
    title: "OPERATIONS",
    items: [
      { label: "Risk", href: "/risk", icon: ShieldAlert },
      { label: "Settings", href: "/settings", icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-60 flex-col border-r border-[var(--border)] bg-[var(--bg-card)]">
      {/* Brand */}
      <div className="flex h-14 items-center gap-2.5 px-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--accent)] text-white text-sm font-bold">T</div>
        <span className="text-sm font-semibold text-[var(--text-primary)] tracking-tight">Trading Ops</span>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {navSections.map((section) => (
          <div key={section.title} className="mb-6">
            <div className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              {section.title}
            </div>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const isActive = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors ${
                      isActive
                        ? "bg-[var(--accent-light)] text-[var(--accent)]"
                        : "text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    <Icon size={16} className="shrink-0" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-[var(--border)] px-5 py-3">
        <div className="text-[10px] text-[var(--text-muted)]">v0.1.0</div>
      </div>
    </aside>
  );
}
