"use client";

import { useState } from "react";
import { AlertTriangle, AlertCircle, Info, Check } from "lucide-react";
import type { Alert, AlertSeverity } from "@/types";
import { Badge } from "@/components/shared/Badge";
import { formatDateTime } from "@/lib/utils";

interface AlertPanelProps {
  alerts: Alert[];
}

const severityTabs: { label: string; value: AlertSeverity | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Critical", value: "critical" },
  { label: "High", value: "high" },
  { label: "Medium", value: "medium" },
  { label: "Low", value: "low" },
];

const severityConfig: Record<AlertSeverity, {
  icon: typeof AlertTriangle;
  stripeColor: string;
  dotColor: string;
  badgeVariant: "danger" | "warning" | "info" | "neutral";
}> = {
  critical: {
    icon: AlertTriangle,
    stripeColor: "bg-[#DC2626]",
    dotColor: "bg-[#DC2626]",
    badgeVariant: "danger",
  },
  high: {
    icon: AlertCircle,
    stripeColor: "bg-[#F59E0B]",
    dotColor: "bg-[#F59E0B]",
    badgeVariant: "warning",
  },
  medium: {
    icon: Info,
    stripeColor: "bg-[#3B82F6]",
    dotColor: "bg-[#3B82F6]",
    badgeVariant: "info",
  },
  low: {
    icon: Info,
    stripeColor: "bg-[#6B7280]",
    dotColor: "bg-[#6B7280]",
    badgeVariant: "neutral",
  },
  info: {
    icon: Info,
    stripeColor: "bg-[#6B7280]",
    dotColor: "bg-[#6B7280]",
    badgeVariant: "neutral",
  },
};

export function AlertPanel({ alerts: initialAlerts }: AlertPanelProps) {
  const [activeTab, setActiveTab] = useState<AlertSeverity | "all">("all");
  const [alerts, setAlerts] = useState(initialAlerts);

  const filtered = activeTab === "all"
    ? alerts
    : alerts.filter((a) => a.severity === activeTab);

  const unacknowledgedCount = alerts.filter((a) => !a.acknowledged).length;

  const handleAcknowledge = (id: string) => {
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a))
    );
  };

  const countBySeverity = (sev: AlertSeverity) =>
    alerts.filter((a) => a.severity === sev && !a.acknowledged).length;

  return (
    <div className="flex h-full flex-col rounded-xl border border-[var(--border)] bg-white shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">Alerts</h3>
          {unacknowledgedCount > 0 && (
            <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-[#DC2626] px-1.5 text-[10px] font-bold text-white">
              {unacknowledgedCount}
            </span>
          )}
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 border-b border-[var(--border)] px-3 py-2">
        {severityTabs.map((tab) => {
          const count = tab.value === "all"
            ? unacknowledgedCount
            : countBySeverity(tab.value);
          const isActive = activeTab === tab.value;
          return (
            <button
              key={tab.value}
              onClick={() => setActiveTab(tab.value)}
              className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                isActive
                  ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              {tab.label}
              {count > 0 && (
                <span className={`text-[10px] ${isActive ? "text-[var(--accent)]" : "text-[var(--text-muted)]"}`}>
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center py-12 text-sm text-[var(--text-muted)]">
            No alerts in this category
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {filtered.map((alert) => {
              const config = severityConfig[alert.severity];
              const Icon = config.icon;
              const isCritical = alert.severity === "critical";

              return (
                <div
                  key={alert.id}
                  className={`relative flex gap-3 px-4 py-3 transition-colors hover:bg-[var(--bg-card-hover)] ${
                    alert.acknowledged ? "opacity-50" : ""
                  }`}
                >
                  {/* Severity stripe */}
                  <div className={`absolute left-0 top-0 h-full w-1 ${config.stripeColor}`} />

                  {/* Icon */}
                  <div className="relative mt-0.5 shrink-0">
                    <Icon size={16} className={`${
                      alert.severity === "critical" ? "text-[#DC2626]" :
                      alert.severity === "high" ? "text-[#F59E0B]" :
                      alert.severity === "medium" ? "text-[#3B82F6]" :
                      "text-[#6B7280]"
                    }`} />
                    {isCritical && !alert.acknowledged && (
                      <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-ping rounded-full bg-[#DC2626]" />
                    )}
                  </div>

                  {/* Content */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-[var(--text-primary)]">
                          {alert.title}
                        </p>
                        <p className="mt-0.5 text-[11px] leading-relaxed text-[var(--text-muted)]">
                          {alert.message}
                        </p>
                      </div>
                      <span className="shrink-0 text-[10px] tabular-nums text-[var(--text-muted)]">
                        {formatDateTime(alert.timestamp)}
                      </span>
                    </div>

                    <div className="mt-1.5 flex items-center gap-2">
                      {alert.relatedSymbol && (
                        <Badge variant="info" size="sm">
                          {alert.relatedSymbol}
                        </Badge>
                      )}
                      {alert.relatedAgent && (
                        <Badge variant="neutral" size="sm">
                          {alert.relatedAgent}
                        </Badge>
                      )}
                      <Badge variant={config.badgeVariant} size="sm">
                        {alert.severity}
                      </Badge>

                      {!alert.acknowledged && (
                        <button
                          onClick={() => handleAcknowledge(alert.id)}
                          className="ml-auto flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium text-[var(--text-muted)] transition-colors hover:bg-slate-100 hover:text-[var(--text-secondary)]"
                        >
                          <Check size={10} />
                          Ack
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
