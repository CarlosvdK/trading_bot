"use client";

import type { AllocationBreakdown } from "@/types";
import { formatCurrency } from "@/lib/utils";

interface AllocationPanelProps {
  title: string;
  data: AllocationBreakdown[];
  className?: string;
}

export function AllocationPanel({ title, data, className = "" }: AllocationPanelProps) {
  const maxPct = Math.max(...data.map((d) => d.pct), 0.01);

  return (
    <div className={`rounded-2xl border border-slate-100 bg-white p-6 shadow-sm ${className}`}>
      <h3 className="mb-4 text-base font-semibold text-slate-700">{title}</h3>
      <div className="space-y-3.5">
        {data.map((item) => (
          <div key={item.label}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-slate-600">{item.label}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-slate-400 tabular-nums">
                  {formatCurrency(item.value)}
                </span>
                <span className="w-14 text-right text-sm font-semibold text-slate-600 tabular-nums">
                  {(item.pct * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${(item.pct / maxPct) * 100}%`,
                  backgroundColor: item.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
