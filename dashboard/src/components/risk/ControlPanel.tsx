"use client";

import { AlertTriangle, XCircle } from "lucide-react";
import type { ControlState } from "@/types";

interface ControlPanelProps {
  state: ControlState;
  onChange: (key: string, value: boolean | number) => void;
}

interface ToggleConfig {
  key: keyof ControlState;
  label: string;
  description: string;
  color: "red" | "yellow" | "default";
}

const toggles: ToggleConfig[] = [
  {
    key: "tradingPaused",
    label: "Pause All Trading",
    description: "Halt all new orders and exits",
    color: "red",
  },
  {
    key: "entriesPaused",
    label: "Pause Entries",
    description: "Block new position entries only",
    color: "red",
  },
  {
    key: "exitsOnly",
    label: "Exits Only Mode",
    description: "Allow only position exits",
    color: "red",
  },
  {
    key: "riskOffMode",
    label: "Risk-Off Mode",
    description: "Tighten all limits, reduce sizing",
    color: "yellow",
  },
  {
    key: "manualApprovalMode",
    label: "Manual Approval Mode",
    description: "Require manual approval for all trades",
    color: "yellow",
  },
];

interface SliderConfig {
  key: keyof ControlState;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}

const sliders: SliderConfig[] = [
  {
    key: "maxPositionSizePct",
    label: "Max Position Size",
    min: 0,
    max: 10,
    step: 0.5,
    unit: "%",
  },
  {
    key: "maxExposurePct",
    label: "Max Exposure",
    min: 0,
    max: 200,
    step: 5,
    unit: "%",
  },
];

function ToggleSwitch({
  enabled,
  onChange,
  color = "default",
}: {
  enabled: boolean;
  onChange: (val: boolean) => void;
  color?: "red" | "yellow" | "default";
}) {
  const trackColors = {
    red: enabled ? "bg-red-500" : "bg-slate-200",
    yellow: enabled ? "bg-amber-500" : "bg-slate-200",
    default: enabled ? "bg-indigo-600" : "bg-slate-200",
  };

  return (
    <button
      role="switch"
      aria-checked={enabled}
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full transition-colors ${trackColors[color]}`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform ${
          enabled ? "translate-x-[18px]" : "translate-x-[3px]"
        }`}
      />
    </button>
  );
}

export function ControlPanel({ state, onChange }: ControlPanelProps) {
  return (
    <div className="flex h-full flex-col rounded-xl border border-[var(--border)] bg-white shadow-sm">
      {/* Header */}
      <div className="border-b border-[var(--border)] px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="h-1 w-1 rounded-full bg-[#F59E0B]" />
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">System Controls</h3>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {/* Toggles */}
        <div className="space-y-3">
          {toggles.map((toggle) => {
            const value = state[toggle.key] as boolean;
            return (
              <div
                key={toggle.key}
                className="flex items-center justify-between gap-3 rounded-md px-2 py-1.5"
              >
                <div className="min-w-0">
                  <p className="text-xs font-medium text-[var(--text-primary)]">
                    {toggle.label}
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)]">
                    {toggle.description}
                  </p>
                </div>
                <ToggleSwitch
                  enabled={value}
                  onChange={(v) => onChange(toggle.key, v)}
                  color={toggle.color}
                />
              </div>
            );
          })}
        </div>

        {/* Divider */}
        <div className="my-4 border-t border-slate-100" />

        {/* Sliders */}
        <div className="space-y-4">
          {sliders.map((slider) => {
            const value = state[slider.key] as number;
            return (
              <div key={slider.key} className="px-2">
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="text-xs font-medium text-[var(--text-primary)]">
                    {slider.label}
                  </span>
                  <span className="font-mono text-xs tabular-nums text-[var(--text-secondary)]">
                    {value}{slider.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={value}
                  onChange={(e) => onChange(slider.key, parseFloat(e.target.value))}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-100 accent-indigo-600 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-indigo-600"
                />
                <div className="flex justify-between text-[9px] text-[var(--text-muted)]">
                  <span>{slider.min}{slider.unit}</span>
                  <span>{slider.max}{slider.unit}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Divider */}
        <div className="my-4 border-t border-slate-100" />

        {/* Action buttons */}
        <div className="space-y-2 px-2">
          <button className="flex w-full items-center justify-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-xs font-semibold text-amber-600 transition-colors hover:bg-amber-100">
            <AlertTriangle size={14} />
            Tighten Thresholds
          </button>
          <button className="flex w-full items-center justify-center gap-2 rounded-lg border border-orange-200 bg-orange-50 px-4 py-2 text-xs font-semibold text-orange-600 transition-colors hover:bg-orange-100">
            <XCircle size={14} />
            Cancel All Orders
          </button>
        </div>
      </div>
    </div>
  );
}
