interface ProgressBarProps {
  value: number;
  max?: number;
  color?: string;
  label?: string;
  showValue?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
  formatValue?: (v: number) => string;
  formatMax?: (v: number) => string;
  warningThreshold?: number;
  dangerThreshold?: number;
}

export function ProgressBar({
  value,
  max = 1,
  color,
  label,
  showValue = false,
  size = "md",
  className = "",
  formatValue,
  formatMax,
  warningThreshold = 0.7,
  dangerThreshold = 0.9,
}: ProgressBarProps) {
  const pct = Math.min(Math.max((value / max) * 100, 0), 100);
  const ratio = pct / 100;

  const barColor =
    color ||
    (ratio >= dangerThreshold ? "#EF4444" : ratio >= warningThreshold ? "#F59E0B" : "#10B981");

  const hasFormatters = formatValue || formatMax;
  const displayValue = formatValue
    ? formatValue(value)
    : typeof value === "number" && max === 1
      ? `${(value * 100).toFixed(0)}%`
      : `${value.toFixed(2)}`;
  const displayMax = formatMax ? formatMax(max) : undefined;

  const heights = { sm: "h-1.5", md: "h-2", lg: "h-2.5" };
  const h = heights[size];

  return (
    <div className={`flex flex-col gap-1.5 ${className}`}>
      {(label || showValue || hasFormatters) && (
        <div className="flex items-center justify-between">
          {label && <span className="text-sm text-slate-500">{label}</span>}
          {(showValue || hasFormatters) && (
            <span className="text-sm font-semibold tabular-nums text-slate-600">
              {displayValue}
              {displayMax && <span className="font-normal text-slate-400"> / {displayMax}</span>}
            </span>
          )}
        </div>
      )}
      <div className={`w-full rounded-full bg-slate-100 ${h}`}>
        <div
          className={`rounded-full ${h} transition-all duration-500`}
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}
