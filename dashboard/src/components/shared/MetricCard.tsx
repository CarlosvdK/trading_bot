interface MetricCardProps {
  label: string;
  value: string;
  subValue?: string;
  trend?: "up" | "down" | "neutral";
  color?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  subValue,
  trend = "neutral",
  color = "#4F46E5",
  className = "",
}: MetricCardProps) {
  const trendColors = {
    up: "text-emerald-500",
    down: "text-red-500",
    neutral: "text-slate-400",
  };

  return (
    <div className={`rounded-2xl border border-slate-100 bg-white p-6 shadow-sm ${className}`}>
      <div className="mb-3 flex items-center gap-2">
        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm font-medium text-slate-400">{label}</span>
      </div>
      <div className="text-3xl font-bold tracking-tight text-slate-800">{value}</div>
      {subValue && (
        <div className={`mt-2 text-sm font-medium ${trendColors[trend]}`}>{subValue}</div>
      )}
    </div>
  );
}
