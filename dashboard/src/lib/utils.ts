/**
 * Format a number as USD currency string.
 * Negative values are wrapped in parentheses: ($1,234.56)
 */
export function formatCurrency(value: number): string {
  const abs = Math.abs(value);
  const formatted = abs.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return value < 0 ? `(${formatted})` : formatted;
}

/**
 * Format a decimal as a percentage string with sign.
 * e.g. 0.0523 -> "+5.23%", -0.0523 -> "-5.23%"
 */
export function formatPct(value: number, decimals: number = 2): string {
  const pct = value * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(decimals)}%`;
}

/**
 * Format a number in compact form.
 * e.g. 1234567 -> "1.23M", 12345 -> "12.3K", 1234567890 -> "1.23B"
 */
export function formatNumber(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  if (abs >= 1_000_000_000) {
    return `${sign}${(abs / 1_000_000_000).toFixed(2)}B`;
  }
  if (abs >= 1_000_000) {
    return `${sign}${(abs / 1_000_000).toFixed(2)}M`;
  }
  if (abs >= 1_000) {
    return `${sign}${(abs / 1_000).toFixed(1)}K`;
  }
  return `${sign}${abs.toFixed(0)}`;
}

/**
 * Format an ISO date string as a short date.
 * e.g. "2026-03-18T14:32:00Z" -> "Mar 18, 2026"
 */
export function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

/**
 * Format an ISO date string as a short date+time.
 * e.g. "2026-03-18T14:32:00Z" -> "Mar 18 14:32"
 */
export function formatDateTime(dateStr: string): string {
  const date = new Date(dateStr);
  const month = date.toLocaleDateString("en-US", { month: "short" });
  const day = date.getDate();
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  return `${month} ${day} ${hours}:${minutes}`;
}

/**
 * Join class names, filtering out falsy values.
 * e.g. cn("foo", undefined, false, "bar") -> "foo bar"
 */
export function cn(...classes: (string | undefined | false)[]): string {
  return classes.filter(Boolean).join(" ");
}

/**
 * Return a Tailwind text color class based on PnL value.
 */
export function getPnlColor(value: number): string {
  if (value > 0) return "text-emerald-600";
  if (value < 0) return "text-red-500";
  return "text-slate-400";
}

/**
 * Return a Tailwind text color class based on status string.
 */
export function getStatusColor(status: string): string {
  switch (status) {
    case "healthy":
      return "text-emerald-600";
    case "warning":
      return "text-amber-500";
    case "near_stop":
    case "stopped_out":
    case "underperforming":
    case "replace":
      return "text-red-500";
    case "near_target":
      return "text-sky-500";
    case "exit_pending":
      return "text-orange-500";
    default:
      return "text-slate-400";
  }
}

/**
 * Return a Tailwind text color class based on confidence level (0-1).
 */
export function getConfidenceColor(value: number): string {
  if (value >= 0.8) return "text-emerald-600";
  if (value >= 0.65) return "text-sky-500";
  if (value >= 0.5) return "text-amber-500";
  return "text-red-500";
}

/**
 * Return a Tailwind text color class based on alert severity.
 */
export function getSeverityColor(severity: string): string {
  switch (severity) {
    case "critical":
      return "text-red-600";
    case "high":
      return "text-red-500";
    case "medium":
      return "text-amber-500";
    case "low":
      return "text-sky-500";
    case "info":
      return "text-slate-400";
    default:
      return "text-slate-400";
  }
}
