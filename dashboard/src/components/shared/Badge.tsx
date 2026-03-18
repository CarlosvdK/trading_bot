import type { ReactNode } from "react";

type BadgeVariant = "default" | "success" | "warning" | "danger" | "info" | "neutral";

interface BadgeProps {
  label?: string;
  children?: ReactNode;
  variant?: BadgeVariant;
  color?: string;
  size?: "sm" | "md";
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: "bg-[var(--accent-light)] text-[var(--accent)]",
  success: "bg-[var(--positive-light)] text-[var(--positive)]",
  warning: "bg-[var(--warning-light)] text-[#B45309]",
  danger: "bg-[var(--negative-light)] text-[var(--negative)]",
  info: "bg-[var(--info-light)] text-[var(--info)]",
  neutral: "bg-slate-100 text-slate-500",
};

const colorMap: Record<string, string> = {
  green: "bg-[var(--positive-light)] text-[var(--positive)]",
  yellow: "bg-[var(--warning-light)] text-[#B45309]",
  red: "bg-[var(--negative-light)] text-[var(--negative)]",
  blue: "bg-[var(--info-light)] text-[var(--info)]",
  gray: "bg-slate-100 text-slate-500",
  purple: "bg-purple-50 text-purple-600",
};

export function Badge({ label, children, variant, color = "gray", size = "sm", className = "" }: BadgeProps) {
  const colorClass = variant ? variantStyles[variant] : colorMap[color] || colorMap.gray;
  const sizeClass = size === "sm" ? "text-[10px] px-1.5 py-0.5" : "text-xs px-2.5 py-0.5";
  return (
    <span className={`inline-flex items-center rounded-full font-medium ${colorClass} ${sizeClass} ${className}`}>
      {children ?? label}
    </span>
  );
}
