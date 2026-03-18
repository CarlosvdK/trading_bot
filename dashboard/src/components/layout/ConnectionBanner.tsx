"use client";
import { useApi } from "@/hooks/useApi";
import { fetchHealth } from "@/lib/api";
import { Wifi, WifiOff } from "lucide-react";

export function ConnectionBanner() {
  const { data } = useApi(fetchHealth, 5000);

  if (data?.status === "ok") return null;

  return (
    <div className="fixed left-60 right-0 top-0 z-50 flex items-center justify-center gap-2 bg-red-50 border-b border-red-200 px-4 py-2 text-sm text-red-700">
      <WifiOff size={14} />
      <span>API Offline — run: <code className="rounded bg-red-100 px-1.5 py-0.5 font-mono text-xs">python scripts/run_api.py</code></span>
    </div>
  );
}
