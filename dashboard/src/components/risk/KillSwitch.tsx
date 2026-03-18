"use client";

import { useState, useEffect, useCallback } from "react";
import { ShieldAlert, ShieldOff, AlertTriangle } from "lucide-react";

interface KillSwitchProps {
  active: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
}

export function KillSwitch({ active, onActivate, onDeactivate }: KillSwitchProps) {
  const [phase, setPhase] = useState<"idle" | "confirming" | "countdown">("idle");
  const [confirmText, setConfirmText] = useState("");
  const [countdown, setCountdown] = useState(5);

  // For deactivation
  const [deactivatePhase, setDeactivatePhase] = useState<"idle" | "confirming">("idle");
  const [deactivateText, setDeactivateText] = useState("");

  const resetActivation = useCallback(() => {
    setPhase("idle");
    setConfirmText("");
    setCountdown(5);
  }, []);

  const resetDeactivation = useCallback(() => {
    setDeactivatePhase("idle");
    setDeactivateText("");
  }, []);

  // Countdown timer
  useEffect(() => {
    if (phase !== "countdown") return;
    if (countdown <= 0) {
      onActivate();
      resetActivation();
      return;
    }
    const timer = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [phase, countdown, onActivate, resetActivation]);

  const handleActivateClick = () => {
    setPhase("confirming");
  };

  const handleConfirmSubmit = () => {
    if (confirmText === "CONFIRM") {
      setPhase("countdown");
    }
  };

  const handleDeactivateClick = () => {
    setDeactivatePhase("confirming");
  };

  const handleDeactivateConfirm = () => {
    if (deactivateText === "CONFIRM") {
      onDeactivate();
      resetDeactivation();
    }
  };

  // ---- ACTIVE STATE ----
  if (active) {
    return (
      <div className="relative overflow-hidden rounded-xl border-2 border-red-300 bg-red-50 p-6 shadow-sm">
        {/* Pulsing background effect */}
        <div className="absolute inset-0 animate-pulse bg-red-100/50" />

        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-red-100 ring-2 ring-red-500">
              <ShieldOff size={28} className="text-red-600" />
            </div>
            <div>
              <h3 className="text-lg font-bold uppercase tracking-wider text-red-700">
                Kill Switch Active
              </h3>
              <p className="text-sm text-red-600">
                All trading halted. All positions being flattened.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {deactivatePhase === "idle" ? (
              <button
                onClick={handleDeactivateClick}
                className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-card)] px-6 py-2.5 text-sm font-semibold text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-card-hover)]"
              >
                Deactivate
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder='Type "CONFIRM"'
                  value={deactivateText}
                  onChange={(e) => setDeactivateText(e.target.value.toUpperCase())}
                  onKeyDown={(e) => e.key === "Enter" && handleDeactivateConfirm()}
                  className="w-36 rounded-md border border-[var(--border-subtle)] bg-[var(--bg-base)] px-3 py-2 text-sm font-mono text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
                  autoFocus
                />
                <button
                  onClick={handleDeactivateConfirm}
                  disabled={deactivateText !== "CONFIRM"}
                  className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition-opacity disabled:opacity-30"
                >
                  Confirm
                </button>
                <button
                  onClick={resetDeactivation}
                  className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-card)] px-3 py-2 text-sm text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // ---- ARMED / IDLE STATE ----
  return (
    <div className="rounded-xl border-2 border-red-200 bg-red-50 p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-red-100 ring-1 ring-red-300">
            <ShieldAlert size={28} className="text-red-600" />
          </div>
          <div>
            <h3 className="text-lg font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Emergency Kill Switch
            </h3>
            <p className="text-sm text-[var(--text-muted)]">
              Flatten all positions and halt all trading immediately
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {phase === "idle" && (
            <button
              onClick={handleActivateClick}
              className="rounded-lg border-2 border-red-500 bg-red-600 px-8 py-3 text-sm font-bold uppercase tracking-wider text-white transition-all hover:bg-red-700 active:bg-red-800"
            >
              Activate
            </button>
          )}

          {phase === "confirming" && (
            <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
              <AlertTriangle size={18} className="shrink-0 text-[#F59E0B]" />
              <span className="text-xs text-[var(--text-secondary)]">
                Type <span className="font-mono font-bold text-[var(--text-primary)]">CONFIRM</span> to activate:
              </span>
              <input
                type="text"
                placeholder="CONFIRM"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === "Enter" && handleConfirmSubmit()}
                className="w-28 rounded-md border border-red-300 bg-white px-3 py-1.5 text-center font-mono text-sm font-bold text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:border-red-500 focus:outline-none"
                autoFocus
              />
              <button
                onClick={handleConfirmSubmit}
                disabled={confirmText !== "CONFIRM"}
                className="rounded-lg bg-red-600 px-4 py-1.5 text-sm font-bold text-white transition-opacity disabled:opacity-30"
              >
                Execute
              </button>
              <button
                onClick={resetActivation}
                className="text-sm text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              >
                Cancel
              </button>
            </div>
          )}

          {phase === "countdown" && (
            <div className="flex items-center gap-4 rounded-lg border border-red-300 bg-red-50 px-6 py-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-600 font-mono text-lg font-bold text-white">
                {countdown}
              </div>
              <span className="text-sm font-semibold text-red-600">
                Activating kill switch...
              </span>
              <button
                onClick={resetActivation}
                className="rounded-md border border-[var(--border-subtle)] bg-[var(--bg-card)] px-4 py-1.5 text-sm font-medium text-[var(--text-primary)] hover:bg-[var(--bg-card-hover)]"
              >
                Abort
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
