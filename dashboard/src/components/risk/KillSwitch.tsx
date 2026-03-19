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

  // ---- ACTIVE STATE ----
  if (active) {
    return (
      <div className="relative overflow-hidden card border-2 border-red-200 bg-red-50 p-6">
        <div className="absolute inset-0 animate-pulse bg-red-100/30" />
        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-red-100">
              <ShieldOff size={24} className="text-red-600" />
            </div>
            <div>
              <h3 className="text-base font-black uppercase tracking-wider text-red-700">
                Kill Switch Active
              </h3>
              <p className="text-xs text-red-500 mt-0.5">
                All trading halted. Positions being flattened.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {deactivatePhase === "idle" ? (
              <button
                onClick={() => setDeactivatePhase("confirming")}
                className="rounded-xl bg-white px-6 py-2.5 text-sm font-bold text-[var(--text-primary)] shadow-sm transition-all hover:shadow-md"
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
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && deactivateText === "CONFIRM") {
                      onDeactivate();
                      resetDeactivation();
                    }
                  }}
                  className="w-32 rounded-xl border border-[var(--border)] bg-white px-3 py-2 text-sm font-mono text-[var(--text-primary)] placeholder-[var(--text-muted)] outline-none"
                  autoFocus
                />
                <button
                  onClick={() => {
                    if (deactivateText === "CONFIRM") {
                      onDeactivate();
                      resetDeactivation();
                    }
                  }}
                  disabled={deactivateText !== "CONFIRM"}
                  className="rounded-xl bg-[var(--text-primary)] px-4 py-2 text-sm font-bold text-white transition-opacity disabled:opacity-30"
                >
                  Confirm
                </button>
                <button
                  onClick={resetDeactivation}
                  className="text-sm text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
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

  // ---- ARMED STATE ----
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-red-50">
            <ShieldAlert size={24} className="text-red-500" />
          </div>
          <div>
            <h3 className="text-base font-black text-[var(--text-primary)]">
              Emergency Kill Switch
            </h3>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Flatten all positions and halt all trading
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {phase === "idle" && (
            <button
              onClick={() => setPhase("confirming")}
              className="rounded-xl border-2 border-red-400 bg-red-500 px-8 py-2.5 text-sm font-bold uppercase tracking-wider text-white transition-all hover:bg-red-600 active:bg-red-700"
            >
              Activate
            </button>
          )}

          {phase === "confirming" && (
            <div className="flex items-center gap-3 rounded-xl bg-[var(--bg-base)] px-4 py-3">
              <AlertTriangle size={16} className="shrink-0 text-[var(--warning)]" />
              <span className="text-xs text-[var(--text-secondary)]">
                Type <span className="font-mono font-bold">CONFIRM</span>
              </span>
              <input
                type="text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value.toUpperCase())}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && confirmText === "CONFIRM") setPhase("countdown");
                }}
                className="w-24 rounded-lg border border-[var(--border)] bg-white px-3 py-1.5 text-center font-mono text-sm font-bold outline-none"
                autoFocus
              />
              <button
                onClick={() => { if (confirmText === "CONFIRM") setPhase("countdown"); }}
                disabled={confirmText !== "CONFIRM"}
                className="rounded-lg bg-red-500 px-4 py-1.5 text-sm font-bold text-white disabled:opacity-30"
              >
                Execute
              </button>
              <button onClick={resetActivation} className="text-sm text-[var(--text-muted)]">
                Cancel
              </button>
            </div>
          )}

          {phase === "countdown" && (
            <div className="flex items-center gap-4 rounded-xl bg-red-50 px-6 py-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500 font-mono text-lg font-black text-white">
                {countdown}
              </div>
              <span className="text-sm font-bold text-red-600">Activating...</span>
              <button
                onClick={resetActivation}
                className="rounded-lg bg-white px-4 py-1.5 text-sm font-semibold text-[var(--text-primary)] shadow-sm"
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
