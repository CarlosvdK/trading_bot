# Trading Bot — Task Tracker

## Status Legend
- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Done
- `[!]` — Blocked

---

## Phase 0: Safety Foundation (P0)
- [ ] Risk Governor — all limits, kill-switch, PDT rule
- [ ] Data Validator — OHLCV checks, hash registry
- [ ] CSV Data Provider — adjusted OHLCV, corporate actions
- [ ] Config Loader + Validator — range checks, reject unsafe values
- [ ] Secrets Manager — env var loading, no credentials in config
- [ ] Unit tests for all P0 components

## Phase 1: Core Logic (P1)
- [ ] Backtest Engine — portfolio accounting per sleeve
- [ ] Cost Model — commission, spread, impact, borrow, T+2 settlement
- [ ] Core Rebalance Engine — drift-triggered, liquidity gate, MOC orders
- [ ] Barrier Labeler — triple-barrier, leak-proof vol proxy
- [ ] Feature Builder — all 9 feature types, winsorization, VIF check
- [ ] Trade Filter Model — binary classifier, walk-forward, calibration
- [ ] Walk-Forward Pipeline — embargo, purging, OOS Sharpe gate
- [ ] Unit tests for all P1 components

## Phase 2: Execution & Intelligence (P2)
- [ ] Paper Broker — fill simulation, partial fills, slippage
- [ ] Order Manager — rate limiting, order log, session tags
- [ ] Reconciliation Job — daily state comparison
- [ ] Regime Detection — HMM/clustering, allocation adjustment
- [ ] Swing Signal Generator — momentum, vol breakout, risk-on gate
- [ ] Position Sizer — vol-targeting, ML confidence scaling
- [ ] Unit tests for all P2 components

## Phase 3: Monitoring (P3)
- [ ] Drift Monitor — weekly PSI on top-10 features
- [ ] Performance Monitor — daily Brier score, realized Sharpe
- [ ] Alerting — webhook with HMAC, heartbeat monitor
- [ ] Logging — append-only, hash-chained audit log

## Phase 4: Optional (P4)
- [ ] Vol Forecast Model — regression, wire to barrier width
- [ ] Scalp Stubs — interface only, no execution logic

## Infrastructure
- [ ] requirements.txt — pinned exact versions
- [ ] Dockerfile — reproducible environment
- [ ] README.md — with all required risk warnings
- [ ] example.yaml + schema.yaml configs
- [ ] .gitignore updates for .env, data/, logs/

---

## Current Sprint
_Update this section when starting a new sprint._

**Sprint Goal:** _TBD_
**Active Tasks:** _None yet_
