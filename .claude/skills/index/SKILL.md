---
name: index
description: Master skills index — routing table, implementation order, config template, and cross-cutting rules. Consult FIRST when unsure which skill to use.
---

# Algorithmic Trading Bot — Skills Index

## Overview
This directory contains 12 skill files covering every domain needed to build a production-quality multi-sleeve algorithmic trading system. Each file is self-contained and can be given directly to a coding assistant.

---

## Skills Index

| File | Skill | Priority | Depends On |
|---|---|---|---|
| `01_barrier_labeling.md` | Barrier Labeling & ML Label Design | P1 | None |
| `02_walk_forward_validation.md` | Walk-Forward Validation & Leakage Prevention | P1 | 01 |
| `03_feature_engineering.md` | Feature Engineering (No Leakage) | P1 | None |
| `04_risk_governor.md` | Risk Governor | **P0** | None |
| `05_regime_detection.md` | Regime Detection | P2 | 03 |
| `06_backtesting_engine.md` | Backtesting Engine & Cost Model | P1 | 04 |
| `07_data_layer.md` | Data Layer, Validation & Corporate Actions | **P0** | None |
| `08_paper_broker_order_manager.md` | Paper Broker & Order Manager | P2 | 04 |
| `09_secrets_security.md` | Secrets Management & Security | **P0** | None |
| `10_position_sizing.md` | Position Sizing & Volatility Targeting | P1 | 05 |
| `11_model_calibration_drift.md` | ML Model Calibration & Drift Monitoring | P2 | 01, 02 |
| `12_swing_signal_generation.md` | Swing Signal Generation (Non-TA) | P1 | 03, 10 |

---

## Recommended Implementation Order

### Phase 0 — Safety Foundation (Must be first, must have 100% test pass)
```
07_data_layer.md          → Data validation, CSV loading, corporate actions
04_risk_governor.md       → All risk limits, kill-switch, PDT rule
09_secrets_security.md    → Environment variables, config validator
```
> Nothing else runs until these three pass all unit tests.

### Phase 1 — Core Logic
```
01_barrier_labeling.md        → Label creation for ML training
02_walk_forward_validation.md → Leak-free CV pipeline
03_feature_engineering.md     → Feature matrix builder
06_backtesting_engine.md      → Simulation engine
10_position_sizing.md         → Vol-targeting sizer
12_swing_signal_generation.md → Candidate generator
```

### Phase 2 — Intelligence & Execution
```
05_regime_detection.md             → Market regime classifier
08_paper_broker_order_manager.md   → Order routing & paper execution
11_model_calibration_drift.md      → Calibration + monitoring
```

### Phase 3 — Production Hardening
```
Run stress tests (GFC 2008, COVID 2020, 2022 rates)
Reconciliation job
Drift monitoring scheduler
Performance dashboards
```

---

## How to Give Skills to a Coding Assistant

### Option A: One module at a time
```
"Using the skill in [filename], implement the [module name] for our trading system.
The system spec is in AlgoTrading_RedTeam_Validation.docx.
Start with [specific class/function]. Follow the skill file exactly."
```

### Option B: Full system build
```
"Build the trading system using all skills in this directory.
Start with Phase 0 skills (07, 04, 09) and implement in the priority order in 00_index.md.
Do not move to Phase 1 until all Phase 0 unit tests pass."
```

### Option C: Targeted fix
```
"The [specific component] is broken. The correct implementation is in [skill file].
Here is the current broken code: [paste code]
Fix it to match the skill file exactly."
```

---

## Cross-Cutting Concerns

These rules apply EVERYWHERE, regardless of which skill you're implementing:

### Leakage Prevention
- Feature at time `t` uses only data from `t` or earlier
- Scalers/encoders fit inside walk-forward train folds only
- Embargo gap = label_horizon + 2 days minimum
- Run `leakage_audit()` before every training run

### Risk First
- Risk Governor is called before EVERY order
- No auto-liquidation of Core sleeve
- Kill-switch requires manual restart in production
- All risk parameters validated on startup

### Data Integrity
- Always use split/dividend-adjusted prices
- Validate data on load (no duplicates, no future dates, OHLCV consistency)
- Forward-fill max 1 day; never interpolate
- Hash registry detects silently corrupted files

### Security
- No credentials in any config file
- `.env` in `.gitignore`
- Config validator rejects dangerous parameter values
- Every order logged including rejections

---

## Configuration Template

All skills reference a shared YAML config. Use this structure:

```yaml
# config/example.yaml

system:
  mode: paper               # paper | live
  environment: staging      # dev | staging | prod
  log_level: INFO

data:
  data_dir: "data/ohlcv"
  universe_file: "data/universe.csv"
  corporate_actions_file: "data/corporate_actions.csv"
  max_ffill_days: 1
  validate_on_load: true
  timezone: "America/New_York"

portfolio:
  initial_nav: 100000
  benchmark_symbol: "SPY"
  sleeve_allocations:
    core: 0.60
    swing: 0.30
    cash_buffer: 0.10

risk:
  max_portfolio_drawdown: 0.15
  max_daily_loss_pct: 0.03
  kill_switch_cooldown_days: 5
  manual_restart_required: true
  swing:
    max_weekly_loss: 0.05
    max_concurrent_positions: 10
    max_position_pct: 0.15
    max_sector_pct: 0.30
    per_trade_stop_multiplier: 2.0
  pdt:
    enforce: true
    account_threshold: 25000

cost_model:
  commission_bps: 0.5
  spread_cost_bps: 1.0
  market_impact_factor: 0.1
  participation_rate: 0.05
  settlement_lag_days: 2

labeling:
  k1: 2.0
  k2: 1.0
  horizon_days: 10
  vol_window: 21
  embargo_days: 12

walk_forward:
  initial_train_days: 756
  test_days: 126
  step_days: 63
  expanding: true

features:
  return_windows: [5, 10, 21]
  vol_windows: [5, 21]
  index_symbol: "SPY"

ml:
  entry_threshold: 0.60
  calibration_method: "isotonic"
  drift_alert_psi: 0.20
  min_oos_auc_to_deploy: 0.55

swing_signals:
  momentum_signal_enabled: true
  momentum_threshold_pct: 0.04
  vol_expansion_signal_enabled: true
  vol_expansion_ratio: 1.5

sizing:
  holding_days: 10
  target_position_vol_pct: 0.005
  max_position_pct_swing: 0.15

regime_detection:
  method: "hmm"
  n_regimes: 4

execution:
  max_orders_per_minute: 20
  max_notional_per_minute: 500000
```

---

## Required Warnings

These must appear in the project README:

```
⚠️ RISK WARNING: Algorithmic trading involves substantial risk of total loss of capital.
This software is for educational and research purposes only. It does not constitute
financial advice. Backtests are subject to survivorship bias, look-ahead bias, and
underestimated transaction costs. Do not deploy with capital you cannot afford to
lose entirely. Monitor all live systems manually. Consult a licensed broker and
legal counsel before live deployment.
```

---

## Testing Requirements

Before any live deployment, ALL of these must pass:

```bash
pytest tests/test_risk_governor.py           # All risk limits enforced correctly
pytest tests/test_portfolio_accounting.py    # P&L calculation exact
pytest tests/test_labeler_no_leakage.py      # Barrier labels use no future data
pytest tests/test_walk_forward.py            # Embargo gaps correct
pytest tests/test_cost_model.py              # Fill prices, partial fills, fees
pytest tests/test_no_secrets_in_config.py    # No credentials in config files
```
