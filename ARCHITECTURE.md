# ARCHITECTURE.md — Multi-Sleeve Algorithmic Trading System

> Canonical blueprint. All implementations must conform to this document.
> Source of truth for interfaces and contracts: `.claude/skills/**/SKILL.md`

---

## 1. System Goals & Non-Goals

### Goals
- Execute a multi-sleeve equity/ETF portfolio with independent risk budgets
- Use ML-filtered signals (not raw TA) for Swing sleeve trade selection
- Enforce hard risk limits at every level — portfolio, sleeve, position, regulatory
- Produce backtests that are **harder to beat than live** (conservative cost model)
- Maintain full audit trail of every order, fill, and rejection
- Support paper and live execution via identical code paths

### Non-Goals
- High-frequency or sub-second execution (daily bar resolution)
- Options, futures, or crypto instruments
- Fully autonomous live trading without human oversight
- Real-time streaming data (batch daily updates)
- Scalp sleeve implementation (stub only in v1)

### Safety-First Principles
1. **Risk Governor gates every order** — no bypass, no exceptions
2. **Kill-switch requires manual restart** in production
3. **No future data** in features, labels, or regime detection
4. **Secrets never in config or source** — environment variables only
5. **Every design choice makes backtests harder to beat**, not easier

---

## 2. System Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE                           │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Config   │  │   Secrets    │  │   Regime Detector     │ │
│  │  Loader   │  │   Manager    │  │   (HMM / KMeans)      │ │
│  └────┬─────┘  └──────┬───────┘  └──────────┬────────────┘ │
│       │               │                      │              │
│       └───────────────┼──────────────────────┘              │
│                       ▼                                     │
│              ┌─────────────────┐                            │
│              │  RISK GOVERNOR  │ ◄── PortfolioState         │
│              │  (P0 — gates    │                            │
│              │   ALL orders)   │                            │
│              └────────┬────────┘                            │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                  SIGNAL PLANE                               │
│                       │                                     │
│  ┌────────────────────┼────────────────────────┐            │
│  │                    │                        │            │
│  ▼                    ▼                        ▼            │
│ ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│ │  CORE    │  │    SWING     │  │  SCALP (stub)         │  │
│ │  Sleeve  │  │    Sleeve    │  │                       │  │
│ │          │  │              │  │                       │  │
│ │ Target-  │  │ Candidate    │  │  Not implemented      │  │
│ │ weight   │  │ Generator    │  │  in v1                │  │
│ │ rebalance│  │     ↓        │  │                       │  │
│ │          │  │ ML Filter    │  │                       │  │
│ │          │  │     ↓        │  │                       │  │
│ │          │  │ Vol-Target   │  │                       │  │
│ │          │  │ Sizer        │  │                       │  │
│ └────┬─────┘  └──────┬───────┘  └───────────────────────┘  │
│      │               │                                      │
└──────┼───────────────┼──────────────────────────────────────┘
       │               │
       └───────┬───────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│                  EXECUTION PLANE                             │
│                                                              │
│  ┌─────────────────┐    ┌────────────────┬────────────────┐ │
│  │  Order Manager   │───►│  Paper Broker  │  Live Broker   │ │
│  │  (rate limit,    │    │  (default)     │  (stub/IBKR)   │ │
│  │   logging)       │    └────────────────┴────────────────┘ │
│  └─────────────────┘                                         │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐    ┌────────────────┐                  │
│  │  Order Log       │    │  Settlement    │                  │
│  │  (SQLite)        │    │  Queue (T+2)   │                  │
│  └─────────────────┘    └────────────────┘                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    DATA PLANE                                │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ DataProvider  │  │  Validator   │  │  Corporate Actions │ │
│  │ (ABC)         │  │  (OHLCV)    │  │  Handler           │ │
│  │   └─ CSV      │  └──────────────┘  └────────────────────┘ │
│  │   └─ API*     │                                           │
│  └──────────────┘  ┌──────────────┐  ┌────────────────────┐ │
│                     │  Hash        │  │  Universe File     │ │
│                     │  Registry    │  │  (survivorship)    │ │
│                     └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      ML PLANE                                │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Barrier   │  │ Feature  │  │ Walk-Fwd │  │ Calibrator │  │
│  │ Labeler   │  │ Builder  │  │ Pipeline │  │ (isotonic) │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│                                                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐ │
│  │ PSI Drift Monitor        │  │ Retrain Trigger Logic    │ │
│  └──────────────────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 Data Layer (P0)
**Skill**: `.claude/skills/data-layer/SKILL.md`
**Source**: `src/data/`

| Component | File | Description |
|---|---|---|
| `DataProvider` | `src/data/provider.py` | ABC with `load_symbol()`, `available_symbols()`, `get_universe()` |
| `CSVDataProvider` | `src/data/csv_provider.py` | CSV implementation with caching, hash registry |
| `validate_ohlcv()` | `src/data/validator.py` | 9-point validation: columns, monotonic dates, no future dates, OHLCV consistency, zero prices, volume, gaps, extreme moves, staleness |
| `apply_corporate_actions()` | `src/data/corporate_actions.py` | Split/reverse-split price adjustment |
| `handle_missing_data()` | `src/data/missing.py` | Forward-fill max 1 day, never interpolate |

**Interface contract**:
```python
class DataProvider(ABC):
    def load_symbol(self, symbol: str, start_date=None, end_date=None) -> pd.DataFrame:
        """Returns OHLCV DataFrame, DatetimeIndex, split/div adjusted."""
    def available_symbols(self) -> List[str]: ...
    def get_universe(self, date: str) -> List[str]:
        """Point-in-time universe — survivorship bias control."""
```

### 3.2 Risk Governor (P0)
**Skill**: `.claude/skills/risk-governor/SKILL.md`
**Source**: `src/risk/`

| Component | File | Description |
|---|---|---|
| `RiskGovernor` | `src/risk/risk_governor.py` | Central risk enforcement — pre-trade + periodic checks |
| `RiskConfig` | `src/risk/risk_governor.py` | Dataclass with all risk parameters and safe defaults |
| `PortfolioState` | `src/risk/risk_governor.py` | Mutable state: NAV, peak NAV, positions, sleeve values |

**Risk limits hierarchy** (6 levels):
1. Portfolio kill-switch: -15% peak-to-trough
2. Daily loss halt: -3% of NAV
3. Swing weekly loss: -5% of swing sleeve
4. Per-position: 15% of sleeve max, 30% sector cap
5. Gross exposure: 100% of NAV
6. PDT: 3 round-trips / 5 business days (if account < $25k)

**Interface contract**:
```python
class RiskGovernor:
    def pre_trade_check(self, symbol, side, notional, sleeve, state, ...) -> Tuple[bool, str]:
        """Returns (allowed, reason). Called before EVERY order."""
    def periodic_check(self, state, current_date) -> List[str]:
        """Returns list of triggered alerts. Called every 15 min (swing) / EOD (core)."""
    def manual_reset_kill_switch(self, operator_id: str): ...
```

### 3.3 Secrets & Security (P0)
**Skill**: `.claude/skills/secrets-security/SKILL.md`
**Source**: `src/utils/`

| Component | File | Description |
|---|---|---|
| `get_secret()` | `src/utils/secrets.py` | Env-var-only secret loading, never reads config files |
| `load_config()` | `src/utils/config_loader.py` | YAML loading + risk param range validation + secret scan |
| `AuditLogger` | `src/utils/audit.py` | Append-only log with SHA-256 hash chaining |
| HMAC signing | `src/utils/webhooks.py` | `sign_webhook_payload()` / `verify_webhook_signature()` |

**Interface contract**:
```python
def get_secret(key: str, required: bool = True) -> Optional[str]:
    """Load from env vars only. Raises ValueError if required and missing."""

def load_config(path: str) -> dict:
    """Load YAML, validate no secrets in values, validate risk param ranges."""
```

### 3.4 Barrier Labeling (P1)
**Skill**: `.claude/skills/barrier-labeling/SKILL.md`
**Source**: `src/ml/`

| Component | File | Description |
|---|---|---|
| `barrier_label()` | `src/ml/labeler.py` | Single-sample triple-barrier label (binary: 1=TP, 0=SL/timeout) |
| `build_labels()` | `src/ml/labeler.py` | Vectorized label builder for all signal dates |
| `purge_and_embargo()` | `src/ml/labeler.py` | Remove training samples whose label window overlaps test period |
| `compute_vol_proxy()` | `src/ml/labeler.py` | Rolling 21-day annualized vol from log returns (no TA libraries) |

**Interface contract**:
```python
def barrier_label(prices, entry_idx, tp_pct, sl_pct, horizon=10) -> int:
    """Returns 1 (TP hit) or 0 (SL/timeout). Uses only prices[entry_idx:entry_idx+horizon+1]."""

def build_labels(df, signal_dates, k1=2.0, k2=1.0, vol_window=21, horizon=10) -> pd.DataFrame:
    """Returns DataFrame with entry_date index, columns: label, tp_pct, sl_pct, vol_at_entry."""
```

### 3.5 Walk-Forward Validation (P1)
**Skill**: `.claude/skills/walk-forward-validation/SKILL.md`
**Source**: `src/ml/`

| Component | File | Description |
|---|---|---|
| `walk_forward_splits()` | `src/ml/validation.py` | Yields (train_idx, test_idx) with embargo gaps |
| `purge_training_labels()` | `src/ml/validation.py` | Remove samples with forward-looking labels |
| `run_walk_forward()` | `src/ml/validation.py` | Full pipeline: split, train, calibrate, predict, metrics |
| `leakage_audit()` | `src/ml/validation.py` | Spearman correlation with future returns per feature |

**Interface contract**:
```python
def walk_forward_splits(index, initial_train_days=756, test_days=126,
                        step_days=63, embargo_days=12, expanding=True
                        ) -> Iterator[Tuple[DatetimeIndex, DatetimeIndex]]: ...

def leakage_audit(features, prices, max_allowed_corr=0.05) -> dict:
    """Returns {'issues': [...], 'passed': bool}. Run before every training run."""
```

### 3.6 Feature Engineering (P1)
**Skill**: `.claude/skills/feature-engineering/SKILL.md`
**Source**: `src/ml/`

| Component | File | Description |
|---|---|---|
| `build_features()` | `src/ml/features.py` | Full feature matrix: returns, vol, gap, volume surprise, index context, relative return |
| `winsorize_zscore()` | `src/ml/features.py` | Rolling z-score normalization (backward-looking only) |
| `check_feature_collinearity()` | `src/ml/features.py` | VIF check for multicollinearity |

**Feature set** (all backward-looking):
1. Rolling returns: 5d, 10d, 21d
2. Rolling vol: 5d, 21d (annualized)
3. Vol ratio: 5d/21d
4. Vol-of-vol proxy
5. Gap return (open vs prev close)
6. Volume surprise (vs 21d avg)
7. Momentum consistency (up-day fraction, 10d)
8. Index context (SPY returns + vol, 5d/21d)
9. Relative return (symbol - index, 5d)

### 3.7 Backtesting Engine (P1)
**Skill**: `.claude/skills/backtesting-engine/SKILL.md`
**Source**: `src/backtest/`

| Component | File | Description |
|---|---|---|
| `CostModel` | `src/backtest/cost_model.py` | Commission, spread, market impact (sqrt model), partial fills, T+2 settlement |
| `Position` | `src/backtest/portfolio.py` | Symbol, qty, avg_cost, stop/target prices |
| `SleeveAccount` | `src/backtest/portfolio.py` | Per-sleeve cash, positions, realized P&L, trade log |
| `Backtester` | `src/backtest/engine.py` | Daily loop: settlement → retrain → signals → execute → barriers → MTM |

**Non-negotiable invariants**:
- Fills at **next bar open** (not signal bar close)
- T+2 settlement: proceeds unavailable for 2 business days
- Partial fills: orders > 1% ADV get partial treatment
- Fill price bounded by bar high/low

**Interface contract**:
```python
class Backtester:
    def run(self, signal_func, ml_retrain_func=None, wf_boundaries=None) -> dict:
        """Returns {total_return, sharpe, max_drawdown, calmar, trades, nav_history}."""

class CostModel:
    def fill_price(self, bar_open, side, notional, adv, bar_high, bar_low) -> float: ...
    def partial_fill_qty(self, order_qty, adv, fill_price, participation_rate=0.05) -> float: ...
    def total_roundtrip_bps(self, notional, adv, holding_days=10, is_short=False) -> float: ...
```

### 3.8 Position Sizing (P1)
**Skill**: `.claude/skills/position-sizing/SKILL.md`
**Source**: `src/swing/`

| Component | File | Description |
|---|---|---|
| `vol_target_size()` | `src/swing/sizing.py` | Base notional from vol budget per position |
| `ml_probability_size_scale()` | `src/swing/sizing.py` | Linear scale 0.5x–1.5x by ML confidence |
| `regime_adjusted_size()` | `src/swing/sizing.py` | Multiply by regime + VVol factor |
| `compute_swing_position_size()` | `src/swing/sizing.py` | Full pipeline: vol → ML scale → regime → shares |
| `compute_barriers()` | `src/swing/sizing.py` | TP/SL prices from vol * k1/k2 |
| `core_rebalance_orders()` | `src/core/rebalance.py` | Drift-band rebalance for Core sleeve |

**Core formula**:
```
target_notional = (sleeve_nav * target_vol_pct) / (instrument_vol * sqrt(holding_days / 252))
```

### 3.9 Swing Signal Generation (P1)
**Skill**: `.claude/skills/swing-signal-generation/SKILL.md`
**Source**: `src/swing/`

| Component | File | Description |
|---|---|---|
| `momentum_breakout_candidates()` | `src/swing/signals.py` | 5d return > 4%, today up, volume > 1.2x avg |
| `volatility_expansion_candidates()` | `src/swing/signals.py` | Short vol / long vol > 1.5 |
| `is_risk_on()` | `src/swing/signals.py` | Index gate: 5d ret > -1%, vol ratio < 1.5 |
| `apply_ml_filter()` | `src/swing/signals.py` | Filter candidates by calibrated P(favorable) >= 0.60 |
| `generate_swing_signals()` | `src/swing/signals.py` | Full pipeline: gate → candidates → ML → size → orders |

**Design principle**: Signals are broad (high recall). The ML filter is the edge.

### 3.10 Regime Detection (P2)
**Skill**: `.claude/skills/regime-detection/SKILL.md`
**Source**: `src/ml/`

| Component | File | Description |
|---|---|---|
| `build_regime_features()` | `src/ml/regime.py` | 9 features from index: returns, vol, vol ratio, VVol, trend strength, drawdown |
| `fit_regime_model()` | `src/ml/regime.py` | HMM (preferred) or KMeans fallback |
| `predict_regime()` | `src/ml/regime.py` | Predict regime labels for new data |
| `label_regimes()` | `src/ml/regime.py` | Post-hoc naming: `{vol_level}_{trend_type}` |
| `smooth_regime()` | `src/ml/regime.py` | Persistence filter: N consecutive days before switch |
| `REGIME_ALLOCATION` | `src/ml/regime.py` | Mapping: regime → swing_multiplier + swing_enabled |

### 3.11 Paper Broker & Order Manager (P2)
**Skill**: `.claude/skills/paper-broker-order-manager/SKILL.md`
**Source**: `src/execution/`

| Component | File | Description |
|---|---|---|
| `Order` / `Fill` | `src/execution/types.py` | Dataclasses shared by paper and live paths |
| `OrderManager` | `src/execution/order_manager.py` | Rate limiting → Risk Governor → broker routing → logging |
| `PaperBroker` | `src/execution/paper_broker.py` | Fill simulation: next-bar open, slippage, partial fills |
| `LiveBrokerStub` | `src/execution/live_broker.py` | Raises `NotImplementedError` — safety gate |
| `GracefulShutdown` | `src/execution/shutdown.py` | SIGTERM/SIGINT handler, state persistence |

**Interface contract**:
```python
class OrderManager:
    def submit(self, order: Order, current_date) -> Optional[Fill]:
        """Rate limit → Risk Governor → Broker → Log. Returns Fill or None."""

class PaperBroker:
    def execute(self, order: Order, current_date) -> Optional[Fill]:
        """Fill at next bar open (MARKET) or close (MOC). Same interface as live."""
```

### 3.12 Model Calibration & Drift (P2)
**Skill**: `.claude/skills/model-calibration-drift/SKILL.md`
**Source**: `src/ml/`

| Component | File | Description |
|---|---|---|
| `calibrate_model()` | `src/ml/calibration.py` | Post-hoc isotonic/Platt calibration on held-out fold |
| `reliability_diagram()` | `src/ml/calibration.py` | ECE computation + calibration plot |
| `compute_psi()` | `src/ml/drift.py` | Population Stability Index per feature |
| `monitor_feature_drift()` | `src/ml/drift.py` | Full drift report: STABLE / MONITOR / RETRAIN_NOW |
| `compute_live_metrics()` | `src/ml/drift.py` | AUC, Brier, F1, win rate on recent predictions |
| `should_retrain()` | `src/ml/drift.py` | Decision logic: time + drift + performance |
| `save_model()` / `load_model_with_meta()` | `src/ml/persistence.py` | `.pkl` + `.json` sidecar with training metadata |

**Retrain triggers**:
- Scheduled: > 90 days since last train
- Drift: any feature PSI > 0.25
- Performance: AUC < 0.53 (urgent) or < 0.50 (disable strategy)

---

## 4. Data Flow

```
                      ┌──────────────┐
                      │  CSV Files   │
                      │  (OHLCV)     │
                      └──────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  DataProvider   │──► validate_ohlcv()
                    │  + corp actions│──► handle_missing_data()
                    └────────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌──────────────┐
     │  Feature   │  │  Barrier   │  │  Regime      │
     │  Builder   │  │  Labeler   │  │  Detector    │
     └─────┬──────┘  └─────┬──────┘  └──────┬───────┘
           │               │                 │
           └───────┬───────┘                 │
                   ▼                         │
          ┌────────────────┐                 │
          │  Walk-Forward  │                 │
          │  Pipeline      │                 │
          │  (train+cal)   │                 │
          └────────┬───────┘                 │
                   ▼                         │
          ┌────────────────┐                 │
          │  Calibrated    │◄────────────────┘
          │  ML Model      │    (regime feeds sizing)
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │  Signal Gen    │──► candidates + ML filter
          │  + Sizer       │──► vol-target + regime adjust
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │  Risk Governor │──► pre_trade_check()
          └────────┬───────┘
                   │
                   ▼
          ┌────────────────┐
          │  Order Manager │──► Paper/Live Broker
          │  → Fill → Log  │
          └────────────────┘
```

---

## 5. Control Flow (Daily Loop)

```
for each trading_day:

  1. SETTLEMENT
     └─ Process T+2 settlement queue → release cash to sleeves

  2. DATA UPDATE
     └─ Load today's OHLCV bar for all symbols
     └─ validate_ohlcv() on new data

  3. REGIME CHECK
     └─ predict_regime(today) → regime_name
     └─ smooth_regime() → accept or hold previous
     └─ get_regime_allocation() → swing_multiplier, swing_enabled

  4. RISK CHECK (periodic)
     └─ risk_governor.periodic_check(state)
     └─ Update peak_nav, check drawdown, check daily loss
     └─ If kill-switch triggered → halt all trading

  5. CORE SLEEVE (if EOD / rebalance day)
     └─ Compute current vs target weights
     └─ If drift > 5% → generate MOC rebalance orders
     └─ risk_governor.pre_trade_check() each order
     └─ order_manager.submit() → fill at close

  6. SWING SLEEVE (if swing_enabled)
     └─ is_risk_on(index) → gate check
     └─ momentum_breakout_candidates()
     └─ volatility_expansion_candidates()
     └─ Deduplicate by symbol
     └─ apply_ml_filter(candidates, threshold=0.60)
     └─ compute_swing_position_size() for each
     └─ risk_governor.pre_trade_check() each order
     └─ order_manager.submit() → fill at next open

  7. BARRIER CHECK
     └─ For each open Swing position:
        └─ If price <= stop_price → close (stop hit)
        └─ If price >= target_price → close (target hit)
        └─ If holding_days > horizon → close (timeout)

  8. MARK TO MARKET
     └─ Update NAV for all sleeves
     └─ Record to nav_history
     └─ Log portfolio state

  9. ML RETRAIN (if walk-forward boundary)
     └─ purge_training_labels()
     └─ leakage_audit()
     └─ Train new model on expanded window
     └─ Calibrate on held-out 20%
     └─ If OOS AUC >= 0.55 → deploy
```

---

## 6. State & Storage

| State | Location | Persistence |
|---|---|---|
| OHLCV data | `data/ohlcv/*.csv` | Files on disk |
| Universe file | `data/universe.csv` | File on disk |
| Corporate actions | `data/corporate_actions.csv` | File on disk |
| Data hash registry | `data/ohlcv/.data_hashes.json` | Auto-generated |
| YAML config | `config/*.yaml` | File on disk |
| Portfolio state | In-memory `PortfolioState` | Serialized on shutdown |
| Order log | SQLite or in-memory list | Append-only |
| Audit log | `logs/audit.jsonl` | Append-only, hash-chained |
| ML models | `models/*.pkl` + `*.json` sidecar | Versioned by train date |
| NAV history | DataFrame → `results/nav_history.csv` | Per backtest run |
| Trade log | DataFrame → `results/trades.csv` | Per backtest run |

---

## 7. Configuration

All modules share a single YAML config. Full template in `.claude/skills/index/SKILL.md`.

**Key sections**:
```yaml
system:     # mode (paper/live), environment, log_level
data:       # data_dir, universe_file, max_ffill_days, validate_on_load
portfolio:  # initial_nav, benchmark, sleeve_allocations (core: 0.60, swing: 0.30, cash: 0.10)
risk:       # drawdown limits, daily loss, swing limits, exposure, PDT
cost_model: # commission, spread, market impact, participation rate, T+2
labeling:   # k1, k2, horizon_days, vol_window, embargo_days
walk_forward: # train/test/step days, expanding window
features:   # return/vol windows, index symbol
ml:         # entry threshold, calibration method, drift thresholds
swing_signals: # momentum/vol expansion params, risk-on gate
sizing:     # holding_days, target_vol_pct, max_position_pct
regime_detection: # method, n_regimes, persistence days
execution:  # rate limits, order routing
```

**Validation**: `load_config()` enforces safe ranges via `RISK_PARAM_RANGES` dict. Any value outside bounds → startup failure.

---

## 8. Testing Strategy

### Required test suites (must pass before any deployment)

| Test File | What It Validates |
|---|---|
| `tests/test_risk_governor.py` | All 6 risk levels, kill-switch, PDT, manual reset |
| `tests/test_portfolio_accounting.py` | Position P&L, sleeve NAV, realized vs unrealized |
| `tests/test_labeler_no_leakage.py` | Labels use only `prices[entry:entry+horizon]`, no future data |
| `tests/test_walk_forward.py` | Embargo gaps correct, purging removes contaminated samples |
| `tests/test_cost_model.py` | Fill price bounded by high/low, partial fills, fees, T+2 |
| `tests/test_no_secrets_in_config.py` | No secret-looking values in YAML files, .env not in git |
| `tests/test_data_validator.py` | All 9 validation checks trigger correctly |
| `tests/test_feature_no_leakage.py` | Features at time t use only data <= t |

### Testing principles
- **P0 modules must have 100% test pass** before any P1 work begins
- Walk-forward tests must verify embargo gap = `label_horizon + 2` days minimum
- Backtest tests must verify fills at next-bar open, never signal-bar close
- Risk Governor tests must cover every rejection path with explicit assertions

---

## 9. Failure Modes & Kill-Switch

| Failure | Detection | Response |
|---|---|---|
| Portfolio drawdown > 15% | `periodic_check()` | **Kill-switch**: halt ALL orders, require manual restart |
| Daily loss > 3% | `periodic_check()` | Halt new orders for remainder of day |
| Swing weekly loss > 5% | `pre_trade_check()` | Halt swing sleeve until end of week |
| ML model AUC < 0.50 | `compute_live_metrics()` | **Disable Swing** immediately |
| ML model AUC < 0.53 | `compute_live_metrics()` | Retrain urgently |
| Feature PSI > 0.25 | `monitor_feature_drift()` | Retrain immediately |
| Data file hash changed | `CSVDataProvider._check_hash()` | Warning + re-hash |
| Data > 3 days stale | `validate_ohlcv()` | Warning in validation report |
| Broker execution error | `OrderManager.submit()` | Log rejection, skip order |
| SIGTERM/SIGINT | `GracefulShutdown` | Finish current cycle, save state, exit cleanly |

**Kill-switch properties**:
- Triggered automatically by `RiskGovernor._trigger_kill_switch()`
- `manual_restart_required: true` in production (default)
- Cooldown: 5 business days if manual restart not required
- Logged to audit trail with timestamp and reason

---

## 10. Extensibility

| Extension Point | Mechanism |
|---|---|
| New data source (API, DB) | Implement `DataProvider` ABC |
| New broker | Implement `execute(order, date) -> Fill` matching `PaperBroker` interface |
| New signal type | Add function returning `List[dict]` with `symbol, signal_type, direction, signal_date` |
| New features | Add to `build_features()`, run `leakage_audit()`, check VIF |
| New regime model | Implement `fit_regime_model()` + `predict_regime()` contract |
| New cost components | Extend `CostModel` dataclass fields + `total_roundtrip_bps()` |
| Scalp sleeve | Implement under `src/scalp/` following same Risk Governor integration pattern |

---

## 11. Implementation Order & Gates

### Phase 0 — Safety Foundation
```
07 Data Layer          → src/data/
04 Risk Governor       → src/risk/
09 Secrets & Security  → src/utils/
```
**Gate**: ALL P0 unit tests pass. No exceptions.

### Phase 1 — Core Logic
```
01 Barrier Labeling         → src/ml/labeler.py
02 Walk-Forward Validation  → src/ml/validation.py
03 Feature Engineering      → src/ml/features.py
06 Backtesting Engine       → src/backtest/
10 Position Sizing          → src/swing/sizing.py
12 Swing Signal Generation  → src/swing/signals.py
```
**Gate**: Backtest runs end-to-end on sample data. leakage_audit() passes. OOS AUC >= 0.55.

### Phase 2 — Intelligence & Execution
```
05 Regime Detection                → src/ml/regime.py
08 Paper Broker & Order Manager    → src/execution/
11 Model Calibration & Drift       → src/ml/calibration.py, drift.py
```
**Gate**: Paper trading produces realistic fills. Drift monitor runs without errors.

### Phase 3 — Production Hardening
```
Stress tests: GFC 2008, COVID 2020, 2022 rates
Reconciliation job
Drift monitoring scheduler (weekly)
Performance dashboards
```
**Gate**: System survives all stress scenarios without kill-switch bypass.

---

## 12. TODO — Missing Files Required by This Blueprint

The following files/directories do not yet exist and must be created during implementation:

### Source code
- [ ] `src/__init__.py`
- [ ] `src/data/__init__.py`
- [ ] `src/data/provider.py` — `DataProvider` ABC + `CSVDataProvider`
- [ ] `src/data/validator.py` — `validate_ohlcv()`, `validate_all_symbols()`
- [ ] `src/data/corporate_actions.py` — `apply_corporate_actions()`
- [ ] `src/data/missing.py` — `handle_missing_data()`
- [ ] `src/risk/__init__.py`
- [ ] `src/risk/risk_governor.py` — `RiskGovernor`, `RiskConfig`, `PortfolioState`
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/secrets.py` — `get_secret()`, convenience accessors
- [ ] `src/utils/config_loader.py` — `load_config()`, validators
- [ ] `src/utils/audit.py` — `AuditLogger`
- [ ] `src/utils/webhooks.py` — HMAC signing/verification
- [ ] `src/ml/__init__.py`
- [ ] `src/ml/labeler.py` — `barrier_label()`, `build_labels()`, `purge_and_embargo()`
- [ ] `src/ml/validation.py` — `walk_forward_splits()`, `run_walk_forward()`, `leakage_audit()`
- [ ] `src/ml/features.py` — `build_features()`, `winsorize_zscore()`, `check_feature_collinearity()`
- [ ] `src/ml/regime.py` — regime detection pipeline
- [ ] `src/ml/calibration.py` — `calibrate_model()`, `reliability_diagram()`
- [ ] `src/ml/drift.py` — `compute_psi()`, `monitor_feature_drift()`, `should_retrain()`
- [ ] `src/ml/persistence.py` — `save_model()`, `load_model_with_meta()`
- [ ] `src/backtest/__init__.py`
- [ ] `src/backtest/cost_model.py` — `CostModel`
- [ ] `src/backtest/portfolio.py` — `Position`, `SleeveAccount`
- [ ] `src/backtest/engine.py` — `Backtester`
- [ ] `src/swing/__init__.py`
- [ ] `src/swing/sizing.py` — vol-target sizing pipeline
- [ ] `src/swing/signals.py` — signal generators + ML filter integration
- [ ] `src/core/__init__.py`
- [ ] `src/core/rebalance.py` — `core_rebalance_orders()`
- [ ] `src/scalp/__init__.py` — stub only
- [ ] `src/execution/__init__.py`
- [ ] `src/execution/types.py` — `Order`, `Fill`, enums
- [ ] `src/execution/order_manager.py` — `OrderManager`
- [ ] `src/execution/paper_broker.py` — `PaperBroker`
- [ ] `src/execution/live_broker.py` — `LiveBrokerStub`
- [ ] `src/execution/shutdown.py` — `GracefulShutdown`

### Config
- [ ] `config/example.yaml` — full config template (from index skill)
- [ ] `config/schema.py` — optional config schema validation

### Data templates
- [ ] `data/universe.csv` — point-in-time universe template
- [ ] `data/corporate_actions.csv` — corporate actions template
- [ ] `data/ohlcv/.gitkeep`

### Tests
- [ ] `tests/__init__.py`
- [ ] `tests/test_risk_governor.py`
- [ ] `tests/test_portfolio_accounting.py`
- [ ] `tests/test_labeler_no_leakage.py`
- [ ] `tests/test_walk_forward.py`
- [ ] `tests/test_cost_model.py`
- [ ] `tests/test_no_secrets_in_config.py`
- [ ] `tests/test_data_validator.py`
- [ ] `tests/test_feature_no_leakage.py`

### Infrastructure
- [ ] `requirements.txt` — pinned dependencies
- [ ] `scripts/run_backtest.py` — CLI entry point
- [ ] `scripts/run_paper_trading.py` — paper trading entry point
- [ ] `models/.gitkeep` — ML model storage
- [ ] `results/.gitkeep` — backtest output storage
- [ ] `logs/.gitkeep` — audit and application logs
- [ ] `Dockerfile` — containerized deployment
