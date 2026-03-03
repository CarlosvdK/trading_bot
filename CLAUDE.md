# CLAUDE.md — Operating Rules for Trading Bot Repo

## Skill Usage Policy
- **Always** consult relevant `.claude/skills/**/SKILL.md` before writing code in that domain.
- Follow skill instructions exactly — they encode red-team-validated best practices.
- Use the skill index (`.claude/skills/index/SKILL.md`) to find the right skill for a task.
- Skill routing table:

| Keyword / Context | Skill to Consult |
|---|---|
| risk limits, drawdown, kill-switch, PDT, exposure | `risk-governor` |
| CSV, OHLCV, data loading, validation, corporate actions | `data-layer` |
| triple barrier, TP/SL labels, label leakage | `barrier-labeling` |
| walk-forward, train/test split, embargo, purging | `walk-forward-validation` |
| features, rolling returns, volume surprise, z-score | `feature-engineering` |
| regime, HMM, market state, vol regime | `regime-detection` |
| calibration, PSI, drift, Brier score, retrain | `model-calibration-drift` |
| position size, vol-target, Kelly, bet sizing | `position-sizing` |
| swing signal, momentum, breakout, candidate gen | `swing-signal-generation` |
| order manager, paper broker, fill sim, slippage | `paper-broker-order-manager` |
| API keys, secrets, .env, credentials, security | `secrets-security` |
| backtest, cost model, Sharpe, portfolio accounting | `backtesting-engine` |

## Plan Mode Default
- For any non-trivial task (3+ steps or architectural decisions), start with a **PLAN** section before writing code.
- The plan should list files to create/modify, key design decisions, and verification steps.

## Verification Before Done
- Never claim a task is done without running relevant tests or showing evidence of correctness.
- For new modules: write and run at least one unit test.
- For bug fixes: show the failing case passes after the fix.

## Demand Elegance
- If a fix or implementation feels hacky, propose a cleaner design before committing.
- Prefer composition over inheritance. Prefer pure functions over stateful classes where practical.
- Follow the spec's anti-patterns list (Section H11) strictly.

## Autonomous Bug Fixing
- When given failing tests or error logs, fix the root cause directly without asking basic questions.
- Only ask clarifying questions when the intent is genuinely ambiguous.

## Self-Improvement Loop
- After ANY user correction or prevented mistake, append a new rule to `tasks/lessons.md`.
- If the lesson is broadly applicable, also update this CLAUDE.md file.

## Project Structure
```
trading_bot/
├── .claude/skills/          # Domain skill files (read-only reference)
├── config/                  # YAML configs + schema
├── data/                    # CSV data, templates
├── src/                     # All source code
│   ├── core/               # Core sleeve rebalance
│   ├── swing/              # Swing signal generation
│   ├── scalp/              # Stub only
│   ├── risk/               # Risk governor (P0)
│   ├── execution/          # Order manager, brokers
│   ├── backtest/           # Backtester, cost model
│   ├── ml/                 # ML pipeline
│   ├── data/               # Data providers, validators
│   └── utils/              # Config, logging, secrets
├── tests/                   # All tests
├── scripts/                 # Entry points
├── tasks/                   # Workflow tracking
│   ├── todo.md
│   ├── lessons.md
│   └── notes/
├── CLAUDE.md                # This file
├── requirements.txt
├── Dockerfile
└── README.md
```

## Critical Anti-Patterns (from spec H11)
- **NO** future data in features — all features at time t use only data up to t.
- **NO** fitting preprocessors on full dataset — always fit inside walk-forward train fold only.
- **NO** auto-liquidation of Core to fund Swing — only cash buffer can fund Swing.
- **NO** credentials in config files or source code — environment variables only.
- **NO** trading during market halt or circuit breaker conditions.
- **NO** deployment without passing all unit tests.
- **NO** ignoring partial fills in backtest.

## Implementation Priority
1. **P0**: Risk Governor, Data Validator, Config Validator, Tests
2. **P1**: Backtest Engine, Cost Model, Core Rebalance, Barrier Labeler, ML Pipeline
3. **P2**: Paper Broker, Order Manager, Reconciliation, Regime Detection
4. **P3**: Drift Monitor, Alerting, Logging
5. **P4**: Vol Forecast, Scalp Stubs
