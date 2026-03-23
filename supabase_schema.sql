
-- ============================================================
-- AGENT SCORES: Track each agent's performance over time
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_scores (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    hit_rate DOUBLE PRECISION DEFAULT 0,
    avg_return DOUBLE PRECISION DEFAULT 0,
    sharpe DOUBLE PRECISION DEFAULT 0,
    independence_score DOUBLE PRECISION DEFAULT 0,
    composite_weight DOUBLE PRECISION DEFAULT 1.0,
    total_picks INTEGER DEFAULT 0,
    profitable_picks INTEGER DEFAULT 0,
    losing_picks INTEGER DEFAULT 0,
    calibration_quality DOUBLE PRECISION DEFAULT 0.5,
    uniqueness DOUBLE PRECISION DEFAULT 0.5,
    regime_effectiveness DOUBLE PRECISION DEFAULT 0.5,
    drawdown_behavior DOUBLE PRECISION DEFAULT 0.5,
    stability DOUBLE PRECISION DEFAULT 0.5,
    false_positive_rate DOUBLE PRECISION DEFAULT 0.3,
    status TEXT DEFAULT 'healthy',
    UNIQUE(agent_id, timestamp)
);

-- ============================================================
-- TRADE HISTORY: Every trade pick, vote, and outcome
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_history (
    id BIGSERIAL PRIMARY KEY,
    trade_id TEXT UNIQUE NOT NULL,
    agent_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    strategy_used TEXT,
    reasoning TEXT,
    suggested_hold_days INTEGER,
    proposed_at TIMESTAMPTZ DEFAULT NOW(),
    -- Voting
    approval_pct DOUBLE PRECISION,
    num_voters INTEGER,
    vote_result TEXT,  -- approved, rejected
    supporting_agents TEXT[],
    dissenting_agents TEXT[],
    -- Execution
    executed BOOLEAN DEFAULT FALSE,
    entry_price DOUBLE PRECISION,
    entry_date TIMESTAMPTZ,
    exit_price DOUBLE PRECISION,
    exit_date TIMESTAMPTZ,
    quantity INTEGER,
    fees DOUBLE PRECISION DEFAULT 0,
    slippage DOUBLE PRECISION DEFAULT 0,
    -- Outcome
    actual_return DOUBLE PRECISION,
    actual_hold_days INTEGER,
    outcome TEXT,  -- profitable, losing, open, cancelled
    pnl DOUBLE PRECISION,
    -- Context
    regime_at_entry TEXT,
    sector TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- PIPELINE RUNS: Each daily scan cycle
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    scan_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',
    universe_scanned INTEGER DEFAULT 0,
    proposals_generated INTEGER DEFAULT 0,
    specialist_reviewed INTEGER DEFAULT 0,
    voted_on INTEGER DEFAULT 0,
    approved INTEGER DEFAULT 0,
    rejected INTEGER DEFAULT 0,
    regime TEXT,
    notes TEXT
);

-- ============================================================
-- AGENT LEARNING: What each agent learned from outcomes
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_learning (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    trade_id TEXT REFERENCES trade_history(trade_id),
    lesson_type TEXT NOT NULL,  -- correct_call, false_positive, false_negative, regime_miss
    symbol TEXT,
    strategy_used TEXT,
    confidence_at_pick DOUBLE PRECISION,
    actual_outcome DOUBLE PRECISION,
    lesson TEXT,
    weight_adjustment DOUBLE PRECISION DEFAULT 0,
    learned_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- PORTFOLIO SNAPSHOTS: Daily portfolio state
-- ============================================================
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL UNIQUE,
    total_value DOUBLE PRECISION,
    cash_balance DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    daily_pnl DOUBLE PRECISION,
    gross_exposure DOUBLE PRECISION,
    net_exposure DOUBLE PRECISION,
    open_positions INTEGER,
    drawdown DOUBLE PRECISION,
    regime TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- RISK EVENTS: Kill switch activations, limit breaches
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT,
    details JSONB,
    agent_id TEXT,
    symbol TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- AGENT ACTIVITY: Real-time feed of what agents are doing
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_activity (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,  -- news_scan, analysis, thesis, proposal, vote, execution, monitoring, learning, retraining, playbook, regime
    symbol TEXT,
    summary TEXT NOT NULL,
    details JSONB,
    market_mode TEXT,  -- weekend, premarket, market, overnight
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_agent_scores_agent ON agent_scores(agent_id);
CREATE INDEX IF NOT EXISTS idx_trade_history_agent ON trade_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_outcome ON trade_history(outcome);
CREATE INDEX IF NOT EXISTS idx_agent_learning_agent ON agent_learning(agent_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type);
CREATE INDEX IF NOT EXISTS idx_agent_activity_type ON agent_activity(activity_type);
CREATE INDEX IF NOT EXISTS idx_agent_activity_created ON agent_activity(created_at DESC);
