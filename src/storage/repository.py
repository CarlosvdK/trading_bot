"""
TradingRepository — single interface for all persistent storage operations.

Handles agent scores, trade history, learning, portfolio snapshots,
pipeline runs, and risk events via Supabase.
"""

import logging
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TradingRepository:
    """Persistent storage for the entire trading system."""

    def __init__(self, supabase_client):
        self.sb = supabase_client

    # ==============================================================
    # AGENT SCORES
    # ==============================================================

    def save_agent_score(self, agent_id: str, metrics: Dict[str, Any]) -> None:
        """Save a snapshot of an agent's current score metrics."""
        row = {"agent_id": agent_id, **metrics}
        try:
            self.sb.table("agent_scores").insert(row).execute()
        except Exception as e:
            logger.error(f"Failed to save agent score for {agent_id}: {e}")

    def get_latest_agent_scores(self) -> List[Dict[str, Any]]:
        """Get the most recent score for every agent."""
        try:
            result = self.sb.table("agent_scores") \
                .select("*") \
                .order("timestamp", desc=True) \
                .execute()
            # Deduplicate — keep only latest per agent
            seen = set()
            latest = []
            for row in result.data:
                if row["agent_id"] not in seen:
                    seen.add(row["agent_id"])
                    latest.append(row)
            return latest
        except Exception as e:
            logger.error(f"Failed to get agent scores: {e}")
            return []

    def get_agent_score_history(
        self, agent_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get score history for a specific agent."""
        try:
            result = self.sb.table("agent_scores") \
                .select("*") \
                .eq("agent_id", agent_id) \
                .order("timestamp", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get score history for {agent_id}: {e}")
            return []

    def bulk_save_agent_scores(self, scores: List[Dict[str, Any]]) -> None:
        """Save scores for all agents at once (after a scoring cycle)."""
        if not scores:
            return
        try:
            self.sb.table("agent_scores").insert(scores).execute()
            logger.info(f"Saved scores for {len(scores)} agents")
        except Exception as e:
            logger.error(f"Failed to bulk save agent scores: {e}")

    # ==============================================================
    # TRADE HISTORY
    # ==============================================================

    def record_proposal(
        self,
        agent_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        strategy: str,
        reasoning: str,
        hold_days: int,
        sector: str = "",
        regime: str = "",
    ) -> str:
        """Record a new trade proposal. Returns the trade_id."""
        trade_id = f"trade-{uuid.uuid4().hex[:12]}"
        row = {
            "trade_id": trade_id,
            "agent_id": agent_id,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "strategy_used": strategy,
            "reasoning": reasoning,
            "suggested_hold_days": hold_days,
            "sector": sector,
            "regime_at_entry": regime,
            "outcome": "proposed",
        }
        try:
            self.sb.table("trade_history").insert(row).execute()
            return trade_id
        except Exception as e:
            logger.error(f"Failed to record proposal: {e}")
            return trade_id

    def record_vote_result(
        self,
        trade_id: str,
        approval_pct: float,
        num_voters: int,
        vote_result: str,
        supporting: List[str],
        dissenting: List[str],
    ) -> None:
        """Update a proposal with voting results."""
        try:
            self.sb.table("trade_history") \
                .update({
                    "approval_pct": approval_pct,
                    "num_voters": num_voters,
                    "vote_result": vote_result,
                    "supporting_agents": supporting,
                    "dissenting_agents": dissenting,
                }) \
                .eq("trade_id", trade_id) \
                .execute()
        except Exception as e:
            logger.error(f"Failed to record vote for {trade_id}: {e}")

    def record_execution(
        self,
        trade_id: str,
        entry_price: float,
        quantity: int,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Record that a trade was executed."""
        try:
            self.sb.table("trade_history") \
                .update({
                    "executed": True,
                    "entry_price": entry_price,
                    "entry_date": datetime.utcnow().isoformat(),
                    "quantity": quantity,
                    "fees": fees,
                    "slippage": slippage,
                    "outcome": "open",
                }) \
                .eq("trade_id", trade_id) \
                .execute()
        except Exception as e:
            logger.error(f"Failed to record execution for {trade_id}: {e}")

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        actual_return: float,
        pnl: float,
        actual_hold_days: int,
        fees: float = 0.0,
    ) -> None:
        """Record trade exit and outcome."""
        outcome = "profitable" if pnl > 0 else "losing"
        try:
            self.sb.table("trade_history") \
                .update({
                    "exit_price": exit_price,
                    "exit_date": datetime.utcnow().isoformat(),
                    "actual_return": actual_return,
                    "actual_hold_days": actual_hold_days,
                    "pnl": pnl,
                    "outcome": outcome,
                    "fees": fees,
                }) \
                .eq("trade_id", trade_id) \
                .execute()
        except Exception as e:
            logger.error(f"Failed to record exit for {trade_id}: {e}")

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all currently open trades."""
        try:
            result = self.sb.table("trade_history") \
                .select("*") \
                .eq("outcome", "open") \
                .order("entry_date", desc=True) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []

    def get_agent_trades(
        self, agent_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get trade history for a specific agent."""
        try:
            result = self.sb.table("trade_history") \
                .select("*") \
                .eq("agent_id", agent_id) \
                .order("proposed_at", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get trades for {agent_id}: {e}")
            return []

    def get_all_closed_trades(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get all closed trades for scoring."""
        try:
            result = self.sb.table("trade_history") \
                .select("*") \
                .in_("outcome", ["profitable", "losing"]) \
                .order("exit_date", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get closed trades: {e}")
            return []

    def get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent trades regardless of status."""
        try:
            result = self.sb.table("trade_history") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    # ==============================================================
    # AGENT LEARNING
    # ==============================================================

    def record_lesson(
        self,
        agent_id: str,
        trade_id: str,
        lesson_type: str,
        symbol: str,
        strategy: str,
        confidence_at_pick: float,
        actual_outcome: float,
        lesson: str,
        weight_adjustment: float = 0.0,
    ) -> None:
        """Record what an agent learned from a trade outcome."""
        row = {
            "agent_id": agent_id,
            "trade_id": trade_id,
            "lesson_type": lesson_type,
            "symbol": symbol,
            "strategy_used": strategy,
            "confidence_at_pick": confidence_at_pick,
            "actual_outcome": actual_outcome,
            "lesson": lesson,
            "weight_adjustment": weight_adjustment,
        }
        try:
            self.sb.table("agent_learning").insert(row).execute()
        except Exception as e:
            logger.error(f"Failed to record lesson for {agent_id}: {e}")

    def get_agent_lessons(
        self, agent_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get learning history for an agent."""
        try:
            result = self.sb.table("agent_learning") \
                .select("*") \
                .eq("agent_id", agent_id) \
                .order("learned_at", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get lessons for {agent_id}: {e}")
            return []

    # ==============================================================
    # PIPELINE RUNS
    # ==============================================================

    def start_pipeline_run(self, regime: str = "") -> str:
        """Start a new pipeline run. Returns scan_id."""
        scan_id = f"scan-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        row = {
            "scan_id": scan_id,
            "status": "running",
            "regime": regime,
        }
        try:
            self.sb.table("pipeline_runs").insert(row).execute()
            return scan_id
        except Exception as e:
            logger.error(f"Failed to start pipeline run: {e}")
            return scan_id

    def complete_pipeline_run(
        self, scan_id: str, stats: Dict[str, Any]
    ) -> None:
        """Mark a pipeline run as complete with stats."""
        try:
            self.sb.table("pipeline_runs") \
                .update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    **stats,
                }) \
                .eq("scan_id", scan_id) \
                .execute()
        except Exception as e:
            logger.error(f"Failed to complete pipeline run {scan_id}: {e}")

    def get_recent_pipeline_runs(
        self, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent pipeline runs."""
        try:
            result = self.sb.table("pipeline_runs") \
                .select("*") \
                .order("started_at", desc=True) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get pipeline runs: {e}")
            return []

    # ==============================================================
    # PORTFOLIO SNAPSHOTS
    # ==============================================================

    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Save a daily portfolio snapshot."""
        try:
            self.sb.table("portfolio_snapshots") \
                .upsert(snapshot, on_conflict="snapshot_date") \
                .execute()
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")

    def get_portfolio_history(
        self, days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get portfolio snapshots for equity curve."""
        try:
            result = self.sb.table("portfolio_snapshots") \
                .select("*") \
                .order("snapshot_date", desc=True) \
                .limit(days) \
                .execute()
            return list(reversed(result.data))
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return []

    # ==============================================================
    # RISK EVENTS
    # ==============================================================

    def record_risk_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
        agent_id: str = "",
        symbol: str = "",
    ) -> None:
        """Record a risk event (kill switch, limit breach, etc.)."""
        row = {
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "agent_id": agent_id,
            "symbol": symbol,
        }
        try:
            self.sb.table("risk_events").insert(row).execute()
        except Exception as e:
            logger.error(f"Failed to record risk event: {e}")

    def get_active_risk_events(self) -> List[Dict[str, Any]]:
        """Get unresolved risk events."""
        try:
            result = self.sb.table("risk_events") \
                .select("*") \
                .eq("resolved", False) \
                .order("created_at", desc=True) \
                .execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get risk events: {e}")
            return []

    def resolve_risk_event(self, event_id: int) -> None:
        """Mark a risk event as resolved."""
        try:
            self.sb.table("risk_events") \
                .update({"resolved": True}) \
                .eq("id", event_id) \
                .execute()
        except Exception as e:
            logger.error(f"Failed to resolve risk event {event_id}: {e}")

    # ==============================================================
    # AGENT ACTIVITY FEED
    # ==============================================================

    def log_activity(
        self,
        agent_id: str,
        activity_type: str,
        summary: str,
        symbol: str = "",
        details: Optional[Dict] = None,
        market_mode: str = "",
    ) -> None:
        """Log an agent activity event for the live feed."""
        row = {
            "agent_id": agent_id,
            "activity_type": activity_type,
            "summary": summary,
            "symbol": symbol,
            "details": details or {},
            "market_mode": market_mode,
        }
        try:
            self.sb.table("agent_activity").insert(row).execute()
        except Exception as e:
            logger.error(f"Failed to log activity for {agent_id}: {e}")

    def bulk_log_activity(self, events: List[Dict[str, Any]]) -> None:
        """Log multiple activity events at once."""
        if not events:
            return
        try:
            self.sb.table("agent_activity").insert(events).execute()
        except Exception as e:
            logger.error(f"Failed to bulk log activity: {e}")

    def get_recent_activity(
        self, limit: int = 50, activity_type: str = ""
    ) -> List[Dict[str, Any]]:
        """Get recent agent activity for the live feed."""
        try:
            query = self.sb.table("agent_activity") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit)
            if activity_type:
                query = query.eq("activity_type", activity_type)
            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []

    # ==============================================================
    # AGGREGATE QUERIES (for dashboard)
    # ==============================================================

    def get_agent_leaderboard(self) -> List[Dict[str, Any]]:
        """Get all agents sorted by composite weight (latest scores)."""
        scores = self.get_latest_agent_scores()
        return sorted(scores, key=lambda x: x.get("composite_weight", 0), reverse=True)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get high-level system statistics."""
        try:
            trades = self.sb.table("trade_history") \
                .select("outcome", count="exact") \
                .execute()
            total = trades.count or 0

            profitable = self.sb.table("trade_history") \
                .select("id", count="exact") \
                .eq("outcome", "profitable") \
                .execute()
            wins = profitable.count or 0

            open_trades = self.sb.table("trade_history") \
                .select("id", count="exact") \
                .eq("outcome", "open") \
                .execute()
            currently_open = open_trades.count or 0

            return {
                "total_trades": total,
                "profitable_trades": wins,
                "losing_trades": total - wins - currently_open,
                "open_trades": currently_open,
                "win_rate": wins / max(total - currently_open, 1),
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "losing_trades": 0,
                "open_trades": 0,
                "win_rate": 0,
            }
