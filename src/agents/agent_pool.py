"""Agent Pool — orchestrates all agents, peer pre-voting, idea collection, and consensus voting."""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.agents.agent_dna import AgentDNA
from src.agents.trading_agent import TradingAgent, TradePick
from src.agents.idea_bucket import IdeaBucket
from src.agents.voting_engine import VotingEngine, ApprovedTrade
from src.agents.scorekeeper import AgentScorekeeper

logger = logging.getLogger(__name__)


class AgentPool:
    """
    Orchestrates the full multi-agent voting pipeline:
    1. Each agent scans and generates picks
    2. Similar agents (peer groups) pre-vote on each other's picks
    3. Surviving picks go to the idea bucket
    4. Full voting engine runs consensus vote
    5. Approved trades are returned
    """

    def __init__(
        self,
        agents: List[TradingAgent],
        voting_engine: Optional[VotingEngine] = None,
        scorekeeper: Optional[AgentScorekeeper] = None,
        peer_approval_threshold: float = 0.5,
    ):
        """
        Args:
            agents: List of all TradingAgent instances.
            voting_engine: VotingEngine for consensus voting.
            scorekeeper: AgentScorekeeper for adaptive weighting.
            peer_approval_threshold: Min approval % in peer pre-vote to proceed.
        """
        self.agents = agents
        self.voting_engine = voting_engine or VotingEngine()
        self.scorekeeper = scorekeeper or AgentScorekeeper()
        self.peer_approval_threshold = peer_approval_threshold
        self._idea_bucket = IdeaBucket()

        # Build lookup maps
        self._agent_map: Dict[str, TradingAgent] = {
            a.dna.agent_id: a for a in agents
        }
        self._dna_map: Dict[str, AgentDNA] = {
            a.dna.agent_id: a.dna for a in agents
        }

        # Build peer groups
        self._peer_groups: Dict[str, List[str]] = self._build_peer_groups()

    def daily_scan(
        self,
        universe_data: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
    ) -> List[ApprovedTrade]:
        """
        Run full daily scan pipeline.

        Args:
            universe_data: {symbol: OHLCV DataFrame} for all symbols.
            index_df: Index OHLCV DataFrame.
            current_date: Date to scan at.

        Returns:
            List of ApprovedTrade from consensus voting.
        """
        self._idea_bucket.clear()

        # Phase 1: Each agent generates raw picks
        all_raw_picks: Dict[str, List[TradePick]] = {}
        for agent in self.agents:
            try:
                picks = agent.scan(universe_data, index_df, current_date)
                if picks:
                    all_raw_picks[agent.dna.agent_id] = picks
            except Exception as e:
                logger.warning(
                    f"Agent {agent.dna.agent_id} scan failed: {e}"
                )

        # Phase 2: Peer pre-voting
        approved_picks = self._peer_pre_vote(all_raw_picks)

        # Phase 3: Submit to idea bucket
        for picks in approved_picks.values():
            self._idea_bucket.submit(picks)

        # Phase 4: Run consensus vote
        candidates = self._idea_bucket.get_candidates()
        weights = self._get_weights()

        approved = self.voting_engine.run_vote(
            candidates, weights, self._dna_map
        )

        logger.info(
            f"Daily scan: {self._idea_bucket.total_picks} picks from "
            f"{len(all_raw_picks)} agents -> {len(approved)} approved trades"
        )

        return approved

    def _peer_pre_vote(
        self, raw_picks: Dict[str, List[TradePick]]
    ) -> Dict[str, List[TradePick]]:
        """
        Run peer pre-voting: agents in the same peer group vote on
        each other's picks before they reach the main voting pool.

        Picks that pass peer pre-vote get a peer_approved flag and bonus.
        Picks that fail are still submitted but without the bonus.

        Args:
            raw_picks: {agent_id: [TradePick]} from each agent.

        Returns:
            {agent_id: [TradePick]} with peer_approved flags set.
        """
        result: Dict[str, List[TradePick]] = {}

        for agent_id, picks in raw_picks.items():
            peer_group = self._dna_map[agent_id].peer_group
            if not peer_group or peer_group not in self._peer_groups:
                # No peer group, submit directly
                result[agent_id] = picks
                continue

            peers = self._peer_groups[peer_group]
            peer_agents = [
                pid for pid in peers
                if pid != agent_id and pid in self._agent_map
            ]

            if len(peer_agents) < 2:
                # Not enough peers, submit directly
                result[agent_id] = picks
                continue

            approved_picks: List[TradePick] = []
            for pick in picks:
                # Count how many peers also picked this symbol (or would)
                votes_for = 0
                votes_total = len(peer_agents)

                for peer_id in peer_agents:
                    peer_picks = raw_picks.get(peer_id, [])
                    # Check if peer picked the same symbol in same direction
                    peer_match = any(
                        pp.symbol == pick.symbol
                        and pp.direction == pick.direction
                        for pp in peer_picks
                    )
                    if peer_match:
                        votes_for += 1

                approval_pct = votes_for / votes_total if votes_total > 0 else 0

                if approval_pct >= self.peer_approval_threshold:
                    pick.peer_approved = True
                    pick.peer_approval_pct = approval_pct
                else:
                    pick.peer_approved = False
                    pick.peer_approval_pct = approval_pct

                # All picks go through, but peer-approved ones get bonus
                approved_picks.append(pick)

            result[agent_id] = approved_picks

        return result

    def _build_peer_groups(self) -> Dict[str, List[str]]:
        """Build peer group membership from agent DNAs."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for agent in self.agents:
            if agent.dna.peer_group:
                groups[agent.dna.peer_group].append(agent.dna.agent_id)
        return dict(groups)

    def _get_weights(self) -> Dict[str, float]:
        """Get current weights for all agents."""
        weights = {}
        for agent in self.agents:
            weights[agent.dna.agent_id] = self.scorekeeper.get_weight(
                agent.dna.agent_id
            )
        return weights

    def record_outcomes(self, fills: List[dict]) -> None:
        """
        Record trade outcomes for all agents that picked these symbols.

        Args:
            fills: List of dicts with keys: symbol, actual_return, actual_days.
        """
        # Build lookup of which agents picked which symbols
        agent_picks: Dict[str, Dict[str, TradePick]] = {}
        for pick in self._idea_bucket.all_picks:
            if pick.agent_id not in agent_picks:
                agent_picks[pick.agent_id] = {}
            agent_picks[pick.agent_id][pick.symbol] = pick

        for fill in fills:
            symbol = fill["symbol"]
            actual_return = fill["actual_return"]
            actual_days = fill["actual_days"]

            for agent_id, picks_by_sym in agent_picks.items():
                if symbol in picks_by_sym:
                    self.scorekeeper.record_outcome(
                        agent_id,
                        picks_by_sym[symbol],
                        actual_return,
                        actual_days,
                    )

    def get_leaderboard(self) -> pd.DataFrame:
        """Get agent performance leaderboard."""
        return self.scorekeeper.get_leaderboard()

    def get_scan_stats(self) -> dict:
        """Get statistics from the last scan."""
        return self._idea_bucket.get_stats()

    def save_state(self, path: str) -> None:
        """Save scorekeeper state to file."""
        self.scorekeeper.save(path)

    def load_state(self, path: str) -> None:
        """Load scorekeeper state from file."""
        self.scorekeeper.load(path)
