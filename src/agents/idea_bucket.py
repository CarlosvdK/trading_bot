"""Idea Bucket — collects all TradePicks from all agents before voting."""

from collections import defaultdict
from typing import Dict, List, Optional

from src.agents.trading_agent import TradePick
from src.agents.sector_mapping import get_sector


class IdeaBucket:
    """Collects TradePick submissions from all agents and groups by symbol."""

    def __init__(self):
        self._picks: List[TradePick] = []

    def submit(self, picks: List[TradePick]) -> None:
        """Submit a batch of picks from an agent."""
        self._picks.extend(picks)

    def submit_single(self, pick: TradePick) -> None:
        """Submit a single pick."""
        self._picks.append(pick)

    def get_candidates(self) -> Dict[str, List[TradePick]]:
        """
        Get all picks grouped by symbol.

        Returns:
            Dict mapping symbol to list of TradePicks for that symbol.
        """
        grouped: Dict[str, List[TradePick]] = defaultdict(list)
        for pick in self._picks:
            grouped[pick.symbol].append(pick)
        return dict(grouped)

    def clear(self) -> None:
        """Clear all submitted picks."""
        self._picks.clear()

    def get_stats(self) -> dict:
        """
        Get summary statistics about current picks.

        Returns:
            Dict with pick counts, sector breakdown, strategy breakdown.
        """
        if not self._picks:
            return {
                "total_picks": 0,
                "unique_symbols": 0,
                "unique_agents": 0,
                "by_sector": {},
                "by_strategy": {},
                "by_direction": {},
                "avg_confidence": 0.0,
            }

        sectors: Dict[str, int] = defaultdict(int)
        strategies: Dict[str, int] = defaultdict(int)
        directions: Dict[str, int] = defaultdict(int)
        agents = set()
        symbols = set()
        confidences = []

        for pick in self._picks:
            sector = pick.sector or get_sector(pick.symbol) or "unknown"
            sectors[sector] += 1
            strategies[pick.strategy_used] += 1
            directions[pick.direction] += 1
            agents.add(pick.agent_id)
            symbols.add(pick.symbol)
            confidences.append(pick.confidence)

        return {
            "total_picks": len(self._picks),
            "unique_symbols": len(symbols),
            "unique_agents": len(agents),
            "by_sector": dict(sectors),
            "by_strategy": dict(strategies),
            "by_direction": dict(directions),
            "avg_confidence": sum(confidences) / len(confidences),
        }

    @property
    def total_picks(self) -> int:
        """Total number of picks submitted."""
        return len(self._picks)

    @property
    def all_picks(self) -> List[TradePick]:
        """All submitted picks (read-only copy)."""
        return list(self._picks)
