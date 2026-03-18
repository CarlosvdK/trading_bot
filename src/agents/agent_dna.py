"""Agent DNA — personality and configuration for each trading agent."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentDNA:
    """Defines a trading agent's personality, knowledge, and strategy config."""

    agent_id: str                    # unique name like "tech_momentum_aggressive"
    display_name: str                # human readable

    # Knowledge distribution
    primary_sectors: List[str]       # 1-2 sectors they know deeply
    secondary_sectors: List[str] = field(default_factory=list)  # 2-3 sectors moderate knowledge
    # remaining sectors = no knowledge (won't vote on them)

    # Strategy configuration
    primary_strategy: str = "momentum"  # momentum | mean_reversion | value | growth | event_driven | volatility | sentiment | breakout
    secondary_strategy: Optional[str] = None

    # Personality parameters (all 0.0 to 1.0)
    risk_appetite: float = 0.5           # 0=ultra conservative, 1=aggressive
    contrarian_factor: float = 0.0       # 0=follows crowd, 1=always fades consensus
    conviction_style: float = 0.5        # 0=many small picks, 1=few high-conviction picks
    regime_sensitivity: float = 0.3      # 0=ignores regime, 1=heavily adjusts

    # Lookback configuration
    lookback_days: int = 50              # how far back they look (5-252)

    # Timeframe preference
    holding_period: str = "swing"        # scalp(1-3d) | swing(5-15d) | position(30-90d) | macro(90+d)

    # Thresholds
    min_confidence: float = 0.6          # minimum confidence to submit a pick (0.5-0.9)
    max_picks_per_scan: int = 5          # how many picks per daily scan (1-10)

    # Peer group for pre-voting (agents with similar skills vote first)
    peer_group: str = ""                 # cluster name for pre-vote among similar agents

    def knows_sector(self, sector: str) -> bool:
        """Check if this agent has knowledge of a given sector."""
        return sector in self.primary_sectors or sector in self.secondary_sectors

    def sector_weight(self, sector: str) -> float:
        """Return knowledge weight for a sector (1.0 primary, 0.5 secondary, 0.0 none)."""
        if sector in self.primary_sectors:
            return 1.0
        if sector in self.secondary_sectors:
            return 0.5
        return 0.0

    @property
    def all_sectors(self) -> List[str]:
        """All sectors this agent can vote on."""
        return self.primary_sectors + self.secondary_sectors

    @property
    def holding_days_range(self) -> tuple:
        """Return (min_days, max_days) for the holding period."""
        ranges = {
            "scalp": (1, 3),
            "swing": (5, 15),
            "position": (30, 90),
            "macro": (90, 252),
        }
        return ranges.get(self.holding_period, (5, 15))

    def __post_init__(self):
        """Validate DNA parameters."""
        assert 0.0 <= self.risk_appetite <= 1.0, f"risk_appetite must be 0-1, got {self.risk_appetite}"
        assert 0.0 <= self.contrarian_factor <= 1.0, f"contrarian_factor must be 0-1, got {self.contrarian_factor}"
        assert 0.0 <= self.conviction_style <= 1.0, f"conviction_style must be 0-1, got {self.conviction_style}"
        assert 0.0 <= self.regime_sensitivity <= 1.0, f"regime_sensitivity must be 0-1, got {self.regime_sensitivity}"
        assert 5 <= self.lookback_days <= 252, f"lookback_days must be 5-252, got {self.lookback_days}"
        assert self.primary_strategy in VALID_STRATEGIES, f"Invalid strategy: {self.primary_strategy}"
        assert self.holding_period in VALID_HOLDING_PERIODS, f"Invalid holding: {self.holding_period}"


VALID_STRATEGIES = {
    "momentum", "mean_reversion", "value", "growth",
    "event_driven", "volatility", "sentiment", "breakout",
}

VALID_HOLDING_PERIODS = {"scalp", "swing", "position", "macro"}

ALL_SECTORS = [
    "technology", "healthcare", "financials", "energy",
    "consumer_discretionary", "consumer_staples", "industrials",
    "materials", "real_estate", "utilities", "communication_services",
]
