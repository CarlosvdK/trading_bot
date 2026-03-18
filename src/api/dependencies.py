"""FastAPI dependency injection."""

from fastapi import Request


def get_repo(request: Request):
    """Get TradingRepository from app state. Returns None if unavailable."""
    return getattr(request.app.state, "repo", None)


def get_ibkr(request: Request):
    """Get IBKRBroker from app state. Returns None if disconnected."""
    return getattr(request.app.state, "ibkr", None)


def get_risk_governor(request: Request):
    """Get RiskGovernor from app state."""
    return getattr(request.app.state, "risk_governor", None)
