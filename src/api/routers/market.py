"""Market status endpoints."""

from datetime import datetime
from fastapi import APIRouter, Depends
from src.api.dependencies import get_ibkr

router = APIRouter(prefix="/api/market", tags=["Market"])


@router.get("/status")
async def market_status(ibkr=Depends(get_ibkr)):
    """IBKR connection and market status."""
    now = datetime.now()
    # Simple market hours check (ET)
    # TODO: use proper timezone handling
    hour = now.hour
    weekday = now.weekday()
    market_open = weekday < 5 and 9 <= hour < 16

    return {
        "ibkrConnected": ibkr is not None,
        "marketOpen": market_open,
        "timestamp": now.isoformat(),
    }


@router.get("/regime")
async def market_regime():
    """Current detected market regime."""
    try:
        import yfinance as yf
        import numpy as np

        spy = yf.download("SPY", period="3mo", progress=False)
        if spy.empty:
            return {"regime": "unknown"}

        close = spy["Close"].squeeze()
        log_ret = np.log(close / close.shift(1)).dropna()
        vol_21 = log_ret.iloc[-21:].std() * np.sqrt(252)
        ret_21 = (close.iloc[-1] / close.iloc[-22]) - 1

        high_vol = vol_21 > 0.20
        if high_vol:
            if ret_21 > 0.01:
                regime = "high_vol_trending_up"
            elif ret_21 < -0.01:
                regime = "high_vol_trending_down"
            else:
                regime = "high_vol_choppy"
        else:
            if ret_21 > 0.01:
                regime = "low_vol_trending_up"
            elif ret_21 < -0.01:
                regime = "low_vol_trending_down"
            else:
                regime = "low_vol_choppy"

        return {"regime": regime, "vol21d": round(float(vol_21), 4), "ret21d": round(float(ret_21), 4)}
    except Exception:
        return {"regime": "unknown"}
