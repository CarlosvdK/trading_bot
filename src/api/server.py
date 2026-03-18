"""FastAPI backend for trading dashboard."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logger = logging.getLogger("trading_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to IBKR and Supabase on startup."""
    # Supabase (always try)
    try:
        from src.storage.supabase_client import get_supabase_client
        from src.storage.repository import TradingRepository
        sb = get_supabase_client()
        app.state.repo = TradingRepository(sb)
        logger.info("Supabase connected")
    except Exception as e:
        logger.warning(f"Supabase unavailable: {e}")
        app.state.repo = None

    # IBKR (optional — dashboard works without it)
    app.state.ibkr = None
    try:
        from src.trading.ibkr_broker import IBKRBroker
        config = {
            "host": os.getenv("IBKR_HOST", "127.0.0.1"),
            "port": int(os.getenv("IBKR_PORT", "4002")),
            "client_id": int(os.getenv("IBKR_CLIENT_ID", "1")),
        }
        broker = IBKRBroker(config)
        broker._connect()
        app.state.ibkr = broker
        logger.info("IBKR connected")
    except Exception as e:
        logger.warning(f"IBKR unavailable: {e}")

    # Risk Governor
    try:
        from src.risk_management.risk_governor import RiskGovernor, RiskConfig
        app.state.risk_governor = RiskGovernor(RiskConfig())
    except Exception:
        app.state.risk_governor = None

    yield

    # Cleanup
    if app.state.ibkr:
        try:
            app.state.ibkr.disconnect()
        except Exception:
            pass


app = FastAPI(title="Trading Ops API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from src.api.routers import agents, portfolio, pipeline, risk, market

app.include_router(agents.router)
app.include_router(portfolio.router)
app.include_router(pipeline.router)
app.include_router(risk.router)
app.include_router(market.router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ibkr_connected": app.state.ibkr is not None,
        "supabase_connected": app.state.repo is not None,
    }
