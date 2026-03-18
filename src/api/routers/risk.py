"""Risk endpoints — alerts, controls, kill switch."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from src.api.dependencies import get_repo, get_ibkr, get_risk_governor

router = APIRouter(prefix="/api/risk", tags=["Risk"])


@router.get("/summary")
async def risk_summary(ibkr=Depends(get_ibkr), repo=Depends(get_repo), rg=Depends(get_risk_governor)):
    """Current risk metrics."""
    summary = {
        "currentDrawdown": 0, "maxDrawdownLimit": 0.15,
        "dailyLoss": 0, "dailyLossLimit": 0.03,
        "grossExposure": 0, "grossExposureLimit": 1.0,
        "netExposure": 0, "leverage": 1.0,
        "concentrationRisk": 0, "correlationRisk": 0,
        "liquidityRisk": 0, "regimeMismatch": False,
        "killSwitchActive": False, "killSwitchReason": "",
    }

    if rg:
        summary["killSwitchActive"] = rg.kill_switch_active
        summary["killSwitchReason"] = str(rg.kill_switch_reason) if rg.kill_switch_reason else ""
        summary["maxDrawdownLimit"] = rg.config.max_portfolio_drawdown
        summary["dailyLossLimit"] = rg.config.max_daily_loss_pct
        summary["grossExposureLimit"] = rg.config.max_gross_exposure_pct

    if ibkr:
        try:
            acct = ibkr.get_account_summary()
            nav = acct.get("NetLiquidation", 0)
            gross = acct.get("GrossPositionValue", 0)
            if nav > 0:
                summary["grossExposure"] = gross / nav
                summary["netExposure"] = (nav - acct.get("TotalCashValue", 0)) / nav
        except Exception:
            pass

    return summary


@router.get("/alerts")
async def risk_alerts(repo=Depends(get_repo)):
    if not repo:
        return []
    try:
        return repo.get_active_risk_events()
    except Exception:
        return []


@router.get("/controls")
async def risk_controls(rg=Depends(get_risk_governor)):
    """Current control state."""
    return {
        "tradingPaused": False,
        "killSwitchActive": rg.kill_switch_active if rg else False,
        "killSwitchReason": str(rg.kill_switch_reason) if rg and rg.kill_switch_reason else "",
    }


class KillSwitchRequest(BaseModel):
    activate: bool
    operator: str = "dashboard_user"


@router.post("/killswitch")
async def toggle_kill_switch(req: KillSwitchRequest, rg=Depends(get_risk_governor), repo=Depends(get_repo)):
    if not rg:
        return {"error": "Risk governor not available"}

    if req.activate:
        from src.risk_management.risk_governor import KillSwitchReason
        from datetime import date
        rg._trigger_kill_switch(KillSwitchReason.MANUAL, date.today())
        if repo:
            try:
                repo.record_risk_event("kill_switch", "critical", f"Kill switch activated by {req.operator}")
            except Exception:
                pass
        return {"status": "activated"}
    else:
        rg.manual_reset_kill_switch(req.operator)
        if repo:
            try:
                repo.record_risk_event("kill_switch_reset", "info", f"Kill switch deactivated by {req.operator}")
            except Exception:
                pass
        return {"status": "deactivated"}


@router.post("/alerts/{event_id}/resolve")
async def resolve_alert(event_id: int, repo=Depends(get_repo)):
    if not repo:
        return {"error": "Database not available"}
    try:
        repo.resolve_risk_event(event_id)
        return {"status": "resolved"}
    except Exception as e:
        return {"error": str(e)}
