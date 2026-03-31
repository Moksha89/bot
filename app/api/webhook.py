"""
Webhook endpoint for receiving external trading signals (e.g., from TradingView).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


class WebhookSignal(BaseModel):
    """External signal received via webhook."""
    symbol: str
    direction: str  # BUY or SELL
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    message: Optional[str] = None
    secret: Optional[str] = None


class WebhookResponse(BaseModel):
    status: str
    message: str
    timestamp: str


@router.post("/signal", response_model=WebhookResponse)
async def receive_signal(signal: WebhookSignal) -> WebhookResponse:
    """
    Receive an external trading signal.
    In the future this can be connected to the execution engine.
    """
    logger.info(
        "Webhook signal received: %s %s @ %s",
        signal.direction,
        signal.symbol,
        signal.price,
    )

    if signal.direction not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="direction must be BUY or SELL")

    return WebhookResponse(
        status="received",
        message=f"Signal {signal.direction} {signal.symbol} queued for processing",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
