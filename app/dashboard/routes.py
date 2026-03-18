"""
Dashboard routes for the admin web UI.
Provides endpoints for monitoring bot status, trades, and P&L.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select, func, desc

from app.config import settings
from app.db.models import (
    Position,
    Order,
    Signal,
    DailyPnL,
    BalanceHistory,
    ErrorLog,
    TradeResult,
)
from app.db.session import async_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

templates = Jinja2Templates(directory="app/templates")


# --- API response models ---

class BotStatus(BaseModel):
    is_running: bool
    mode: str
    symbol: str
    symbols: list[str]
    timeframe: str
    kill_switch: bool
    uptime: str
    strategy: str
    features: dict


class AccountInfo(BaseModel):
    balance: float
    equity: float
    available: float
    pnl: float


class TradeInfo(BaseModel):
    id: int
    symbol: str
    direction: str
    size: float
    entry_price: float
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    pnl: Optional[float]
    result: str
    is_open: bool
    opened_at: str
    closed_at: Optional[str]


class DashboardStats(BaseModel):
    total_trades: int
    open_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    today_pnl: float
    today_trades: int


# --- Scheduler reference (set from main.py) ---
_scheduler_ref: Optional[object] = None


def set_scheduler(scheduler: object) -> None:
    global _scheduler_ref
    _scheduler_ref = scheduler


# --- API Endpoints ---

@router.get("/api/status", response_model=BotStatus)
async def get_bot_status() -> BotStatus:
    """Get current bot status."""
    is_running = False
    if _scheduler_ref is not None:
        is_running = getattr(_scheduler_ref, "is_running", False)

    return BotStatus(
        is_running=is_running,
        mode=settings.trading.mode,
        symbol=settings.trading.symbol,
        symbols=settings.trading.symbols,
        timeframe=settings.trading.timeframe,
        kill_switch=settings.kill_switch,
        uptime="N/A",
        strategy=settings.trading.strategy,
        features={
            "multi_symbol": len(settings.trading.symbols) > 1,
            "trailing_stop": settings.trailing_stop_enabled,
            "session_filter": settings.session_filter_enabled,
            "news_filter": settings.news_filter_enabled,
            "sentiment": settings.sentiment_enabled,
            "ml_scoring": settings.ml_scoring_enabled,
            "auto_optimize": settings.auto_optimize_enabled,
            "ai_analysis": settings.ai_analysis_enabled,
        },
    )


@router.get("/api/stats", response_model=DashboardStats)
async def get_stats() -> DashboardStats:
    """Get trading statistics."""
    async with async_session() as session:
        # Total trades
        total_q = await session.execute(select(func.count(Position.id)))
        total_trades = total_q.scalar() or 0

        # Open trades
        open_q = await session.execute(
            select(func.count(Position.id)).where(Position.is_open.is_(True))
        )
        open_trades = open_q.scalar() or 0

        # Wins/losses
        wins_q = await session.execute(
            select(func.count(Position.id)).where(Position.result == TradeResult.WIN)
        )
        wins = wins_q.scalar() or 0

        losses_q = await session.execute(
            select(func.count(Position.id)).where(Position.result == TradeResult.LOSS)
        )
        losses = losses_q.scalar() or 0

        closed = wins + losses
        win_rate = (wins / closed * 100) if closed > 0 else 0.0

        # Total P&L
        pnl_q = await session.execute(
            select(func.sum(Position.pnl)).where(Position.is_open.is_(False))
        )
        total_pnl = pnl_q.scalar() or 0.0

        # Today's stats
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_q = await session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily = today_q.scalar_one_or_none()
        today_pnl = daily.total_pnl if daily else 0.0
        today_trades = daily.trade_count if daily else 0

    return DashboardStats(
        total_trades=total_trades,
        open_trades=open_trades,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 1),
        total_pnl=round(total_pnl, 2),
        today_pnl=round(today_pnl, 2),
        today_trades=today_trades,
    )


@router.get("/api/positions")
async def get_positions(open_only: bool = False) -> list[TradeInfo]:
    """Get positions, optionally filtered to open only."""
    async with async_session() as session:
        query = select(Position).order_by(desc(Position.opened_at))
        if open_only:
            query = query.where(Position.is_open.is_(True))
        else:
            query = query.limit(50)

        result = await session.execute(query)
        positions = result.scalars().all()

        return [
            TradeInfo(
                id=p.id,
                symbol=p.symbol,
                direction=p.direction,
                size=p.size,
                entry_price=p.entry_price,
                exit_price=p.exit_price,
                stop_loss=p.stop_loss,
                take_profit=p.take_profit,
                pnl=p.pnl,
                result=p.result.value if p.result else "OPEN",
                is_open=p.is_open,
                opened_at=p.opened_at.isoformat() if p.opened_at else "",
                closed_at=p.closed_at.isoformat() if p.closed_at else None,
            )
            for p in positions
        ]


@router.get("/api/signals")
async def get_recent_signals(limit: int = 20) -> list[dict]:
    """Get recent signals."""
    async with async_session() as session:
        result = await session.execute(
            select(Signal).order_by(desc(Signal.timestamp)).limit(limit)
        )
        signals = result.scalars().all()
        return [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat() if s.timestamp else "",
                "symbol": s.symbol,
                "direction": s.direction.value if s.direction else "",
                "ema_fast": s.ema_fast,
                "ema_slow": s.ema_slow,
                "rsi": s.rsi,
                "close_price": s.close_price,
                "spread": s.spread,
                "reason": s.reason,
                "acted_on": s.acted_on,
            }
            for s in signals
        ]


@router.get("/api/daily-pnl")
async def get_daily_pnl(limit: int = 30) -> list[dict]:
    """Get daily P&L history."""
    async with async_session() as session:
        result = await session.execute(
            select(DailyPnL).order_by(desc(DailyPnL.date)).limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "date": r.date,
                "total_pnl": r.total_pnl,
                "trade_count": r.trade_count,
                "wins": r.wins,
                "losses": r.losses,
            }
            for r in records
        ]


@router.get("/api/errors")
async def get_recent_errors(limit: int = 20) -> list[dict]:
    """Get recent errors."""
    async with async_session() as session:
        result = await session.execute(
            select(ErrorLog).order_by(desc(ErrorLog.timestamp)).limit(limit)
        )
        errors = result.scalars().all()
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                "module": e.module,
                "error_type": e.error_type,
                "message": e.message,
            }
            for e in errors
        ]


@router.post("/api/kill-switch")
async def toggle_kill_switch(enable: bool = True) -> dict:
    """Toggle the emergency kill switch."""
    settings.kill_switch = enable
    status = "ACTIVATED" if enable else "DEACTIVATED"
    logger.warning("Kill switch %s", status)
    return {"kill_switch": enable, "status": status}


@router.get("/api/performance")
async def get_performance() -> dict:
    """Get performance analytics: Sharpe ratio, drawdown, equity curve."""
    from app.services.performance import PerformanceAnalyzer

    import dataclasses
    analyzer = PerformanceAnalyzer()
    async with async_session() as session:
        metrics = await analyzer.calculate_metrics(session)
        return dataclasses.asdict(metrics)


@router.get("/api/backtest")
async def run_backtest(
    symbol: str = "XAUUSD",
    timeframe: str = "HOUR",
    strategy: str = "ema_crossover",
    initial_balance: float = 10000.0,
) -> dict:
    """Run a backtest on historical data."""
    from app.services.backtester import BacktestEngine
    import dataclasses

    # For demo, generate synthetic data if no API connection
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    prices = [2000.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.005)))

    df = pd.DataFrame({
        "open": prices,
        "high": [p * (1 + abs(rng.normal(0, 0.002))) for p in prices],
        "low": [p * (1 - abs(rng.normal(0, 0.002))) for p in prices],
        "close": [p * (1 + rng.normal(0, 0.001)) for p in prices],
        "volume": [rng.integers(100, 1000) for _ in prices],
    })

    engine = BacktestEngine(initial_balance=initial_balance)
    result = engine.run(df, symbol=symbol, timeframe=timeframe, strategy_type=strategy)
    # Convert to dict, limit trades list
    result_dict = dataclasses.asdict(result)
    result_dict["trades"] = result_dict["trades"][:50]  # Limit for response size
    result_dict["equity_curve"] = result_dict["equity_curve"][::5]  # Downsample
    return result_dict


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Serve the dashboard HTML page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})
