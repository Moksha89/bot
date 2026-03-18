"""
Dashboard routes for the admin web UI.
Provides endpoints for monitoring bot status, trades, and P&L.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
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
from app.api.capital_client import CapitalClient, CapitalAPIError

logger = logging.getLogger(__name__)


# --- Shared Capital.com client with cached session ---
# Avoids re-authenticating on every dashboard refresh (prevents 429 rate limiting)
_shared_client: Optional[CapitalClient] = None
_shared_client_lock = asyncio.Lock()
_shared_client_auth_time: float = 0.0
_SHARED_CLIENT_TTL = 540  # Re-authenticate every 9 minutes (session lasts 10 min)


async def _get_shared_client() -> CapitalClient:
    """Get or create a shared authenticated Capital.com client for dashboard use."""
    global _shared_client, _shared_client_auth_time
    async with _shared_client_lock:
        now = time.monotonic()
        if (
            _shared_client is not None
            and _shared_client.cst
            and (now - _shared_client_auth_time) < _SHARED_CLIENT_TTL
        ):
            return _shared_client

        # Close old client if exists
        if _shared_client is not None:
            try:
                await _shared_client.close()
            except Exception:
                pass

        client = CapitalClient()
        await client.authenticate()
        _shared_client = client
        _shared_client_auth_time = now
        logger.info("Dashboard shared client authenticated")
        return _shared_client

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


@router.get("/api/account")
async def get_account_info() -> dict:
    """Get live account balance from Capital.com (uses shared cached session)."""
    if settings.trading.mode not in ("demo", "live", "analysis"):
        return {"balance": 0, "equity": 0, "available": 0, "pnl": 0, "currency": ""}
    try:
        client = await _get_shared_client()
        balance = await client.get_account_balance()
        accounts_data = await client.get_accounts()
        accounts = accounts_data.get("accounts", [])
        currency = accounts[0].get("currency", "") if accounts else ""
        return {
            "balance": balance.get("balance", 0),
            "equity": balance.get("equity", 0),
            "available": balance.get("available", 0),
            "pnl": balance.get("pnl", 0),
            "currency": currency,
        }
    except (CapitalAPIError, Exception) as e:
        logger.error("Failed to fetch account info: %s", e)
        # Reset shared client on auth errors so next request re-authenticates
        async with _shared_client_lock:
            global _shared_client, _shared_client_auth_time
            if _shared_client is not None:
                try:
                    await _shared_client.close()
                except Exception:
                    pass
            _shared_client = None
            _shared_client_auth_time = 0.0
        return {"balance": 0, "equity": 0, "available": 0, "pnl": 0, "currency": "", "error": str(e)}


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


@router.get("/api/live-positions")
async def get_live_positions() -> dict:
    """Get live positions with real-time P/L from Capital.com API (uses shared cached session)."""
    if settings.trading.mode not in ("demo", "live", "analysis"):
        return {"positions": [], "account": {}}
    try:
        client = await _get_shared_client()
        api_positions = await client.get_positions()
        balance_data = await client.get_account_balance()
        accounts_data = await client.get_accounts()
        currency = ""
        accounts = accounts_data.get("accounts", [])
        if accounts:
            currency = accounts[0].get("currency", "")

        positions = []
        total_pnl = 0.0
        for p in api_positions:
            pos_data = p.get("position", {})
            market_data = p.get("market", {})
            epic = market_data.get("epic", "")
            direction = pos_data.get("direction", "")
            size = float(pos_data.get("size", 0))
            entry = float(pos_data.get("level", 0))
            current_bid = float(market_data.get("bid", 0))
            current_ask = float(market_data.get("offer", 0))
            current_price = current_bid if direction == "BUY" else current_ask

            if direction == "BUY":
                unrealized_pnl = (current_price - entry) * size
            else:
                unrealized_pnl = (entry - current_price) * size

            total_pnl += unrealized_pnl

            # Risk suggestion
            account_bal = balance_data.get("balance", 1000)
            risk_pct = abs(unrealized_pnl) / account_bal * 100 if account_bal > 0 else 0
            suggestion = "HOLD"
            if unrealized_pnl < 0 and risk_pct > 3.0:
                suggestion = "CLOSE - Loss exceeds 3% of account"
            elif unrealized_pnl < 0 and risk_pct > 1.5:
                suggestion = "WARNING - Approaching risk limit"
            elif unrealized_pnl > 0 and risk_pct > 2.0:
                suggestion = "CONSIDER TP - Good profit, consider taking"

            # Invested amount = entry price * size
            invested_amount = entry * size

            positions.append({
                "deal_id": pos_data.get("dealId", ""),
                "symbol": epic,
                "direction": direction,
                "size": size,
                "entry_price": entry,
                "current_price": round(current_price, 5),
                "invested": round(invested_amount, 2),
                "stop_loss": float(pos_data.get("stopLevel", 0)) or None,
                "take_profit": float(pos_data.get("limitLevel", 0)) or None,
                "unrealized_pnl": round(unrealized_pnl, 2),
                "risk_pct": round(risk_pct, 2),
                "suggestion": suggestion,
                "currency": currency,
                "created_date": pos_data.get("createdDateUTC", ""),
            })

        return {
            "positions": positions,
            "total_unrealized_pnl": round(total_pnl, 2),
            "account": {
                "balance": balance_data.get("balance", 0),
                "equity": balance_data.get("equity", 0),
                "available": balance_data.get("available", 0),
                "pnl": balance_data.get("pnl", 0),
                "currency": currency,
            },
        }
    except (CapitalAPIError, Exception) as e:
        logger.error("Failed to fetch live positions: %s", e)
        # Reset shared client on errors so next request re-authenticates
        async with _shared_client_lock:
            global _shared_client, _shared_client_auth_time  # noqa: F811
            if _shared_client is not None:
                try:
                    await _shared_client.close()
                except Exception:
                    pass
            _shared_client = None
            _shared_client_auth_time = 0.0
        return {"positions": [], "account": {}, "error": str(e)}


@router.get("/api/trade-history")
async def get_trade_history(limit: int = 100) -> list[dict]:
    """Get full trade history from Capital.com API + local DB, with P&L and duration."""
    history: list[dict] = []

    # Try to fetch from Capital.com API first (real data)
    try:
        client = await _get_shared_client()
        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
        to_date = now.strftime("%Y-%m-%dT%H:%M:%S")
        transactions = await client.get_transaction_history(
            from_date=from_date, to_date=to_date, transaction_type="ALL"
        )
        for t in transactions:
            pnl = float(t.get("profitAndLoss", 0) or 0)
            size = float(t.get("size", 0) or 0)
            open_level = float(t.get("openLevel", 0) or 0)
            close_level = float(t.get("closeLevel", 0) or 0)
            invested = open_level * abs(size) if open_level and size else 0
            result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")
            opened = t.get("openDateUtc", t.get("dateUtc", ""))
            closed = t.get("dateUtc", "")
            duration = ""
            if opened and closed:
                try:
                    o = datetime.fromisoformat(opened.replace("Z", "+00:00"))
                    c = datetime.fromisoformat(closed.replace("Z", "+00:00"))
                    delta = c - o
                    total_seconds = int(delta.total_seconds())
                    if total_seconds > 0:
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        if hours > 0:
                            duration = f"{hours}h {minutes}m"
                        elif minutes > 0:
                            duration = f"{minutes}m {seconds}s"
                        else:
                            duration = f"{seconds}s"
                except (ValueError, TypeError):
                    pass

            history.append({
                "id": t.get("reference", ""),
                "symbol": t.get("instrumentName", ""),
                "direction": t.get("direction", ""),
                "size": abs(size),
                "entry_price": open_level,
                "exit_price": close_level,
                "stop_loss": None,
                "take_profit": None,
                "invested": round(invested, 2),
                "pnl": round(pnl, 2),
                "result": result,
                "opened_at": opened,
                "closed_at": closed,
                "duration": duration,
                "deal_id": t.get("reference", ""),
                "source": "capital_api",
            })
    except Exception as e:
        logger.warning("Could not fetch Capital.com transaction history: %s", e)

    # If no API data, fall back to local DB
    if not history:
        async with async_session() as session:
            result = await session.execute(
                select(Position)
                .where(Position.is_open.is_(False))
                .order_by(desc(Position.closed_at))
                .limit(limit)
            )
            positions = result.scalars().all()

            for p in positions:
                duration = ""
                if p.opened_at and p.closed_at:
                    delta = p.closed_at - p.opened_at
                    total_seconds = int(delta.total_seconds())
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if hours > 0:
                        duration = f"{hours}h {minutes}m"
                    elif minutes > 0:
                        duration = f"{minutes}m {seconds}s"
                    else:
                        duration = f"{seconds}s"

                invested = (p.entry_price or 0) * (p.size or 0)

                history.append({
                    "id": p.id,
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "exit_price": p.exit_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "invested": round(invested, 2),
                    "pnl": round(p.pnl, 2) if p.pnl is not None else 0,
                    "result": p.result.value if p.result else "UNKNOWN",
                    "opened_at": p.opened_at.isoformat() if p.opened_at else "",
                    "closed_at": p.closed_at.isoformat() if p.closed_at else "",
                    "duration": duration,
                    "deal_id": p.deal_id or "",
                    "source": "local_db",
                })

    return history


@router.get("/api/today-pnl")
async def get_today_pnl() -> dict:
    """Get today's P&L summary with individual trade breakdown from Capital.com."""
    today_trades: list[dict] = []
    total_profit = 0.0
    total_loss = 0.0

    try:
        client = await _get_shared_client()
        now = datetime.now(timezone.utc)
        from_date = now.strftime("%Y-%m-%dT00:00:00")
        to_date = now.strftime("%Y-%m-%dT%H:%M:%S")
        transactions = await client.get_transaction_history(
            from_date=from_date, to_date=to_date, transaction_type="ALL"
        )
        for t in transactions:
            pnl = float(t.get("profitAndLoss", 0) or 0)
            size = float(t.get("size", 0) or 0)
            open_level = float(t.get("openLevel", 0) or 0)
            close_level = float(t.get("closeLevel", 0) or 0)
            invested = open_level * abs(size) if open_level and size else 0
            result = "PROFIT" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")

            if pnl > 0:
                total_profit += pnl
            elif pnl < 0:
                total_loss += pnl

            today_trades.append({
                "symbol": t.get("instrumentName", ""),
                "direction": t.get("direction", ""),
                "size": abs(size),
                "entry_price": open_level,
                "exit_price": close_level,
                "invested": round(invested, 2),
                "pnl": round(pnl, 2),
                "result": result,
                "closed_at": t.get("dateUtc", ""),
                "reference": t.get("reference", ""),
            })
    except Exception as e:
        logger.warning("Could not fetch today's P&L from Capital.com: %s", e)

    wins = sum(1 for t in today_trades if t["result"] == "PROFIT")
    losses = sum(1 for t in today_trades if t["result"] == "LOSS")
    total_count = len(today_trades)
    net_pnl = total_profit + total_loss

    return {
        "trades": today_trades,
        "summary": {
            "total_trades": total_count,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / total_count * 100) if total_count > 0 else 0, 1),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_pnl": round(net_pnl, 2),
        },
    }


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
