"""
Main FastAPI application entry point for the Capital.com Trading Bot.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.api.webhook import router as webhook_router
from app.dashboard.routes import router as dashboard_router, set_scheduler
from app.services.scheduler import TradingScheduler

# --- Logging setup ---
logging.basicConfig(
    level=getattr(logging, settings.server.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log"),
    ],
)
logger = logging.getLogger(__name__)

# --- Scheduler globals ---
trading_scheduler: TradingScheduler = TradingScheduler()
ap_scheduler: AsyncIOScheduler = AsyncIOScheduler()

# Interval mapping for strategy check frequency
TIMEFRAME_INTERVALS: dict[str, dict[str, int]] = {
    "MINUTE": {"minutes": 1},
    "MINUTE_5": {"minutes": 5},
    "MINUTE_15": {"minutes": 15},
    "MINUTE_30": {"minutes": 30},
    "HOUR": {"hours": 1},
    "HOUR_4": {"hours": 4},
    "DAY": {"days": 1},
}


async def run_trading_cycle() -> None:
    """Wrapper for the scheduled trading cycle."""
    try:
        result = await trading_scheduler.run_cycle()
        signal = result.get("signal")
        if signal:
            logger.info(
                "Cycle complete: signal=%s errors=%d",
                signal.get("direction", "?"),
                len(result.get("errors", [])),
            )
    except Exception as e:
        logger.error("Trading cycle error: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown."""
    # Startup
    logger.info("Starting Capital.com Trading Bot...")
    logger.info(
        "Mode: %s | Symbol: %s | Timeframe: %s",
        settings.trading.mode,
        settings.trading.symbol,
        settings.trading.timeframe,
    )

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize trading scheduler
    initialized = await trading_scheduler.initialize()
    set_scheduler(trading_scheduler)

    if initialized:
        # Set up periodic trading cycle
        interval = TIMEFRAME_INTERVALS.get(
            settings.trading.timeframe, {"hours": 1}
        )
        ap_scheduler.add_job(
            run_trading_cycle,
            "interval",
            **interval,
            id="trading_cycle",
            max_instances=1,
            replace_existing=True,
        )
        ap_scheduler.start()
        logger.info("Trading scheduler started (interval: %s)", interval)

        # Run first cycle immediately
        await run_trading_cycle()
    else:
        logger.warning(
            "Trading scheduler failed to initialize, running in API-only mode"
        )

    yield

    # Shutdown
    logger.info("Shutting down trading bot...")
    ap_scheduler.shutdown(wait=False)
    await trading_scheduler.stop()
    logger.info("Trading bot stopped")


# --- FastAPI app ---
app = FastAPI(
    title="Capital.com Trading Bot",
    description="Automated trading bot with strategy engine, risk management, and dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(webhook_router)
app.include_router(dashboard_router)


@app.get("/healthz")
async def healthz() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": settings.trading.mode,
        "symbol": settings.trading.symbol,
        "kill_switch": settings.kill_switch,
        "scheduler_running": trading_scheduler.is_running,
    }


@app.get("/api/config")
async def get_config() -> dict:
    """Get current bot configuration (non-sensitive)."""
    return {
        "trading": {
            "symbol": settings.trading.symbol,
            "timeframe": settings.trading.timeframe,
            "mode": settings.trading.mode,
        },
        "strategy": {
            "ema_fast": settings.strategy.ema_fast,
            "ema_slow": settings.strategy.ema_slow,
            "rsi_period": settings.strategy.rsi_period,
            "rsi_buy_range": [
                settings.strategy.rsi_buy_min,
                settings.strategy.rsi_buy_max,
            ],
            "rsi_sell_range": [
                settings.strategy.rsi_sell_min,
                settings.strategy.rsi_sell_max,
            ],
            "max_spread": settings.strategy.max_spread,
        },
        "risk": {
            "risk_per_trade": settings.risk.risk_per_trade,
            "max_daily_loss": settings.risk.max_daily_loss,
            "max_consecutive_losses": settings.risk.max_consecutive_losses,
            "cooldown_minutes": settings.risk.cooldown_minutes,
            "max_open_trades": settings.risk.max_open_trades,
            "sl_atr_multiplier": settings.risk.sl_atr_multiplier,
            "tp_risk_reward": settings.risk.tp_risk_reward,
        },
    }


@app.post("/api/run-cycle")
async def trigger_cycle() -> dict:
    """Manually trigger a trading cycle."""
    result = await trading_scheduler.run_cycle()
    return result
