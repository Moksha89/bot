"""
Scheduler service.
Runs the strategy loop at configured intervals, manages the trading cycle.
"""

import logging
import traceback
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.capital_client import CapitalClient, CapitalAPIError
from app.config import settings
from app.db.models import BalanceHistory
from app.db.session import async_session
from app.execution.order_manager import OrderManager
from app.execution.position_manager import PositionManager
from app.execution.risk_manager import RiskManager
from app.services.market_data import MarketDataService
from app.services.ai_analyst import AIAnalyst
from app.services.notifier import TelegramNotifier
from app.strategy.signals import SignalGenerator

logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    Orchestrates the trading loop:
    1. Fetch market data
    2. Generate signals
    3. Check risk
    4. Execute trades
    5. Monitor positions
    6. Send notifications
    """

    def __init__(self) -> None:
        self.client = CapitalClient()
        self.market_data = MarketDataService(self.client)
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(self.client, self.risk_manager)
        self.position_manager = PositionManager(self.client)
        self.notifier = TelegramNotifier()
        self.ai_analyst = AIAnalyst()
        self._is_running = False
        self._authenticated = False

    async def initialize(self) -> bool:
        """Initialize the scheduler: authenticate and set up resources."""
        try:
            if settings.trading.mode in ("demo", "live"):
                await self.client.authenticate()
                self._authenticated = True
                logger.info("Authenticated with Capital.com")
            else:
                logger.info("Running in %s mode, skipping authentication", settings.trading.mode)
                self._authenticated = False

            await self.notifier.notify_bot_started()
            self._is_running = True
            return True
        except CapitalAPIError as e:
            logger.error("Failed to initialize: %s", e)
            await self.notifier.notify_error("scheduler", f"Init failed: {e}")
            return False

    async def run_cycle(self) -> dict:
        """
        Run one trading cycle.
        Returns a summary dict of what happened.
        """
        if not self._is_running:
            return {"status": "stopped", "message": "Scheduler is not running"}

        if settings.kill_switch:
            return {"status": "killed", "message": "Kill switch is activated"}

        symbol = settings.trading.symbol
        timeframe = settings.trading.timeframe
        summary: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": None,
            "trade": None,
            "positions_checked": 0,
            "errors": [],
        }

        async with async_session() as session:
            try:
                # Step 1: Get account balance
                account_balance = 10000.0  # Default for paper/analysis mode
                if self._authenticated:
                    try:
                        balance_data = await self.client.get_account_balance()
                        account_balance = balance_data.get("balance", 10000.0)
                        await self._record_balance(session, balance_data)
                    except CapitalAPIError as e:
                        logger.warning("Failed to get balance: %s", e)
                        summary["errors"].append(f"Balance fetch: {e}")

                # Step 2: Fetch candles
                df = await self.market_data.get_candles(symbol, timeframe)
                if df is None or len(df) < 60:
                    msg = f"Insufficient data for {symbol} ({len(df) if df is not None else 0} candles)"
                    summary["errors"].append(msg)
                    logger.warning(msg)
                    return summary

                # Step 3: Get spread
                spread = await self.market_data.get_spread(symbol)

                # Step 4: Check open positions
                has_long = await self.position_manager.has_open_long(session, symbol)
                has_short = await self.position_manager.has_open_short(session, symbol)

                # Step 5: Generate signal
                signal = self.signal_generator.generate(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    spread=spread,
                    has_open_long=has_long,
                    has_open_short=has_short,
                )
                summary["signal"] = {
                    "direction": signal.direction,
                    "close": signal.close_price,
                    "ema_fast": signal.ema_fast,
                    "ema_slow": signal.ema_slow,
                    "rsi": signal.rsi,
                    "reasons": signal.reasons,
                }

                logger.info(
                    "Signal: %s %s | Close=%.5f EMA_F=%.5f EMA_S=%.5f RSI=%.2f",
                    signal.direction,
                    symbol,
                    signal.close_price,
                    signal.ema_fast,
                    signal.ema_slow,
                    signal.rsi,
                )

                # Step 5b: AI market analysis before trade execution
                ai_analysis = None
                if (
                    signal.direction != "NO_TRADE"
                    and settings.ai_analysis_enabled
                    and self.ai_analyst.enabled
                ):
                    ai_analysis = await self.ai_analyst.analyze_market(
                        symbol=symbol,
                        timeframe=timeframe,
                        df=df,
                        signal_direction=signal.direction,
                        indicators={
                            "ema_fast": signal.ema_fast,
                            "ema_slow": signal.ema_slow,
                            "rsi": signal.rsi,
                            "atr": signal.atr,
                        },
                        spread=spread,
                        account_balance=account_balance,
                    )
                    summary["ai_analysis"] = ai_analysis
                    logger.info(
                        "AI analysis: %s (confidence: %.1f%%) - %s",
                        ai_analysis["recommendation"],
                        ai_analysis["confidence"],
                        ai_analysis["analysis"][:100],
                    )

                    # If AI rejects the trade, override signal to NO_TRADE
                    if ai_analysis["recommendation"] == "REJECT":
                        original_direction = signal.direction
                        logger.warning(
                            "AI rejected %s signal for %s: %s",
                            original_direction,
                            symbol,
                            ai_analysis["analysis"],
                        )
                        signal.direction = "NO_TRADE"
                        signal.reasons.append(
                            f"AI REJECTED: {ai_analysis['analysis']}"
                        )
                        await self.notifier.notify_risk_limit(
                            f"AI rejected {original_direction} on {symbol}: "
                            f"{ai_analysis['analysis']}"
                        )
                    elif ai_analysis["recommendation"] == "HOLD":
                        original_direction = signal.direction
                        logger.info(
                            "AI suggests HOLD for %s on %s, skipping trade",
                            original_direction,
                            symbol,
                        )
                        signal.direction = "NO_TRADE"
                        signal.reasons.append(
                            f"AI HOLD: {ai_analysis['analysis']}"
                        )

                # Step 6: Process signal
                result = await self.order_manager.process_signal(
                    session, signal, account_balance
                )
                summary["trade"] = result

                if result and result.get("status", "").upper() == "FILLED":
                    await self.notifier.notify_trade_opened(
                        direction=signal.direction,
                        symbol=symbol,
                        size=result.get("size", 0),
                        entry=result.get("entry", 0),
                        sl=result.get("sl"),
                        tp=result.get("tp"),
                        mode=result.get("mode", ""),
                    )

                # Step 7: Check SL/TP for paper positions
                if settings.trading.mode in ("paper", "analysis"):
                    prices = await self.market_data.get_current_prices([symbol])
                    closed = await self.position_manager.check_sl_tp(session, prices)
                    summary["positions_checked"] = len(closed)
                    for c in closed:
                        await self.notifier.notify_trade_closed(
                            direction=c.get("direction", ""),
                            symbol=c.get("symbol", ""),
                            pnl=c.get("pnl", 0),
                            result=c.get("result", ""),
                        )

                # Step 8: Sync positions from API
                if self._authenticated:
                    await self.position_manager.sync_positions_from_api(session)

            except Exception as e:
                error_msg = f"Cycle error: {e}"
                summary["errors"].append(error_msg)
                logger.error("%s\n%s", error_msg, traceback.format_exc())
                await self.notifier.notify_error("scheduler", error_msg)

        return summary

    async def _record_balance(
        self, session: AsyncSession, balance_data: dict
    ) -> None:
        """Record balance snapshot to the database."""
        record = BalanceHistory(
            balance=balance_data.get("balance", 0),
            equity=balance_data.get("equity", 0),
            available=balance_data.get("available", 0),
            pnl=balance_data.get("pnl", 0),
        )
        session.add(record)
        await session.commit()

    async def stop(self) -> None:
        """Stop the scheduler and clean up."""
        self._is_running = False
        await self.client.close()
        logger.info("Trading scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running
