"""
Scheduler service.
Runs the strategy loop at configured intervals, manages the trading cycle.
Supports multi-symbol trading, session/news filters, trailing stops,
ML scoring, sentiment analysis, and auto-optimization.
"""

import logging
import traceback
from datetime import datetime, timezone

import pandas as pd
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
from app.services.session_filter import SessionFilter
from app.services.news_filter import NewsFilter
from app.services.trailing_stop import TrailingStopManager, TrailingStopConfig, TrailingMode
from app.services.sentiment import SentimentAnalyzer
from app.services.ml_scorer import MLSignalScorer
from app.services.auto_optimizer import AutoOptimizer
from app.services.portfolio import PortfolioManager
from app.strategy.indicators import calculate_atr
from app.strategy.signals import SignalGenerator
from app.strategy.strategies import StrategySelector

logger = logging.getLogger(__name__)


class TradingScheduler:
    """
    Orchestrates the trading loop:
    1. Check session and news filters
    2. Fetch market data for all symbols
    3. Generate signals (multi-strategy)
    4. ML scoring and sentiment check
    5. AI analysis (text + chart vision)
    6. Portfolio risk check
    7. Execute trades
    8. Update trailing stops
    9. Monitor positions
    10. Send notifications
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
        self._latest_atr: dict[str, float] = {}  # ATR values from last symbol processing

        # New feature modules
        self.strategy_selector = StrategySelector()
        self.session_filter = SessionFilter(
            allowed_sessions=[
                s.strip() for s in settings.allowed_sessions.split(",") if s.strip()
            ],
            enabled=settings.session_filter_enabled,
        )
        self.news_filter = NewsFilter(
            enabled=settings.news_filter_enabled,
            buffer_before=settings.news_buffer_minutes,
            buffer_after=settings.news_buffer_minutes,
        )
        self.trailing_stop_manager = TrailingStopManager(
            TrailingStopConfig(
                enabled=settings.trailing_stop_enabled,
                mode=TrailingMode.ATR,
                activation_r=settings.trailing_stop_activation_r,
                atr_multiplier=settings.trailing_stop_atr_mult,
            )
        )
        self.sentiment_analyzer = SentimentAnalyzer(enabled=settings.sentiment_enabled)
        self.ml_scorer = MLSignalScorer(enabled=settings.ml_scoring_enabled)
        self.auto_optimizer = AutoOptimizer(
            enabled=settings.auto_optimize_enabled,
            optimize_interval_hours=settings.auto_optimize_interval_hours,
        )
        self.portfolio_manager = PortfolioManager(
            max_portfolio_risk=settings.risk.max_portfolio_risk,
            max_correlated_exposure=settings.risk.max_correlated_exposure,
            max_positions=settings.risk.max_total_positions,
        )

    async def initialize(self) -> bool:
        """Initialize the scheduler: authenticate and set up resources."""
        try:
            if settings.trading.mode in ("demo", "live", "analysis"):
                await self.client.authenticate()
                self._authenticated = True
                logger.info("Authenticated with Capital.com (%s mode)", settings.trading.mode)
            elif settings.trading.mode == "paper":
                logger.info("Running in paper mode, skipping authentication")
                self._authenticated = False
            else:
                logger.info("Running in %s mode, skipping authentication", settings.trading.mode)
                self._authenticated = False

            # Fetch news events if enabled
            if settings.news_filter_enabled:
                await self.news_filter.fetch_events()

            await self.notifier.notify_bot_started()
            self._is_running = True
            return True
        except CapitalAPIError as e:
            logger.error("Failed to initialize: %s", e)
            await self.notifier.notify_error("scheduler", f"Init failed: {e}")
            return False

    async def run_cycle(self) -> dict:
        """
        Run one trading cycle for all configured symbols.
        Returns a summary dict of what happened.
        """
        if not self._is_running:
            return {"status": "stopped", "message": "Scheduler is not running"}

        if settings.kill_switch:
            return {"status": "killed", "message": "Kill switch is activated"}

        # Check session filter
        session_ok, session_msg = self.session_filter.is_trading_allowed()
        if not session_ok:
            logger.info("Session filter: %s", session_msg)
            return {"status": "session_blocked", "message": session_msg}

        symbols = settings.trading.symbols
        timeframe = settings.trading.timeframe
        summary: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "timeframe": timeframe,
            "session": session_msg,
            "symbol_results": {},
            "errors": [],
        }

        async with async_session() as session:
            try:
                # Get account balance
                account_balance = 10000.0
                if self._authenticated:
                    try:
                        balance_data = await self.client.get_account_balance()
                        account_balance = balance_data.get("balance", 10000.0)
                        await self._record_balance(session, balance_data)
                    except CapitalAPIError as e:
                        logger.warning("Failed to get balance: %s", e)
                        summary["errors"].append(f"Balance fetch: {e}")

                # Auto-optimization check
                if await self.auto_optimizer.should_optimize():
                    opt_df = await self.market_data.get_candles(symbols[0], timeframe, count=200)
                    if opt_df is not None and len(opt_df) >= 60:
                        opt_result = await self.auto_optimizer.optimize(session, opt_df)
                        if opt_result:
                            summary["optimization"] = {
                                "ema_fast": opt_result.ema_fast,
                                "ema_slow": opt_result.ema_slow,
                                "win_rate": opt_result.win_rate,
                                "profit_factor": opt_result.profit_factor,
                            }

                # ML model training check (periodic)
                if self.ml_scorer.enabled and not self.ml_scorer._is_trained:
                    try:
                        train_result = await self.ml_scorer.train(session)
                        if train_result.get("status") == "trained":
                            logger.info("ML model trained: accuracy=%.4f", train_result.get("accuracy_cv", 0))
                    except Exception as e:
                        logger.warning("ML training failed: %s", e)

                # Process each symbol
                for symbol in symbols:
                    try:
                        result = await self._process_symbol(
                            session, symbol, timeframe, account_balance,
                        )
                        summary["symbol_results"][symbol] = result
                    except Exception as e:
                        error_msg = f"Error processing {symbol}: {e}"
                        summary["errors"].append(error_msg)
                        logger.error("%s\n%s", error_msg, traceback.format_exc())

                # Update trailing stops for all symbols
                try:
                    prices = await self.market_data.get_current_prices(symbols)
                    adjusted = await self.trailing_stop_manager.update_trailing_stops(
                        session, prices, atr_values=self._latest_atr,
                    )
                    if adjusted:
                        summary["trailing_adjustments"] = adjusted
                        for adj in adjusted:
                            logger.info(
                                "Trailing SL: %s %s %.5f -> %.5f",
                                adj["direction"], adj["symbol"],
                                adj.get("old_sl", 0), adj["new_sl"],
                            )
                except Exception as e:
                    logger.warning("Trailing stop update failed: %s", e)

                # Check SL/TP for paper positions
                if settings.trading.mode in ("paper", "analysis"):
                    try:
                        prices = await self.market_data.get_current_prices(symbols)
                        closed = await self.position_manager.check_sl_tp(session, prices)
                        summary["positions_closed"] = len(closed)
                        for c in closed:
                            await self.notifier.notify_trade_closed(
                                direction=c.get("direction", ""),
                                symbol=c.get("symbol", ""),
                                pnl=c.get("pnl", 0),
                                result=c.get("result", ""),
                            )
                    except Exception as e:
                        logger.warning("SL/TP check failed: %s", e)

                # Sync positions from API
                if self._authenticated:
                    await self.position_manager.sync_positions_from_api(session)

            except Exception as e:
                error_msg = f"Cycle error: {e}"
                summary["errors"].append(error_msg)
                logger.error("%s\n%s", error_msg, traceback.format_exc())
                await self.notifier.notify_error("scheduler", error_msg)

        return summary

    async def _process_symbol(
        self,
        session: AsyncSession,
        symbol: str,
        timeframe: str,
        account_balance: float,
    ) -> dict:
        """Process a single symbol through the full trading pipeline."""
        result: dict = {
            "signal": None,
            "trade": None,
            "filters": {},
        }

        # News filter check
        news_ok, news_msg = self.news_filter.is_trading_allowed(symbol)
        result["filters"]["news"] = news_msg
        if not news_ok:
            logger.info("News filter blocked %s: %s", symbol, news_msg)
            return result

        # Fetch candles
        df = await self.market_data.get_candles(symbol, timeframe)
        if df is None or len(df) < 60:
            result["error"] = f"Insufficient data ({len(df) if df is not None else 0} candles)"
            return result

        # Get spread
        spread = await self.market_data.get_spread(symbol)

        # Compute and store latest ATR value for trailing stop updates.
        # We calculate ATR directly here because add_all_indicators() returns
        # a copy, so the original df won't have the 'atr' column.
        if len(df) > 0 and all(c in df.columns for c in ("high", "low", "close")):
            atr_series = calculate_atr(df["high"], df["low"], df["close"])
            last_atr = atr_series.iloc[-1]
            if pd.notna(last_atr):
                self._latest_atr[symbol] = float(last_atr)

        # Update portfolio price history for correlation tracking
        if "close" in df.columns:
            self.portfolio_manager.update_price_history(symbol, df["close"])

        # Check open positions
        has_long = await self.position_manager.has_open_long(session, symbol)
        has_short = await self.position_manager.has_open_short(session, symbol)

        # Generate signal
        signal = self.signal_generator.generate(
            df=df, symbol=symbol, timeframe=timeframe, spread=spread,
            has_open_long=has_long, has_open_short=has_short,
        )

        result["signal"] = {
            "direction": signal.direction,
            "close": signal.close_price,
            "ema_fast": signal.ema_fast,
            "ema_slow": signal.ema_slow,
            "rsi": signal.rsi,
            "reasons": signal.reasons,
        }

        logger.info(
            "Signal: %s %s | Close=%.5f EMA_F=%.5f EMA_S=%.5f RSI=%.2f",
            signal.direction, symbol, signal.close_price,
            signal.ema_fast, signal.ema_slow, signal.rsi,
        )

        if signal.direction == "NO_TRADE":
            return result

        # ML scoring
        if self.ml_scorer.enabled:
            ml_score = self.ml_scorer.score_signal(
                ema_fast=signal.ema_fast, ema_slow=signal.ema_slow,
                rsi=signal.rsi, atr=signal.atr,
                spread=spread, direction=signal.direction,
                close_price=signal.close_price,
            )
            result["ml_score"] = ml_score
            if ml_score.get("recommendation") == "strong_avoid":
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"ML scorer: strong_avoid (score={ml_score['score']:.3f})")
                logger.info("ML scorer rejected signal for %s", symbol)
                return result

        # Sentiment check
        if self.sentiment_analyzer.enabled:
            sentiment = await self.sentiment_analyzer.analyze(symbol)
            result["sentiment"] = {
                "score": sentiment.score,
                "label": sentiment.label,
                "confidence": sentiment.confidence,
            }
            favorable, sent_msg = self.sentiment_analyzer.is_sentiment_favorable(
                sentiment, signal.direction,
            )
            if not favorable:
                original_direction = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"Sentiment: {sent_msg}")
                logger.info("Sentiment blocked %s for %s", original_direction, symbol)
                return result

        # AI market analysis (text + chart vision)
        ai_analysis = None
        if (
            signal.direction != "NO_TRADE"
            and settings.ai_analysis_enabled
            and self.ai_analyst.enabled
        ):
            # Run advanced chart analysis
            chart_analysis = await self.ai_analyst.analyze_with_chart(
                symbol=symbol, timeframe=timeframe, df=df,
                signal_direction=signal.direction,
                indicators={
                    "ema_fast": signal.ema_fast, "ema_slow": signal.ema_slow,
                    "rsi": signal.rsi, "atr": signal.atr,
                },
            )

            # Run standard market analysis
            standard_analysis = await self.ai_analyst.analyze_market(
                symbol=symbol, timeframe=timeframe, df=df,
                signal_direction=signal.direction,
                indicators={
                    "ema_fast": signal.ema_fast, "ema_slow": signal.ema_slow,
                    "rsi": signal.rsi, "atr": signal.atr,
                },
                spread=spread, account_balance=account_balance,
            )

            # Combine analyses — use the more conservative recommendation
            if standard_analysis["recommendation"] == "REJECT" or chart_analysis["recommendation"] == "REJECT":
                combined_rec = "REJECT"
            elif standard_analysis["recommendation"] == "HOLD" or chart_analysis["recommendation"] == "HOLD":
                combined_rec = "HOLD"
            else:
                combined_rec = "CONFIRM"

            combined_confidence = min(
                standard_analysis["confidence"], chart_analysis["confidence"]
            )
            ai_analysis = {
                "recommendation": combined_rec,
                "confidence": combined_confidence,
                "analysis": standard_analysis["analysis"],
                "chart_analysis": chart_analysis.get("analysis", ""),
                "risk_notes": standard_analysis["risk_notes"],
            }
            result["ai_analysis"] = ai_analysis

            logger.info(
                "AI analysis for %s %s: %s (confidence: %.1f%%)",
                signal.direction, symbol, combined_rec, combined_confidence,
            )

            if combined_rec == "REJECT":
                original_direction = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"AI REJECTED: {ai_analysis['analysis']}")
                await self.notifier.notify_risk_limit(
                    f"AI rejected {original_direction} on {symbol}: {ai_analysis['analysis']}"
                )
                return result
            elif combined_rec == "HOLD":
                original_direction = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"AI HOLD: {ai_analysis['analysis']}")
                return result

        # Portfolio risk check
        if signal.stop_loss is not None:
            position_risk = account_balance * settings.risk.risk_per_trade
            portfolio_check = await self.portfolio_manager.check_portfolio_risk(
                session, symbol, signal.direction, account_balance, position_risk,
            )
            result["portfolio_risk"] = {
                "is_within_limits": portfolio_check.is_within_limits,
                "message": portfolio_check.message,
                "exposure_pct": portfolio_check.exposure_pct,
            }
            if not portfolio_check.is_within_limits:
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"Portfolio: {portfolio_check.message}")
                logger.info("Portfolio risk blocked trade for %s: %s", symbol, portfolio_check.message)
                return result

        # Execute trade
        trade_result = await self.order_manager.process_signal(
            session, signal, account_balance,
        )
        result["trade"] = trade_result

        if trade_result and trade_result.get("status", "").upper() == "FILLED":
            await self.notifier.notify_trade_opened(
                direction=signal.direction,
                symbol=symbol,
                size=trade_result.get("size", 0),
                entry=trade_result.get("entry", 0),
                sl=trade_result.get("sl"),
                tp=trade_result.get("tp"),
                mode=trade_result.get("mode", ""),
            )

        return result

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
