"""
Scheduler service.
Runs the strategy loop at configured intervals, manages the trading cycle.
Supports multi-symbol trading, session/news filters, trailing stops,
ML scoring, sentiment analysis, and auto-optimization.
"""

import asyncio
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
from app.strategy.indicators import calculate_atr, calculate_ema
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
        self._cycle_lock = asyncio.Lock()
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
                breakeven_r=settings.trailing_stop_breakeven_r,
            ),
            client=self.client,
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
        # Cache for higher-timeframe trend per symbol
        self._htf_trend: dict[str, str] = {}  # symbol -> "up" / "down" / "flat"
        # Daily trade counter & recovery mode
        self._daily_trades: int = 0
        self._daily_losses: int = 0
        self._daily_wins: int = 0
        self._daily_pnl: float = 0.0
        self._last_reset_date: str = ""
        self._recovery_mode: bool = False
        # Symbol momentum scores for ranking (updated each cycle)
        self._symbol_momentum: dict[str, float] = {}

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

            # Sync existing positions from API so we don't place duplicates
            if settings.trading.mode in ("demo", "live"):
                async with async_session() as session:
                    await self.position_manager.sync_positions_from_api(session)
                    logger.info("Synced existing positions from Capital.com on startup")

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
        Uses a lock to prevent concurrent cycles (manual trigger vs scheduled).
        """
        if self._cycle_lock.locked():
            return {"status": "busy", "message": "Another cycle is already running"}

        async with self._cycle_lock:
            return await self._run_cycle_inner()

    async def _run_cycle_inner(self) -> dict:
        """Inner cycle logic, always called under lock."""
        if not self._is_running:
            return {"status": "stopped", "message": "Scheduler is not running"}

        if settings.kill_switch:
            return {"status": "killed", "message": "Kill switch is activated"}

        # Check session filter (only blocks new signal generation, not position monitoring)
        session_ok, session_msg = self.session_filter.is_trading_allowed()
        if not session_ok:
            logger.info("Session filter: %s", session_msg)

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

        # Reset daily counters at midnight UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_reset_date:
            self._daily_trades = 0
            self._daily_losses = 0
            self._daily_wins = 0
            self._daily_pnl = 0.0
            self._recovery_mode = False
            self._last_reset_date = today
            logger.info("Daily counters reset for %s", today)

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

                # Update daily P&L from closed positions in DB
                try:
                    await self._update_daily_pnl(session)
                except Exception as e:
                    logger.warning("Daily P&L update failed: %s", e)

                # Check if daily profit target reached — stop trading
                if self._daily_pnl > 0 and account_balance > 0:
                    daily_pnl_pct = (self._daily_pnl / account_balance) * 100
                    if daily_pnl_pct >= settings.daily_profit_target_pct:
                        logger.info(
                            "Daily profit target reached: %.1f%% (target %.1f%%). Stopping trades.",
                            daily_pnl_pct, settings.daily_profit_target_pct,
                        )
                        summary["status"] = "profit_target_reached"
                        summary["daily_pnl_pct"] = daily_pnl_pct
                        return summary

                # Check daily trade limit (20 normal + 10 recovery)
                max_trades = settings.max_daily_trades
                if self._recovery_mode:
                    max_trades += settings.recovery_extra_trades
                    logger.info(
                        "Recovery mode active: allowing %d extra trades (total %d)",
                        settings.recovery_extra_trades, max_trades,
                    )

                # Enter recovery mode if we have net losses from today's trades
                if self._daily_losses > self._daily_wins and self._daily_trades >= 5:
                    if not self._recovery_mode:
                        self._recovery_mode = True
                        logger.info(
                            "Entering RECOVERY MODE: %d losses vs %d wins, P&L=%.2f",
                            self._daily_losses, self._daily_wins, self._daily_pnl,
                        )

                if self._daily_trades >= max_trades:
                    logger.info(
                        "Daily trade limit reached: %d/%d (recovery=%s)",
                        self._daily_trades, max_trades, self._recovery_mode,
                    )
                    summary["status"] = "daily_limit_reached"
                    summary["daily_trades"] = self._daily_trades
                    # Still process trailing stops and position management below

                summary["daily_stats"] = {
                    "trades": self._daily_trades,
                    "wins": self._daily_wins,
                    "losses": self._daily_losses,
                    "pnl": round(self._daily_pnl, 2),
                    "recovery_mode": self._recovery_mode,
                    "max_trades": max_trades,
                }

                # Auto-optimization check
                if await self.auto_optimizer.should_optimize():
                    opt_df = await self.market_data.get_candles(symbols[0], timeframe, count=200)
                    if opt_df is not None and len(opt_df) >= 60:
                        opt_result = await self.auto_optimizer.optimize(session, opt_df)
                        if opt_result:
                            self.signal_generator.ema_fast = opt_result.ema_fast
                            self.signal_generator.ema_slow = opt_result.ema_slow
                            self.signal_generator.rsi_buy_min = opt_result.rsi_buy_min
                            self.signal_generator.rsi_buy_max = opt_result.rsi_buy_max
                            self.signal_generator.rsi_sell_min = opt_result.rsi_sell_min
                            self.signal_generator.rsi_sell_max = opt_result.rsi_sell_max
                            self.signal_generator.sl_atr_mult = opt_result.sl_atr_multiplier
                            self.signal_generator.tp_rr = opt_result.tp_risk_reward
                            logger.info(
                                "Applied optimized params: EMA=%d/%d RSI_buy=[%.0f,%.0f] RSI_sell=[%.0f,%.0f]",
                                opt_result.ema_fast, opt_result.ema_slow,
                                opt_result.rsi_buy_min, opt_result.rsi_buy_max,
                                opt_result.rsi_sell_min, opt_result.rsi_sell_max,
                            )
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

                # Update higher-timeframe trends for MTF confirmation
                if settings.mtf_enabled:
                    try:
                        await self._update_htf_trends(symbols)
                        summary["htf_trends"] = dict(self._htf_trend)
                    except Exception as e:
                        logger.warning("HTF trend update failed: %s", e)

                # Rank symbols by momentum to prioritize best trending ones
                ranked_symbols = await self._rank_symbols_by_momentum(
                    symbols, timeframe,
                )
                summary["symbol_ranking"] = [
                    {"symbol": s, "score": round(self._symbol_momentum.get(s, 0), 3)}
                    for s in ranked_symbols[:10]
                ]

                # Process symbols (only if session allows AND under daily limit)
                trades_this_cycle = 0
                if session_ok and self._daily_trades < max_trades:
                    for symbol in ranked_symbols:
                        if self._daily_trades >= max_trades:
                            logger.info("Daily trade limit %d reached mid-cycle", max_trades)
                            break
                        try:
                            result = await self._process_symbol(
                                session, symbol, timeframe, account_balance,
                            )
                            summary["symbol_results"][symbol] = result
                            # Track if a trade was placed
                            if result.get("trade") and result["trade"].get("status", "").upper() == "FILLED":
                                self._daily_trades += 1
                                trades_this_cycle += 1
                        except Exception as e:
                            error_msg = f"Error processing {symbol}: {e}"
                            summary["errors"].append(error_msg)
                            logger.error("%s\n%s", error_msg, traceback.format_exc())
                else:
                    summary["status"] = summary.get("status", "session_blocked")

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

                # Time-based exit: close stale trades that haven't moved
                if settings.stale_trade_exit_enabled:
                    try:
                        stale_closed = await self._close_stale_trades(
                            session, account_balance,
                        )
                        if stale_closed:
                            summary["stale_exits"] = stale_closed
                            for sc in stale_closed:
                                await self.notifier.notify_risk_limit(
                                    f"Stale exit: {sc['direction']} {sc['symbol']} "
                                    f"closed after {sc['minutes_held']:.0f}min (P&L={sc['pnl']:.2f})"
                                )
                    except Exception as e:
                        logger.warning("Stale trade exit failed: %s", e)

                # Auto-manage positions (risk analysis + auto-close losers)
                if self._authenticated:
                    try:
                        prices = await self.market_data.get_current_prices(symbols)
                        risk_actions = await self.position_manager.auto_manage_positions(
                            session, prices, self._latest_atr, account_balance,
                        )
                        summary["position_risk"] = risk_actions
                        for action in risk_actions:
                            if action.get("action_taken") == "CLOSED":
                                await self.notifier.notify_risk_limit(
                                    f"Auto-closed {action['direction']} {action['symbol']}: "
                                    f"P&L={action['unrealized_pnl']:.2f} ({action['reasons'][0]})"
                                )
                    except Exception as e:
                        logger.warning("Position risk management failed: %s", e)

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

        # Generate signal using basic EMA crossover first
        signal = self.signal_generator.generate(
            df=df, symbol=symbol, timeframe=timeframe, spread=spread,
            has_open_long=has_long, has_open_short=has_short,
        )

        # If basic signal is NO_TRADE, try multi-strategy selector
        # (scalping, swing, breakout, mean-reversion, EMA crossover)
        if signal.direction == "NO_TRADE":
            from app.strategy.indicators import add_all_indicators
            from app.strategy.rules import MarketSnapshot
            df_ind = add_all_indicators(
                df,
                ema_fast=self.signal_generator.ema_fast,
                ema_slow=self.signal_generator.ema_slow,
                rsi_period=self.signal_generator.rsi_period,
            )
            if len(df_ind) > self.signal_generator.ema_slow + 1:
                latest = df_ind.iloc[-1]
                prev = df_ind.iloc[-2]
                snapshot = MarketSnapshot(
                    close=float(latest["close"]),
                    ema_fast=float(latest["ema_fast"]),
                    ema_slow=float(latest["ema_slow"]),
                    rsi=float(latest["rsi"]),
                    atr=float(latest["atr"]),
                    prev_high=float(prev["high"]),
                    prev_low=float(prev["low"]),
                    spread=spread,
                    has_open_long=has_long,
                    has_open_short=has_short,
                )
                strat_result = self.strategy_selector.evaluate_all(
                    df_ind, snapshot,
                    sl_atr_mult=self.signal_generator.sl_atr_mult,
                    tp_rr=self.signal_generator.tp_rr,
                    rsi_buy_min=self.signal_generator.rsi_buy_min,
                    rsi_buy_max=self.signal_generator.rsi_buy_max,
                    rsi_sell_min=self.signal_generator.rsi_sell_min,
                    rsi_sell_max=self.signal_generator.rsi_sell_max,
                    max_spread=self.signal_generator.max_spread,
                )
                if strat_result.direction != "NO_TRADE" and strat_result.stop_loss is not None and strat_result.take_profit is not None:
                    # Override signal with the multi-strategy result
                    signal.direction = strat_result.direction
                    signal.stop_loss = strat_result.stop_loss
                    signal.take_profit = strat_result.take_profit
                    signal.reasons = strat_result.reasons
                    logger.info(
                        "Multi-strategy override: %s %s via %s (conf=%.3f)",
                        strat_result.direction, symbol,
                        strat_result.strategy.value, strat_result.confidence,
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
            # Record NO_TRADE signal to DB so it appears on dashboard
            await self.order_manager.process_signal(session, signal, account_balance)
            return result

        # Multi-timeframe confirmation: require higher-TF trend to agree
        if settings.mtf_enabled and signal.direction in ("BUY", "SELL"):
            htf_trend = self._htf_trend.get(symbol, "flat")
            blocked = False
            if signal.direction == "BUY" and htf_trend == "down":
                blocked = True
            elif signal.direction == "SELL" and htf_trend == "up":
                blocked = True

            if blocked:
                original = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(
                    f"MTF filter: {original} blocked, higher-TF trend is {htf_trend}"
                )
                logger.info(
                    "MTF filter blocked %s %s (HTF trend=%s)",
                    original, symbol, htf_trend,
                )
                await self.order_manager.process_signal(session, signal, account_balance)
                result["filters"]["mtf"] = f"Blocked: {original} vs HTF {htf_trend}"
                return result
            result["filters"]["mtf"] = f"Passed: {signal.direction} aligns with HTF {htf_trend}"

        # Correlation filter: avoid opening same-direction trades on highly correlated symbols
        if signal.direction in ("BUY", "SELL"):
            corr_blocked, corr_msg = await self._check_correlation_filter(
                session, symbol, signal.direction,
            )
            if corr_blocked:
                original = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"Correlation filter: {corr_msg}")
                logger.info("Correlation filter blocked %s %s: %s", original, symbol, corr_msg)
                await self.order_manager.process_signal(session, signal, account_balance)
                result["filters"]["correlation"] = corr_msg
                return result

        # ML scoring — require score > 0.6 for high-analysis trades
        # Only gate on ML score when the model is actually trained;
        # an untrained model always returns 0.5, which would deadlock trading.
        if self.ml_scorer.enabled and self.ml_scorer._is_trained:
            ml_score = self.ml_scorer.score_signal(
                ema_fast=signal.ema_fast, ema_slow=signal.ema_slow,
                rsi=signal.rsi, atr=signal.atr,
                spread=spread, direction=signal.direction,
                close_price=signal.close_price,
            )
            result["ml_score"] = ml_score
            score_val = ml_score.get("score", 0)
            if score_val < 0.6:
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"ML scorer: low confidence (score={score_val:.3f} < 0.6)")
                logger.info("ML scorer rejected signal for %s (score=%.3f < 0.6)", symbol, score_val)
                await self.order_manager.process_signal(session, signal, account_balance)
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
                await self.order_manager.process_signal(session, signal, account_balance)
                return result

        # AI market analysis (single call to reduce latency — 27 symbols × 3s = 81s)
        ai_analysis = None
        if (
            signal.direction != "NO_TRADE"
            and settings.ai_analysis_enabled
            and self.ai_analyst.enabled
        ):
            # Run standard market analysis only (skip chart vision to halve latency)
            standard_analysis = await self.ai_analyst.analyze_market(
                symbol=symbol, timeframe=timeframe, df=df,
                signal_direction=signal.direction,
                indicators={
                    "ema_fast": signal.ema_fast, "ema_slow": signal.ema_slow,
                    "rsi": signal.rsi, "atr": signal.atr,
                },
                spread=spread, account_balance=account_balance,
            )

            combined_rec = standard_analysis["recommendation"]
            combined_confidence = standard_analysis["confidence"]
            ai_analysis = {
                "recommendation": combined_rec,
                "confidence": combined_confidence,
                "analysis": standard_analysis["analysis"],
                "risk_notes": standard_analysis["risk_notes"],
            }
            result["ai_analysis"] = ai_analysis

            logger.info(
                "AI analysis for %s %s: %s (confidence: %.1f%%)",
                signal.direction, symbol, combined_rec, combined_confidence,
            )

            # Only block trades the AI explicitly rejects (grade D)
            if combined_rec == "REJECT" and combined_confidence < 35:
                original_direction = signal.direction
                signal.direction = "NO_TRADE"
                signal.reasons.append(f"AI REJECTED (conf={combined_confidence:.0f}%): {ai_analysis['analysis']}")
                await self.notifier.notify_risk_limit(
                    f"AI rejected {original_direction} on {symbol}: {ai_analysis['analysis']}"
                )
                await self.order_manager.process_signal(session, signal, account_balance)
                return result
            # CONFIRM or HOLD — proceed with trade (HOLD = mixed signals, still tradeable)
            logger.info(
                "AI approved %s %s: %s (conf=%.1f%%)",
                signal.direction, symbol, combined_rec, combined_confidence,
            )

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
                await self.order_manager.process_signal(session, signal, account_balance)
                return result

        # Adaptive position sizing: adjust risk based on recent performance
        effective_balance = account_balance
        if settings.adaptive_sizing_enabled:
            effective_balance = await self._get_adaptive_balance(session, account_balance)
            result["adaptive_balance"] = effective_balance

        # Execute trade
        trade_result = await self.order_manager.process_signal(
            session, signal, effective_balance,
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

    # ---- Symbol ranking by momentum ----

    async def _rank_symbols_by_momentum(
        self, symbols: list[str], timeframe: str,
    ) -> list[str]:
        """Rank symbols by trend strength and momentum.

        Scores each symbol based on:
        - EMA20/EMA50 separation (trend strength)
        - RSI momentum (away from 50 = stronger)
        - ATR relative to price (volatility = opportunity)

        Returns symbols sorted best-first so the bot prioritises
        the strongest trending markets.
        """
        scores: dict[str, float] = {}
        for symbol in symbols:
            try:
                df = await self.market_data.get_candles(symbol, timeframe, count=60)
                if df is None or len(df) < 52:
                    scores[symbol] = 0.0
                    continue
                closes = df["close"].astype(float)
                ema20 = calculate_ema(closes, 20)
                ema50 = calculate_ema(closes, 50)
                e20 = float(ema20.iloc[-1])
                e50 = float(ema50.iloc[-1])
                price = float(closes.iloc[-1])

                # Trend strength: EMA separation as % of price
                trend_score = abs(e20 - e50) / e50 * 100 if e50 != 0 else 0

                # Momentum: RSI distance from 50 (further = stronger move)
                from app.strategy.indicators import calculate_rsi
                rsi_series = calculate_rsi(closes, 14)
                rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50
                momentum_score = abs(rsi_val - 50) / 50  # 0-1 scale

                # Volatility: ATR as % of price (higher = more opportunity)
                atr_series = calculate_atr(df["high"], df["low"], df["close"])
                atr_val = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0
                vol_score = (atr_val / price * 100) if price > 0 else 0

                # Combined score (weighted)
                total = trend_score * 0.4 + momentum_score * 0.3 + vol_score * 0.3
                scores[symbol] = total
            except Exception as exc:
                logger.warning("Symbol ranking failed for %s: %s", symbol, exc)
                scores[symbol] = 0.0

        self._symbol_momentum = scores

        # Sort by score descending — best trending symbols first
        ranked = sorted(symbols, key=lambda s: scores.get(s, 0), reverse=True)
        top5 = [(s, round(scores.get(s, 0), 3)) for s in ranked[:5]]
        logger.info("Symbol ranking top 5: %s", top5)
        return ranked

    # ---- Daily P&L tracking from closed positions ----

    async def _update_daily_pnl(self, session: AsyncSession) -> None:
        """Recalculate daily P&L from positions closed today."""
        from sqlalchemy import select as sa_select
        from app.db.models import Position

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        result = await session.execute(
            sa_select(Position).where(
                Position.is_open.is_(False),
                Position.closed_at >= today_start,
            )
        )
        closed_today = result.scalars().all()

        pnl = 0.0
        wins = 0
        losses = 0
        for pos in closed_today:
            pos_pnl = pos.pnl or 0.0
            pnl += pos_pnl
            if pos_pnl > 0:
                wins += 1
            elif pos_pnl < 0:
                losses += 1

        self._daily_pnl = pnl
        self._daily_wins = wins
        self._daily_losses = losses
        self._daily_trades = len(closed_today)
        logger.info(
            "Daily P&L update: %.2f (%d trades, %d wins, %d losses)",
            pnl, len(closed_today), wins, losses,
        )

    # ---- Multi-timeframe confirmation helpers ----

    async def _update_htf_trends(self, symbols: list[str]) -> None:
        """Fetch higher-timeframe candles and determine trend for each symbol.

        Uses EMA20 vs EMA50 on the higher TF to classify trend as
        up / down / flat.  Called once per cycle so we don't re-fetch
        per symbol.
        """
        htf = settings.mtf_timeframe  # e.g. "HOUR"
        for symbol in symbols:
            try:
                df = await self.market_data.get_candles(symbol, htf, count=60)
                if df is None or len(df) < 52:
                    self._htf_trend[symbol] = "flat"
                    continue
                closes = df["close"].astype(float)
                ema20 = calculate_ema(closes, 20)
                ema50 = calculate_ema(closes, 50)
                e20 = float(ema20.iloc[-1])
                e50 = float(ema50.iloc[-1])
                sep = (e20 - e50) / e50 if e50 != 0 else 0
                if sep > 0.005:
                    self._htf_trend[symbol] = "up"
                elif sep < -0.005:
                    self._htf_trend[symbol] = "down"
                else:
                    self._htf_trend[symbol] = "flat"
            except Exception as exc:
                logger.warning("HTF trend fetch failed for %s: %s", symbol, exc)
                self._htf_trend.setdefault(symbol, "flat")

    # ---- Correlation filter ----

    # Predefined correlation groups — symbols within the same group tend to
    # move together, so opening the same-direction trade on multiple members
    # is effectively doubling the same bet.
    _CORRELATION_GROUPS: list[set[str]] = [
        # Major USD forex pairs (all move inversely to USD)
        {"EURUSD", "GBPUSD", "AUDUSD"},
        # USD-quote pairs (move with USD)
        {"USDJPY", "USDCAD"},
        # US tech — highly correlated
        {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"},
        # US indices
        {"US100", "US500"},
        # Precious metals
        {"GOLD", "SILVER"},
        # Crypto large-cap
        {"BTCUSD", "ETHUSD"},
        # Crypto alt-coins (follow BTC/ETH)
        {"XRPUSD", "SOLUSD", "DOGEUSD", "ADAUSD", "DOTUSD", "LINKUSD", "LTCUSD"},
    ]

    async def _check_correlation_filter(
        self,
        session: AsyncSession,
        symbol: str,
        direction: str,
    ) -> tuple[bool, str]:
        """Return (blocked, message) if a correlated symbol already has an
        open position in the same direction."""
        # Find which group(s) this symbol belongs to
        groups = [g for g in self._CORRELATION_GROUPS if symbol in g]
        if not groups:
            return False, "No correlation group"

        open_positions = await self.position_manager.get_open_positions(session)
        for grp in groups:
            for pos in open_positions:
                if (
                    pos.symbol != symbol
                    and pos.symbol in grp
                    and pos.direction == direction
                ):
                    return True, (
                        f"Already have {pos.direction} {pos.symbol} "
                        f"(same correlation group as {symbol})"
                    )
        return False, "Correlation check passed"

    # ---- Time-based (stale) trade exit ----

    async def _close_stale_trades(
        self,
        session: AsyncSession,
        account_balance: float,
    ) -> list[dict]:
        """Close trades that have been open longer than stale_trade_minutes
        and are near breakeven (within 0.3% of entry).  This frees up
        capital for better setups instead of letting dead trades sit."""
        from datetime import timedelta

        max_minutes = settings.stale_trade_minutes
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_minutes)
        positions = await self.position_manager.get_open_positions(session)
        results: list[dict] = []

        prices = await self.market_data.get_current_prices(
            [p.symbol for p in positions],
        )

        for pos in positions:
            opened = pos.opened_at
            if opened is None:
                continue
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if opened > cutoff:
                continue  # not stale yet

            price_data = prices.get(pos.symbol)
            if not price_data:
                continue
            current_price = (
                price_data["bid"] if pos.direction == "BUY" else price_data["ask"]
            )
            if current_price <= 0 or not pos.entry_price:
                continue

            # Only exit if the trade is near breakeven (within 0.3% of entry)
            # — don't close winners that are running or losers already past SL.
            pct_move = abs(current_price - pos.entry_price) / pos.entry_price
            if pct_move > 0.003:
                continue  # trade has moved, let SL/TP or trailing handle it

            minutes_held = (datetime.now(timezone.utc) - opened).total_seconds() / 60
            close_result = await self.position_manager.close_position(
                session, pos, current_price, "stale_exit",
            )
            results.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "minutes_held": minutes_held,
                "pnl": close_result.get("pnl", 0),
            })
            logger.info(
                "Stale exit: %s %s after %.0f min (pnl=%.2f)",
                pos.direction, pos.symbol, minutes_held, close_result.get("pnl", 0),
            )
        return results

    # ---- Adaptive position sizing ----

    async def _get_adaptive_balance(
        self,
        session: AsyncSession,
        account_balance: float,
    ) -> float:
        """Return an effective balance for position sizing that scales
        risk down after losses and up after wins.

        Logic (simplified Kelly):
        - Look at the last 10 closed trades.
        - If win rate >= 60%: use 120% of balance (slightly larger size).
        - If win rate 40-60%: use 100% (normal).
        - If win rate < 40%: use 70% of balance (smaller size to protect capital).
        - If fewer than 5 trades: use 80% (conservative until we have data).
        """
        from sqlalchemy import select as sa_select
        from app.db.models import Position, TradeResult

        result = await session.execute(
            sa_select(Position)
            .where(
                Position.is_open.is_(False),
                Position.result.in_([TradeResult.WIN, TradeResult.LOSS]),
            )
            .order_by(Position.closed_at.desc())
            .limit(10)
        )
        recent = result.scalars().all()

        if len(recent) < 5:
            # Not enough data — be conservative
            effective = account_balance * 0.80
            logger.debug("Adaptive sizing: <5 trades, using 80%% of balance")
            return effective

        wins = sum(1 for t in recent if t.result == TradeResult.WIN)
        win_rate = wins / len(recent)

        if win_rate >= 0.6:
            multiplier = 1.20
        elif win_rate >= 0.4:
            multiplier = 1.0
        else:
            multiplier = 0.70

        effective = account_balance * multiplier
        logger.info(
            "Adaptive sizing: %d/%d wins (%.0f%%), multiplier=%.2f, effective=%.2f",
            wins, len(recent), win_rate * 100, multiplier, effective,
        )
        return effective

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
