"""
Order manager module.
Validates signals, places orders, and records results.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.capital_client import CapitalClient, CapitalAPIError
from app.config import settings
from app.db.models import Order, OrderStatus, Position, Signal, SignalDirection, ErrorLog
from app.execution.risk_manager import RiskManager
from app.strategy.signals import TradeSignal

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the full lifecycle of order placement:
    signal validation -> risk check -> order placement -> confirmation -> recording.
    """

    def __init__(
        self,
        client: CapitalClient,
        risk_manager: RiskManager,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager

    async def _record_signal(
        self, session: AsyncSession, signal: TradeSignal, acted_on: bool
    ) -> int:
        """Record a signal to the database."""
        direction_map = {
            "BUY": SignalDirection.BUY,
            "SELL": SignalDirection.SELL,
            "NO_TRADE": SignalDirection.NO_TRADE,
        }
        db_signal = Signal(
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            direction=direction_map.get(signal.direction, SignalDirection.NO_TRADE),
            ema_fast=signal.ema_fast,
            ema_slow=signal.ema_slow,
            rsi=signal.rsi,
            close_price=signal.close_price,
            prev_high=signal.prev_high,
            prev_low=signal.prev_low,
            spread=signal.spread,
            reason="; ".join(signal.reasons),
            acted_on=acted_on,
        )
        session.add(db_signal)
        await session.flush()
        return db_signal.id

    async def _record_error(
        self, session: AsyncSession, module: str, error: Exception
    ) -> None:
        """Record an error to the database."""
        error_log = ErrorLog(
            module=module,
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
        )
        session.add(error_log)
        await session.commit()

    async def process_signal(
        self,
        session: AsyncSession,
        signal: TradeSignal,
        account_balance: float,
    ) -> Optional[dict]:
        """
        Process a trading signal through the full pipeline.
        Returns order result dict or None if rejected/no-trade.
        """
        # Record signal regardless
        if signal.direction == "NO_TRADE":
            await self._record_signal(session, signal, acted_on=False)
            await session.commit()
            logger.info("NO_TRADE signal for %s, skipping", signal.symbol)
            return None

        # Check trading mode
        if settings.trading.mode == "analysis":
            await self._record_signal(session, signal, acted_on=False)
            await session.commit()
            logger.info("Analysis mode: signal recorded but not executed")
            return None

        # Run risk checks
        approved, risk_messages = await self.risk_manager.validate_trade(
            session, signal.symbol, signal.direction, account_balance
        )
        if not approved:
            await self._record_signal(session, signal, acted_on=False)
            await session.commit()
            logger.warning(
                "Signal rejected by risk manager: %s", "; ".join(risk_messages)
            )
            return {"status": "rejected", "reasons": risk_messages}

        # Calculate position size
        if signal.stop_loss is None:
            logger.error("Signal has no stop loss, cannot calculate position size")
            await self._record_signal(session, signal, acted_on=False)
            await session.commit()
            return None

        signal_id = await self._record_signal(session, signal, acted_on=True)

        size = self.risk_manager.calculate_position_size(
            account_balance, signal.close_price, signal.stop_loss
        )

        # Paper mode: simulate order
        if settings.trading.mode == "paper":
            return await self._paper_order(session, signal, signal_id, size)

        # Demo/Live: place real order
        return await self._live_order(session, signal, signal_id, size)

    async def _paper_order(
        self,
        session: AsyncSession,
        signal: TradeSignal,
        signal_id: int,
        size: float,
    ) -> dict:
        """Simulate a paper trade."""
        order = Order(
            signal_id=signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=signal.close_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.FILLED,
            deal_id=f"PAPER-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            deal_reference=f"PAPER-REF-{signal_id}",
        )
        session.add(order)

        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=signal.close_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            deal_id=order.deal_id,
            is_open=True,
        )
        session.add(position)
        await session.commit()

        logger.info(
            "Paper trade placed: %s %s size=%.2f entry=%.5f SL=%.5f TP=%.5f",
            signal.direction,
            signal.symbol,
            size,
            signal.close_price,
            signal.stop_loss,
            signal.take_profit,
        )
        return {
            "status": "filled",
            "mode": "paper",
            "deal_id": order.deal_id,
            "direction": signal.direction,
            "size": size,
            "entry": signal.close_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
        }

    async def _live_order(
        self,
        session: AsyncSession,
        signal: TradeSignal,
        signal_id: int,
        size: float,
    ) -> dict:
        """Place a real order via Capital.com API."""
        order = Order(
            signal_id=signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            size=size,
            entry_price=signal.close_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=OrderStatus.PENDING,
        )
        session.add(order)
        await session.flush()

        try:
            result = await self.client.place_order(
                epic=signal.symbol,
                direction=signal.direction,
                size=size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            deal_reference = result.get("dealReference", "")
            order.deal_reference = deal_reference

            # Confirm order
            if deal_reference:
                try:
                    confirmation = await self.client.confirm_order(deal_reference)
                    deal_id = confirmation.get("dealId", "")
                    deal_status = confirmation.get("dealStatus", "")
                    order.deal_id = deal_id

                    if deal_status == "ACCEPTED":
                        order.status = OrderStatus.FILLED
                        position = Position(
                            symbol=signal.symbol,
                            direction=signal.direction,
                            size=size,
                            entry_price=float(confirmation.get("level", signal.close_price)),
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            deal_id=deal_id,
                            is_open=True,
                        )
                        session.add(position)
                    else:
                        order.status = OrderStatus.REJECTED
                        reason = confirmation.get("reason", "Unknown")
                        order.error_message = f"Rejected: {reason}"
                except CapitalAPIError as e:
                    order.status = OrderStatus.FAILED
                    order.error_message = f"Confirmation failed: {e}"
                    await self._record_error(session, "order_manager", e)
            else:
                order.status = OrderStatus.FAILED
                order.error_message = "No deal reference returned by API"
                logger.error(
                    "Order placed but no deal reference returned for %s %s",
                    signal.direction, signal.symbol,
                )

            await session.commit()

            logger.info(
                "Order placed: %s %s size=%.2f status=%s deal=%s",
                signal.direction,
                signal.symbol,
                size,
                order.status.value,
                order.deal_id,
            )
            return {
                "status": order.status.value,
                "mode": settings.trading.mode,
                "deal_id": order.deal_id,
                "deal_reference": deal_reference,
                "direction": signal.direction,
                "size": size,
                "entry": signal.close_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
            }

        except CapitalAPIError as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            await session.commit()
            await self._record_error(session, "order_manager", e)
            logger.error("Order placement failed: %s", e)
            return {"status": "failed", "error": str(e)}
