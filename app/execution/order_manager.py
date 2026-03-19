"""
Order manager module.
Validates signals, places orders, and records results.
"""

import logging
import math
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
        return await self._live_order(session, signal, signal_id, size, account_balance)

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

    @staticmethod
    def _floor_to(value: float, precision: int) -> float:
        """Round value DOWN (toward negative infinity) to given decimal places."""
        factor = 10 ** precision
        return math.floor(value * factor) / factor

    @staticmethod
    def _ceil_to(value: float, precision: int) -> float:
        """Round value UP (toward positive infinity) to given decimal places."""
        factor = 10 ** precision
        return math.ceil(value * factor) / factor

    def _validate_and_adjust_order(
        self,
        direction: str,
        close_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        constraints: dict,
        account_balance: float,
    ) -> tuple[float, float, float, str]:
        """Validate and adjust SL/TP/size against Capital.com market constraints.

        Returns (adjusted_sl, adjusted_tp, adjusted_size, warning_msg).
        The warning_msg is empty if no adjustments were needed.
        """
        bid = constraints["bid"]
        ask = constraints["ask"]
        min_stop_pct = constraints["min_stop_pct"]
        min_stop_points = constraints["min_stop_points"]
        min_deal_size = constraints["min_deal_size"]
        min_size_increment = constraints["min_size_increment"]
        max_deal_size = constraints.get("max_deal_size", 0)

        warnings: list[str] = []
        adj_sl = stop_loss
        adj_tp = take_profit
        adj_size = size
        max_size_by_margin: float = float("inf")  # updated by margin cap below

        # Determine rounding precision based on price magnitude
        ref_price = bid if direction == "BUY" else ask
        if ref_price > 1000:
            precision = 1
        elif ref_price > 10:
            precision = 2
        elif ref_price > 1:
            precision = 3
        else:
            precision = 5

        # Calculate minimum stop distance as an absolute value
        if min_stop_pct > 0:
            min_distance = ref_price * (min_stop_pct / 100.0)
        elif min_stop_points > 0:
            min_distance = min_stop_points
        else:
            min_distance = 0.0

        # Enforce a minimum SL distance of 0.3% of price regardless of ATR.
        # On short timeframes the ATR-based SL is often just a few pips,
        # which gets hit by normal spread/noise before the trade can move.
        min_sl_pct = 0.003  # 0.3% of price
        min_sl_abs = ref_price * min_sl_pct
        current_sl_distance = abs(close_price - adj_sl)
        if current_sl_distance < min_sl_abs:
            if direction == "BUY":
                adj_sl = close_price - min_sl_abs
            else:
                adj_sl = close_price + min_sl_abs
            # Recalculate TP to maintain original risk-reward ratio
            original_distance = abs(close_price - stop_loss)
            if original_distance > 0:
                original_tp_distance = abs(take_profit - close_price)
                rr = original_tp_distance / original_distance
                if direction == "BUY":
                    adj_tp = close_price + (min_sl_abs * rr)
                else:
                    adj_tp = close_price - (min_sl_abs * rr)
            warnings.append(
                f"SL distance {current_sl_distance:.5f} < min {min_sl_abs:.5f} (0.3% of price)"
            )

        # Use the spread as a floor for the buffer — SL must be at least
        # one full spread away from bid/ask to survive price movement
        # between the constraint check and the order placement.
        spread = abs(ask - bid)
        buffer = max(min_distance * 3.0, spread)

        # For BUY: SL must be below bid; for SELL: SL must be above ask
        if direction == "BUY":
            max_allowed_sl = bid - buffer
            if adj_sl > max_allowed_sl:
                original_sl = adj_sl
                adj_sl = max_allowed_sl
                # Recalculate TP to maintain original risk-reward ratio
                original_risk = close_price - original_sl
                new_risk = close_price - adj_sl
                if original_risk > 0:
                    rr_ratio = (adj_tp - close_price) / original_risk
                    adj_tp = close_price + (new_risk * rr_ratio)
                warnings.append(
                    f"SL adjusted {original_sl:.5f}->{adj_sl:.5f} (bid={bid})"
                )
            # Round BUY SL DOWN so it stays further from price
            adj_sl = self._floor_to(adj_sl, precision)
            # Round BUY TP UP for a slightly better target
            adj_tp = self._ceil_to(adj_tp, precision)
        else:  # SELL
            min_allowed_sl = ask + buffer
            if adj_sl < min_allowed_sl:
                original_sl = adj_sl
                adj_sl = min_allowed_sl
                original_risk = original_sl - close_price
                new_risk = adj_sl - close_price
                if original_risk > 0:
                    rr_ratio = (close_price - adj_tp) / original_risk
                    adj_tp = close_price - (new_risk * rr_ratio)
                warnings.append(
                    f"SL adjusted {original_sl:.5f}->{adj_sl:.5f} (ask={ask})"
                )
            # Round SELL SL UP so it stays further from price
            adj_sl = self._ceil_to(adj_sl, precision)
            # Round SELL TP DOWN for a slightly better target
            adj_tp = self._floor_to(adj_tp, precision)

        # Recalculate position size based on (possibly wider) stop
        sl_distance = abs(close_price - adj_sl)
        if sl_distance > 0:
            risk_amount = account_balance * settings.risk.risk_per_trade
            adj_size = risk_amount / sl_distance

        # Hard cap: position notional value must never exceed account balance.
        # This prevents absurd sizes when sl_distance is tiny (e.g. forex pips).
        if ref_price > 0 and adj_size * ref_price > account_balance:
            adj_size = account_balance / ref_price
            warnings.append(
                f"Size capped by notional value (balance={account_balance:.2f})"
            )

        # Cap position size using the instrument's actual margin factor.
        # margin_factor is a percentage: e.g., 50 = 50% margin required (2:1 leverage).
        # Use only 50% of available balance per single trade to leave room for
        # other open positions and unrealised P&L fluctuations.
        margin_factor_pct = constraints.get("margin_factor", 50)
        if ref_price > 0 and margin_factor_pct > 0:
            margin_per_unit = ref_price * (margin_factor_pct / 100.0)
            usable_balance = account_balance * 0.5  # reserve half for other trades
            max_size_by_margin = usable_balance / margin_per_unit
            if adj_size > max_size_by_margin:
                warnings.append(
                    f"Size {adj_size:.2f} exceeds margin cap {max_size_by_margin:.2f} "
                    f"(margin_factor={margin_factor_pct}%)"
                )
                adj_size = max_size_by_margin

        # Cap by exchange max deal size
        if max_deal_size > 0 and adj_size > max_deal_size:
            warnings.append(
                f"Size {adj_size:.2f} exceeds max deal size {max_deal_size}"
            )
            adj_size = max_deal_size

        # Enforce minimum deal size — but never force it above what margin allows.
        # If the margin-capped size is below the exchange minimum, flag it so
        # the caller can skip the order instead of getting a RISK_CHECK rejection.
        if adj_size < min_deal_size:
            if max_size_by_margin < min_deal_size:
                warnings.append(
                    f"Insufficient margin: max_size={adj_size:.4f} < min_deal={min_deal_size}"
                )
                adj_size = 0  # signal to caller: skip this trade
            else:
                warnings.append(
                    f"Size {adj_size:.4f} below min {min_deal_size}, using min"
                )
                adj_size = min_deal_size

        # Round size down to increment
        if min_size_increment > 0 and adj_size > 0:
            adj_size = math.floor(adj_size / min_size_increment) * min_size_increment
            if adj_size < min_deal_size:
                adj_size = min_deal_size

        return adj_sl, adj_tp, adj_size, "; ".join(warnings)

    async def _live_order(
        self,
        session: AsyncSession,
        signal: TradeSignal,
        signal_id: int,
        size: float,
        account_balance: float = 0.0,
    ) -> dict:
        """Place a real order via Capital.com API."""
        # Fetch market constraints and adjust SL/TP/size
        sl = signal.stop_loss
        tp = signal.take_profit
        adj_size = size
        try:
            constraints = await self.client.get_market_constraints(signal.symbol)

            # Use *available* margin (not total balance) so that margin
            # already consumed by open positions is taken into account.
            try:
                acct = await self.client.get_account_balance()
                available_margin = acct.get("available", account_balance)
            except Exception:
                available_margin = account_balance

            sl, tp, adj_size, adj_warnings = self._validate_and_adjust_order(
                direction=signal.direction,
                close_price=signal.close_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                size=size,
                constraints=constraints,
                account_balance=available_margin,
            )
            if adj_warnings:
                logger.info(
                    "Order adjusted for %s %s: %s",
                    signal.direction, signal.symbol, adj_warnings,
                )
            # If size was zeroed out, skip the trade entirely
            if adj_size <= 0:
                logger.warning(
                    "Skipping %s %s: insufficient margin (available=%.2f)",
                    signal.direction, signal.symbol, available_margin,
                )
                return {"status": "skipped", "reason": "insufficient margin"}
        except Exception as e:
            logger.warning(
                "Could not fetch market constraints for %s, using original values: %s",
                signal.symbol, e,
            )

        order = Order(
            signal_id=signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            size=adj_size,
            entry_price=signal.close_price,
            stop_loss=sl,
            take_profit=tp,
            status=OrderStatus.PENDING,
        )
        session.add(order)
        await session.flush()

        try:
            result = await self.client.place_order(
                epic=signal.symbol,
                direction=signal.direction,
                size=adj_size,
                stop_loss=sl,
                take_profit=tp,
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
                            size=adj_size,
                            entry_price=float(confirmation.get("level", signal.close_price)),
                            stop_loss=sl,
                            take_profit=tp,
                            deal_id=deal_id,
                            is_open=True,
                        )
                        session.add(position)
                    else:
                        order.status = OrderStatus.REJECTED
                        reason = confirmation.get("rejectReason", confirmation.get("reason", "Unknown"))
                        order.error_message = f"Rejected: {reason}"
                        logger.warning(
                            "Order REJECTED for %s %s size=%.2f: %s",
                            signal.direction, signal.symbol, adj_size, reason,
                        )
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
                adj_size,
                order.status.value,
                order.deal_id,
            )
            return {
                "status": order.status.value,
                "mode": settings.trading.mode,
                "deal_id": order.deal_id,
                "deal_reference": deal_reference,
                "direction": signal.direction,
                "size": adj_size,
                "entry": signal.close_price,
                "sl": sl,
                "tp": tp,
            }

        except CapitalAPIError as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            await session.commit()
            await self._record_error(session, "order_manager", e)
            logger.error("Order placement failed: %s", e)
            return {"status": "failed", "error": str(e)}
