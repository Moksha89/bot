"""
Position manager module.
Tracks open positions, checks SL/TP, and handles position closing.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.capital_client import CapitalClient, CapitalAPIError
from app.config import settings
from app.db.models import Position, TradeResult, DailyPnL

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages position lifecycle: monitoring, closing, and P&L tracking.
    """

    def __init__(self, client: CapitalClient) -> None:
        self.client = client

    async def get_open_positions(self, session: AsyncSession) -> list[Position]:
        """Get all open positions from the database."""
        result = await session.execute(
            select(Position).where(Position.is_open.is_(True))
        )
        return list(result.scalars().all())

    async def has_open_position(
        self, session: AsyncSession, symbol: str
    ) -> bool:
        """Check if there's an open position for a given symbol."""
        positions = await self.get_open_positions(session)
        return any(p.symbol == symbol for p in positions)

    async def has_open_long(self, session: AsyncSession, symbol: str) -> bool:
        """Check if there's an open long position for a symbol."""
        positions = await self.get_open_positions(session)
        return any(p.symbol == symbol and p.direction == "BUY" for p in positions)

    async def has_open_short(self, session: AsyncSession, symbol: str) -> bool:
        """Check if there's an open short position for a symbol."""
        positions = await self.get_open_positions(session)
        return any(p.symbol == symbol and p.direction == "SELL" for p in positions)

    async def close_position(
        self,
        session: AsyncSession,
        position: Position,
        exit_price: float,
        reason: str = "manual",
    ) -> dict:
        """
        Close a position and record P&L.
        For demo/live mode, also closes via Capital.com API.
        """
        if settings.trading.mode in ("demo", "live") and position.deal_id:
            try:
                await self.client.close_position(position.deal_id)
            except CapitalAPIError as e:
                logger.error("Failed to close position via API: %s", e)
                return {"status": "error", "message": str(e)}

        # Calculate P&L
        if position.direction == "BUY":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        position.exit_price = exit_price
        position.pnl = round(pnl, 2)
        position.is_open = False
        position.closed_at = datetime.now(timezone.utc)

        if pnl > 0:
            position.result = TradeResult.WIN
        elif pnl < 0:
            position.result = TradeResult.LOSS
        else:
            position.result = TradeResult.BREAKEVEN

        # Update daily P&L
        await self._update_daily_pnl(session, pnl)
        await session.commit()

        logger.info(
            "Position closed: %s %s PnL=%.2f result=%s reason=%s",
            position.direction,
            position.symbol,
            pnl,
            position.result.value,
            reason,
        )
        return {
            "status": "closed",
            "symbol": position.symbol,
            "direction": position.direction,
            "pnl": pnl,
            "result": position.result.value,
        }

    async def check_sl_tp(
        self,
        session: AsyncSession,
        current_prices: dict[str, dict[str, float]],
    ) -> list[dict]:
        """
        Check stop loss and take profit for all open positions (paper mode).
        current_prices: {symbol: {"bid": ..., "ask": ...}}
        Returns list of closed position results.
        """
        if settings.trading.mode not in ("paper", "analysis"):
            return []

        positions = await self.get_open_positions(session)
        results = []

        for pos in positions:
            prices = current_prices.get(pos.symbol)
            if not prices:
                continue

            current_price = prices["bid"] if pos.direction == "BUY" else prices["ask"]

            # Check stop loss
            if pos.stop_loss is not None:
                if pos.direction == "BUY" and current_price <= pos.stop_loss:
                    result = await self.close_position(
                        session, pos, pos.stop_loss, "stop_loss"
                    )
                    results.append(result)
                    continue
                elif pos.direction == "SELL" and current_price >= pos.stop_loss:
                    result = await self.close_position(
                        session, pos, pos.stop_loss, "stop_loss"
                    )
                    results.append(result)
                    continue

            # Check take profit
            if pos.take_profit is not None:
                if pos.direction == "BUY" and current_price >= pos.take_profit:
                    result = await self.close_position(
                        session, pos, pos.take_profit, "take_profit"
                    )
                    results.append(result)
                    continue
                elif pos.direction == "SELL" and current_price <= pos.take_profit:
                    result = await self.close_position(
                        session, pos, pos.take_profit, "take_profit"
                    )
                    results.append(result)
                    continue

        return results

    async def _update_daily_pnl(
        self, session: AsyncSession, pnl: float
    ) -> None:
        """Update or create daily P&L record."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = await session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily = result.scalar_one_or_none()

        if daily:
            daily.total_pnl += pnl
            daily.trade_count += 1
            if pnl > 0:
                daily.wins += 1
            elif pnl < 0:
                daily.losses += 1
        else:
            daily = DailyPnL(
                date=today,
                total_pnl=pnl,
                trade_count=1,
                wins=1 if pnl > 0 else 0,
                losses=1 if pnl < 0 else 0,
            )
            session.add(daily)

    async def analyze_position_risk(
        self,
        position: Position,
        current_price: float,
        atr: float,
        account_balance: float,
    ) -> dict:
        """
        Analyze risk for a position (including manually opened ones).
        Returns risk analysis with suggestion to hold/close.
        """
        if position.direction == "BUY":
            unrealized_pnl = (current_price - position.entry_price) * position.size
            distance_from_entry = current_price - position.entry_price
        else:
            unrealized_pnl = (position.entry_price - current_price) * position.size
            distance_from_entry = position.entry_price - current_price

        # Risk as percentage of account
        risk_pct = abs(unrealized_pnl) / account_balance * 100 if account_balance > 0 else 0

        # How many ATRs the price has moved against us
        atr_distance = abs(distance_from_entry) / atr if atr > 0 else 0

        suggestion = "HOLD"
        reasons = []

        # Auto-close if loss exceeds 3% of account
        if unrealized_pnl < 0 and risk_pct > 3.0:
            suggestion = "CLOSE_LOSS"
            reasons.append(f"Loss exceeds 3% of account ({risk_pct:.1f}%)")

        # Auto-close if price moved more than 3 ATRs against us
        elif unrealized_pnl < 0 and atr_distance > 3.0:
            suggestion = "CLOSE_LOSS"
            reasons.append(f"Price moved {atr_distance:.1f} ATRs against position")

        # Suggest taking profit if up more than 2 ATRs
        elif unrealized_pnl > 0 and atr_distance > 2.0:
            suggestion = "CONSIDER_TP"
            reasons.append(f"Price moved {atr_distance:.1f} ATRs in favor, consider taking profit")

        # Warning if approaching danger zone
        elif unrealized_pnl < 0 and risk_pct > 1.5:
            suggestion = "WARNING"
            reasons.append(f"Loss at {risk_pct:.1f}% of account, approaching risk limit")

        if not reasons:
            reasons.append("Position within acceptable risk parameters")

        return {
            "deal_id": position.deal_id,
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "current_price": current_price,
            "unrealized_pnl": round(unrealized_pnl, 2),
            "risk_pct": round(risk_pct, 2),
            "atr_distance": round(atr_distance, 2),
            "suggestion": suggestion,
            "reasons": reasons,
        }

    async def auto_manage_positions(
        self,
        session: AsyncSession,
        current_prices: dict[str, dict[str, float]],
        atr_values: dict[str, float],
        account_balance: float,
    ) -> list[dict]:
        """
        Monitor all open positions (including manually opened ones).
        Auto-close positions that breach risk limits.
        Returns list of actions taken.
        """
        if settings.trading.mode not in ("demo", "live"):
            return []

        positions = await self.get_open_positions(session)
        actions = []

        for pos in positions:
            prices = current_prices.get(pos.symbol)
            if not prices:
                continue

            current_price = prices["bid"] if pos.direction == "BUY" else prices["ask"]
            if current_price <= 0:
                continue
            atr = atr_values.get(pos.symbol, 0)

            analysis = await self.analyze_position_risk(
                pos, current_price, atr, account_balance
            )

            if analysis["suggestion"] == "CLOSE_LOSS":
                logger.warning(
                    "Auto-closing losing position %s %s: %s",
                    pos.direction, pos.symbol, analysis["reasons"],
                )
                result = await self.close_position(
                    session, pos, current_price, "auto_risk_close"
                )
                analysis["action_taken"] = "CLOSED"
                analysis["close_result"] = result
                actions.append(analysis)
            else:
                actions.append(analysis)

        return actions

    async def sync_positions_from_api(self, session: AsyncSession) -> None:
        """
        Sync positions from Capital.com API with local database.

        This does two things:
        1. Import API positions that don't exist locally (e.g., after DB reset)
        2. Close local positions that no longer exist on the API

        Capital.com can return slightly different deal IDs in order confirmation
        vs the positions API (e.g., suffix differs by 1). So we match by both
        exact deal_id AND by symbol+direction+size as a fallback.
        """
        if settings.trading.mode not in ("demo", "live"):
            return

        try:
            api_positions = await self.client.get_positions()
            api_deal_ids = set()
            api_positions_by_key: dict[tuple[str, str, float], str] = {}
            api_full_data: dict[str, dict] = {}
            for p in api_positions:
                pos_data = p.get("position", {})
                market_data = p.get("market", {})
                deal_id = pos_data.get("dealId", "")
                api_deal_ids.add(deal_id)
                api_full_data[deal_id] = p
                # Also index by symbol+direction+size for fuzzy matching
                key = (
                    market_data.get("epic", ""),
                    pos_data.get("direction", ""),
                    float(pos_data.get("size", 0)),
                )
                api_positions_by_key[key] = deal_id

            # Get all local open positions
            local_positions = await self.get_open_positions(session)
            local_deal_ids = {p.deal_id for p in local_positions if p.deal_id}
            local_keys = {(p.symbol, p.direction, p.size) for p in local_positions}

            # 1. Import API positions that don't exist locally
            for deal_id, p in api_full_data.items():
                if deal_id in local_deal_ids:
                    continue
                pos_data = p.get("position", {})
                market_data = p.get("market", {})
                epic = market_data.get("epic", "")
                direction = pos_data.get("direction", "")
                size = float(pos_data.get("size", 0))
                # Check fuzzy match too
                if (epic, direction, size) in local_keys:
                    continue
                # Only import symbols we're configured to trade
                if epic not in settings.trading.symbols:
                    continue
                # Import this position
                entry_price = float(pos_data.get("level", 0))
                new_pos = Position(
                    symbol=epic,
                    direction=direction,
                    entry_price=entry_price,
                    size=size,
                    stop_loss=float(pos_data.get("stopLevel", 0)) or None,
                    take_profit=float(pos_data.get("limitLevel", 0)) or None,
                    deal_id=deal_id,
                    is_open=True,
                    opened_at=datetime.now(timezone.utc),
                )
                session.add(new_pos)
                logger.info(
                    "Imported API position: %s %s size=%.2f deal=%s",
                    direction, epic, size, deal_id,
                )

            # 2. Close local positions that are no longer open on the API
            for pos in local_positions:
                if not pos.deal_id:
                    continue
                # Check exact deal_id match first
                if pos.deal_id in api_deal_ids:
                    continue
                # Fuzzy match: same symbol, direction, size
                fuzzy_key = (pos.symbol, pos.direction, pos.size)
                if fuzzy_key in api_positions_by_key:
                    new_deal_id = api_positions_by_key[fuzzy_key]
                    logger.info(
                        "Position %s matched to API deal %s by symbol/direction/size, updating deal_id",
                        pos.deal_id, new_deal_id,
                    )
                    pos.deal_id = new_deal_id
                    continue
                # No match found — position was truly closed externally
                logger.info(
                    "Position %s closed externally, updating local DB",
                    pos.deal_id,
                )
                pos.is_open = False
                pos.closed_at = datetime.now(timezone.utc)
                pos.result = TradeResult.EXTERNAL_CLOSE

            await session.commit()
        except CapitalAPIError as e:
            logger.error("Failed to sync positions: %s", e)
