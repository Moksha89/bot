"""
Trailing stop loss module.
Dynamically adjusts stop loss as price moves in favor to lock in profits.
Supports multiple trailing modes: fixed distance, ATR-based, and percentage-based.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Position

logger = logging.getLogger(__name__)


class TrailingMode(str, Enum):
    FIXED = "fixed"       # Fixed pip distance
    ATR = "atr"           # ATR-based distance
    PERCENT = "percent"   # Percentage-based


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop loss."""
    enabled: bool = True
    mode: TrailingMode = TrailingMode.ATR
    # Activate trailing after price moves this many R multiples in profit
    activation_r: float = 1.0
    # Fixed mode: trailing distance in price units
    fixed_distance: float = 0.0
    # ATR mode: multiplier for ATR
    atr_multiplier: float = 1.0
    # Percent mode: trailing by this percentage
    trail_percent: float = 0.5
    # Step size: minimum price move before adjusting SL
    step_size: float = 0.0


class TrailingStopManager:
    """
    Manages trailing stop losses for open positions.
    Called each trading cycle to adjust SL levels.
    """

    def __init__(self, config: TrailingStopConfig | None = None, client: "CapitalClient | None" = None) -> None:
        self.config = config or TrailingStopConfig()
        self.client = client
        # Track highest/lowest price seen for each position
        self._peak_prices: dict[int, float] = {}

    async def update_trailing_stops(
        self,
        session: AsyncSession,
        current_prices: dict[str, dict[str, float]],
        atr_values: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Update trailing stops for all open positions.
        Returns list of positions that had their SL adjusted.
        """
        if not self.config.enabled:
            return []

        result = await session.execute(
            select(Position).where(Position.is_open.is_(True))
        )
        positions = result.scalars().all()
        adjusted: list[dict] = []

        for pos in positions:
            prices = current_prices.get(pos.symbol)
            if not prices:
                continue

            current_price = prices.get("bid", 0) if pos.direction == "BUY" else prices.get("ask", 0)
            if current_price <= 0:
                continue

            atr = (atr_values or {}).get(pos.symbol, 0)
            adjustment = self._calculate_adjustment(pos, current_price, atr)

            if adjustment:
                old_sl = pos.stop_loss
                new_sl = adjustment["new_sl"]
                pos.stop_loss = new_sl

                # Sync to Capital.com API in demo/live mode
                api_synced = False
                if self.client and pos.deal_id and settings.trading.mode in ("demo", "live"):
                    try:
                        await self.client.update_position(pos.deal_id, stop_loss=new_sl)
                        api_synced = True
                    except Exception as e:
                        logger.warning(
                            "Failed to sync trailing SL to API for %s %s: %s",
                            pos.symbol, pos.deal_id, e,
                        )

                adjusted.append({
                    "position_id": pos.id,
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "old_sl": old_sl,
                    "new_sl": new_sl,
                    "current_price": current_price,
                    "reason": adjustment["reason"],
                    "api_synced": api_synced,
                })
                logger.info(
                    "Trailing SL adjusted for %s %s: %.5f -> %.5f (price=%.5f, api_synced=%s)",
                    pos.direction, pos.symbol, old_sl or 0, new_sl, current_price, api_synced,
                )

        if adjusted:
            await session.commit()

        return adjusted

    def _calculate_adjustment(
        self,
        position: Position,
        current_price: float,
        atr: float,
    ) -> Optional[dict]:
        """Calculate new trailing stop level if conditions are met."""
        pos_id = position.id
        entry = position.entry_price
        sl = position.stop_loss

        if entry is None or sl is None:
            return None

        # Calculate initial risk (R)
        initial_risk = abs(entry - sl)
        if initial_risk == 0:
            return None

        # Calculate current profit in R multiples
        if position.direction == "BUY":
            profit_r = (current_price - entry) / initial_risk
        else:
            profit_r = (entry - current_price) / initial_risk

        # Check if trailing should be activated
        if profit_r < self.config.activation_r:
            # Not yet in enough profit to activate trailing
            if pos_id in self._peak_prices:
                del self._peak_prices[pos_id]
            return None

        # Track peak price
        if position.direction == "BUY":
            peak = self._peak_prices.get(pos_id, current_price)
            if current_price > peak:
                self._peak_prices[pos_id] = current_price
                peak = current_price
        else:
            peak = self._peak_prices.get(pos_id, current_price)
            if current_price < peak:
                self._peak_prices[pos_id] = current_price
                peak = current_price

        # Calculate trailing distance
        trail_distance = self._get_trail_distance(current_price, atr)
        if trail_distance <= 0:
            return None

        # Calculate new SL
        if position.direction == "BUY":
            new_sl = round(peak - trail_distance, 5)
            # Only move SL up, never down
            if sl is not None and new_sl <= sl:
                return None
            # Ensure SL doesn't exceed current price
            if new_sl >= current_price:
                return None
        else:
            new_sl = round(peak + trail_distance, 5)
            # Only move SL down, never up
            if sl is not None and new_sl >= sl:
                return None
            if new_sl <= current_price:
                return None

        # Check step size
        if self.config.step_size > 0 and sl is not None:
            if abs(new_sl - sl) < self.config.step_size:
                return None

        return {
            "new_sl": new_sl,
            "reason": f"Trailing {self.config.mode.value}: peak={peak:.5f} trail={trail_distance:.5f}",
        }

    def _get_trail_distance(self, current_price: float, atr: float) -> float:
        """Get trailing distance based on configured mode."""
        if self.config.mode == TrailingMode.FIXED:
            return self.config.fixed_distance
        elif self.config.mode == TrailingMode.ATR:
            if atr <= 0:
                return 0
            return atr * self.config.atr_multiplier
        elif self.config.mode == TrailingMode.PERCENT:
            return current_price * (self.config.trail_percent / 100)
        return 0

    def cleanup_closed_position(self, position_id: int) -> None:
        """Remove tracking data for a closed position."""
        self._peak_prices.pop(position_id, None)
