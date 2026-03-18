"""
Risk manager module.
Controls position sizing, daily loss limits, consecutive loss tracking,
cooldown periods, and duplicate trade prevention.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Position, DailyPnL, TradeResult

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces all risk management rules before allowing trade execution.
    """

    def __init__(self) -> None:
        self.risk_per_trade = settings.risk.risk_per_trade
        self.max_daily_loss = settings.risk.max_daily_loss
        self.max_consecutive_losses = settings.risk.max_consecutive_losses
        self.cooldown_minutes = settings.risk.cooldown_minutes
        self.max_open_trades = settings.risk.max_open_trades
        self._last_loss_time: Optional[datetime] = None
        self._consecutive_losses: int = 0

    async def check_kill_switch(self) -> tuple[bool, str]:
        """Check if the global kill switch is activated."""
        if settings.kill_switch:
            return False, "Kill switch is activated"
        return True, "Kill switch is off"

    async def check_max_open_trades(self, session: AsyncSession) -> tuple[bool, str]:
        """Check if max open trades limit has been reached."""
        result = await session.execute(
            select(func.count(Position.id)).where(Position.is_open.is_(True))
        )
        open_count = result.scalar() or 0
        if open_count >= self.max_open_trades:
            return False, f"Max open trades reached ({open_count}/{self.max_open_trades})"
        return True, f"Open trades: {open_count}/{self.max_open_trades}"

    async def check_no_duplicate(
        self, session: AsyncSession, symbol: str, direction: str
    ) -> tuple[bool, str]:
        """Check that there is no duplicate open position for this symbol and direction."""
        result = await session.execute(
            select(func.count(Position.id)).where(
                Position.is_open.is_(True),
                Position.symbol == symbol,
                Position.direction == direction,
            )
        )
        count = result.scalar() or 0
        if count > 0:
            return False, f"Duplicate {direction} position already open for {symbol}"
        return True, f"No duplicate {direction} position for {symbol}"

    async def check_max_one_per_symbol(
        self, session: AsyncSession, symbol: str
    ) -> tuple[bool, str]:
        """Check that there is at most one open trade per symbol."""
        result = await session.execute(
            select(func.count(Position.id)).where(
                Position.is_open.is_(True),
                Position.symbol == symbol,
            )
        )
        count = result.scalar() or 0
        if count > 0:
            return False, f"Already have an open trade for {symbol}"
        return True, f"No open trade for {symbol}"

    async def check_daily_loss(
        self, session: AsyncSession, account_balance: float
    ) -> tuple[bool, str]:
        """Check if daily loss limit has been reached."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = await session.execute(
            select(DailyPnL).where(DailyPnL.date == today)
        )
        daily = result.scalar_one_or_none()
        if daily and account_balance > 0:
            daily_loss_pct = abs(min(daily.total_pnl, 0)) / account_balance
            if daily_loss_pct >= self.max_daily_loss:
                return False, (
                    f"Daily loss limit reached: {daily_loss_pct:.2%} >= {self.max_daily_loss:.2%}"
                )
            return True, f"Daily loss: {daily_loss_pct:.2%} (limit: {self.max_daily_loss:.2%})"
        return True, "No daily P&L data yet"

    async def check_consecutive_losses(self, session: AsyncSession) -> tuple[bool, str]:
        """Check consecutive loss count, accounting for cooldown period.

        If the last N trades are all losses:
        - If cooldown has NOT expired since the last loss, block trading.
        - If cooldown HAS expired, allow trading to resume (the streak
          can only be broken by placing a new trade).
        """
        result = await session.execute(
            select(Position)
            .where(Position.is_open.is_(False))
            .order_by(Position.closed_at.desc())
            .limit(self.max_consecutive_losses)
        )
        recent = result.scalars().all()

        if len(recent) >= self.max_consecutive_losses:
            all_losses = all(p.result == TradeResult.LOSS for p in recent)
            if all_losses:
                self._consecutive_losses = len(recent)
                last_loss_time = recent[0].closed_at
                self._last_loss_time = last_loss_time

                # Check if cooldown period has elapsed
                if last_loss_time:
                    cooldown_end = last_loss_time + timedelta(minutes=self.cooldown_minutes)
                    now = datetime.now(timezone.utc)
                    # Ensure timezone-aware comparison (SQLite may strip tzinfo)
                    if cooldown_end.tzinfo is None:
                        cooldown_end = cooldown_end.replace(tzinfo=timezone.utc)
                    if now >= cooldown_end:
                        # Cooldown expired — allow trading to resume
                        return True, (
                            f"Had {self.max_consecutive_losses} consecutive losses "
                            f"but cooldown has expired. Trading resumed."
                        )
                    else:
                        remaining = (cooldown_end - now).total_seconds() / 60
                        return False, (
                            f"Hit {self.max_consecutive_losses} consecutive losses. "
                            f"Cooldown active ({remaining:.0f} min remaining)."
                        )

                return False, (
                    f"Hit {self.max_consecutive_losses} consecutive losses. "
                    f"Cooldown required."
                )

        self._consecutive_losses = 0
        return True, f"Consecutive losses below limit ({self.max_consecutive_losses})"

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Calculate position size based on risk per trade.
        risk_amount = account_balance * risk_per_trade
        size = risk_amount / distance_to_stop
        """
        risk_amount = account_balance * self.risk_per_trade
        distance = abs(entry_price - stop_loss)
        if distance == 0:
            logger.warning("Stop loss distance is zero, returning minimum size")
            return 0.01
        size = risk_amount / distance
        return round(max(size, 0.01), 2)

    async def validate_trade(
        self,
        session: AsyncSession,
        symbol: str,
        direction: str,
        account_balance: float,
    ) -> tuple[bool, list[str]]:
        """
        Run all risk checks before allowing a trade.
        Returns (approved, list_of_messages).
        """
        messages: list[str] = []
        approved = True

        checks = [
            await self.check_kill_switch(),
            await self.check_max_open_trades(session),
            await self.check_max_one_per_symbol(session, symbol),
            await self.check_no_duplicate(session, symbol, direction),
            await self.check_daily_loss(session, account_balance),
            await self.check_consecutive_losses(session),
        ]

        for passed, msg in checks:
            messages.append(msg)
            if not passed:
                approved = False
                logger.warning("Risk check failed: %s", msg)

        if approved:
            logger.info("All risk checks passed for %s %s", direction, symbol)
        else:
            logger.warning(
                "Trade rejected for %s %s: %s",
                direction,
                symbol,
                "; ".join(m for i, (p, m) in enumerate(checks) if not p),
            )

        return approved, messages
