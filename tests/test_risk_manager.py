"""
Unit tests for risk manager.
"""

import pytest

from app.execution.risk_manager import RiskManager


class TestPositionSizing:
    def test_basic_position_size(self) -> None:
        rm = RiskManager()
        rm.risk_per_trade = 0.01
        size = rm.calculate_position_size(
            account_balance=10000.0,
            entry_price=1950.0,
            stop_loss=1945.0,
        )
        # risk = 10000 * 0.01 = 100, distance = 5, size = 100/5 = 20
        assert size == 20.0

    def test_small_account(self) -> None:
        rm = RiskManager()
        rm.risk_per_trade = 0.01
        size = rm.calculate_position_size(
            account_balance=500.0,
            entry_price=1.1000,
            stop_loss=1.0950,
        )
        # risk = 5, distance = 0.005, size = 1000
        assert size == 1000.0

    def test_minimum_size(self) -> None:
        rm = RiskManager()
        rm.risk_per_trade = 0.001
        size = rm.calculate_position_size(
            account_balance=100.0,
            entry_price=50000.0,
            stop_loss=49000.0,
        )
        # risk = 0.1, distance = 1000, size = 0.0001 -> min 0.01
        assert size == 0.01

    def test_zero_stop_distance(self) -> None:
        rm = RiskManager()
        size = rm.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss=100.0,
        )
        assert size == 0.01

    def test_sell_position_size(self) -> None:
        rm = RiskManager()
        rm.risk_per_trade = 0.02
        size = rm.calculate_position_size(
            account_balance=10000.0,
            entry_price=1930.0,
            stop_loss=1940.0,
        )
        # risk = 200, distance = 10, size = 20
        assert size == 20.0


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_kill_switch_off(self) -> None:
        from app.config import settings
        original = settings.kill_switch
        settings.kill_switch = False
        rm = RiskManager()
        passed, msg = await rm.check_kill_switch()
        assert passed is True
        settings.kill_switch = original

    @pytest.mark.asyncio
    async def test_kill_switch_on(self) -> None:
        from app.config import settings
        original = settings.kill_switch
        settings.kill_switch = True
        rm = RiskManager()
        passed, msg = await rm.check_kill_switch()
        assert passed is False
        assert "Kill switch" in msg
        settings.kill_switch = original


class TestConsecutiveLossesAndCooldown:
    """Cooldown logic is now integrated into check_consecutive_losses."""

    def test_no_consecutive_losses_by_default(self) -> None:
        rm = RiskManager()
        # With no DB session / no closed positions, consecutive losses
        # tracking starts at 0 and _last_loss_time is None.
        assert rm._consecutive_losses == 0
        assert rm._last_loss_time is None
