"""
Notification service.
Sends alerts via Telegram for trades, errors, and daily summaries.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from app.config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends notifications via Telegram Bot API.
    """

    def __init__(self) -> None:
        self.bot_token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self.enabled = settings.telegram.enabled
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            logger.debug("Telegram notifications disabled, message: %s", text[:100])
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.info("Telegram message sent successfully")
                        return True
                    else:
                        body = await resp.text()
                        logger.error(
                            "Telegram send failed: %s %s", resp.status, body
                        )
                        return False
        except aiohttp.ClientError as e:
            logger.error("Telegram connection error: %s", e)
            return False

    async def notify_bot_started(self) -> None:
        """Send notification that the bot has started."""
        mode = settings.trading.mode.upper()
        symbol = settings.trading.symbol
        await self.send_message(
            f"🤖 <b>Trading Bot Started</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Time: <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>"
        )

    async def notify_trade_opened(
        self,
        direction: str,
        symbol: str,
        size: float,
        entry: float,
        sl: Optional[float],
        tp: Optional[float],
        mode: str = "",
    ) -> None:
        """Send notification for a new trade."""
        mode_str = mode or settings.trading.mode.upper()
        sl_str = f"{sl:.5f}" if sl else "N/A"
        tp_str = f"{tp:.5f}" if tp else "N/A"
        emoji = "📈" if direction == "BUY" else "📉"
        await self.send_message(
            f"{emoji} <b>Trade Opened</b>\n"
            f"Direction: <code>{direction}</code>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Size: <code>{size}</code>\n"
            f"Entry: <code>{entry:.5f}</code>\n"
            f"SL: <code>{sl_str}</code>\n"
            f"TP: <code>{tp_str}</code>\n"
            f"Mode: <code>{mode_str}</code>"
        )

    async def notify_trade_closed(
        self,
        direction: str,
        symbol: str,
        pnl: float,
        result: str,
    ) -> None:
        """Send notification for a closed trade."""
        emoji = "✅" if pnl >= 0 else "❌"
        await self.send_message(
            f"{emoji} <b>Trade Closed</b>\n"
            f"Direction: <code>{direction}</code>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"P&L: <code>{pnl:+.2f}</code>\n"
            f"Result: <code>{result}</code>"
        )

    async def notify_daily_summary(
        self,
        date: str,
        total_pnl: float,
        trade_count: int,
        wins: int,
        losses: int,
        balance: float,
    ) -> None:
        """Send daily P&L summary."""
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        await self.send_message(
            f"📊 <b>Daily Summary - {date}</b>\n"
            f"Total P&L: <code>{total_pnl:+.2f}</code>\n"
            f"Trades: <code>{trade_count}</code>\n"
            f"Wins: <code>{wins}</code> | Losses: <code>{losses}</code>\n"
            f"Win Rate: <code>{win_rate:.1f}%</code>\n"
            f"Balance: <code>{balance:.2f}</code>"
        )

    async def notify_error(self, module: str, error: str) -> None:
        """Send error alert."""
        await self.send_message(
            f"🚨 <b>Error Alert</b>\n"
            f"Module: <code>{module}</code>\n"
            f"Error: <code>{error[:500]}</code>\n"
            f"Time: <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>"
        )

    async def notify_risk_limit(self, message: str) -> None:
        """Send notification when a risk limit is reached."""
        await self.send_message(
            f"⚠️ <b>Risk Limit Reached</b>\n"
            f"<code>{message}</code>"
        )
