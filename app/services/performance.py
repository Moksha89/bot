"""
Performance analytics module.
Calculates Sharpe ratio, max drawdown, equity curves, and other
advanced performance metrics for the dashboard.
"""

import logging
import io
import base64
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Position, BalanceHistory, DailyPnL, TradeResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_duration_hours: float = 0.0
    expectancy: float = 0.0
    risk_reward_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    recovery_factor: float = 0.0
    equity_curve: list[dict] = field(default_factory=list)
    daily_returns: list[dict] = field(default_factory=list)
    monthly_returns: list[dict] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Analyzes trading performance and generates metrics, charts, and reports.
    """

    def __init__(self) -> None:
        pass

    async def calculate_metrics(
        self,
        session: AsyncSession,
        days: int = 90,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Fetch closed positions
        result = await session.execute(
            select(Position)
            .where(
                Position.is_open.is_(False),
                Position.closed_at >= cutoff,
            )
            .order_by(Position.closed_at)
        )
        positions = result.scalars().all()

        metrics = PerformanceMetrics()

        if not positions:
            return metrics

        # Basic stats
        pnls = [p.pnl for p in positions if p.pnl is not None]
        wins = [p for p in positions if p.result == TradeResult.WIN]
        losses = [p for p in positions if p.result == TradeResult.LOSS]

        metrics.total_trades = len(pnls)
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(pnls) if pnls else 0
        metrics.total_pnl = sum(pnls)

        win_pnls = [p.pnl for p in wins if p.pnl is not None]
        loss_pnls = [p.pnl for p in losses if p.pnl is not None]

        metrics.avg_win = np.mean(win_pnls) if win_pnls else 0
        metrics.avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        metrics.largest_win = max(win_pnls) if win_pnls else 0
        metrics.largest_loss = min(loss_pnls) if loss_pnls else 0

        total_wins = sum(win_pnls) if win_pnls else 0
        total_losses = abs(sum(loss_pnls)) if loss_pnls else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else (
            float("inf") if total_wins > 0 else 0
        )

        # Expectancy
        if pnls:
            metrics.expectancy = np.mean(pnls)

        # Risk/Reward ratio
        if metrics.avg_loss != 0:
            metrics.risk_reward_ratio = abs(metrics.avg_win / metrics.avg_loss)

        # Consecutive wins/losses
        metrics.consecutive_wins, metrics.consecutive_losses = self._max_consecutive(positions)

        # Trade duration
        durations = []
        for p in positions:
            if p.opened_at and p.closed_at:
                delta = (p.closed_at - p.opened_at).total_seconds() / 3600
                durations.append(delta)
        metrics.avg_trade_duration_hours = np.mean(durations) if durations else 0

        # Equity curve
        metrics.equity_curve = self._build_equity_curve(positions)

        # Sharpe and Sortino ratios
        if len(pnls) >= 5:
            returns = np.array(pnls)
            metrics.sharpe_ratio = self._calculate_sharpe(returns)
            metrics.sortino_ratio = self._calculate_sortino(returns)

        # Max drawdown
        if metrics.equity_curve:
            metrics.max_drawdown, metrics.max_drawdown_pct = self._calculate_max_drawdown(
                metrics.equity_curve
            )

        # Recovery factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.total_pnl / metrics.max_drawdown

        # Daily returns
        metrics.daily_returns = await self._get_daily_returns(session, days)

        # Monthly returns
        metrics.monthly_returns = self._aggregate_monthly(metrics.daily_returns)

        return metrics

    def _build_equity_curve(self, positions: list[Position]) -> list[dict]:
        """Build equity curve from closed positions."""
        curve: list[dict] = []
        cumulative = 0.0

        for p in positions:
            if p.pnl is not None and p.closed_at:
                cumulative += p.pnl
                curve.append({
                    "date": p.closed_at.isoformat(),
                    "equity": round(cumulative, 2),
                    "pnl": round(p.pnl, 2),
                    "symbol": p.symbol,
                })

        return curve

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

    @staticmethod
    def _calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sortino ratio (only penalizes downside risk)."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0
        return float(np.mean(excess_returns) / np.std(downside) * np.sqrt(252))

    @staticmethod
    def _calculate_max_drawdown(equity_curve: list[dict]) -> tuple[float, float]:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0, 0.0

        equities = [e["equity"] for e in equity_curve]
        peak = equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0

        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd / peak if peak > 0 else 0

        return round(max_dd, 2), round(max_dd_pct, 4)

    @staticmethod
    def _max_consecutive(positions: list[Position]) -> tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for p in positions:
            if p.result == TradeResult.WIN:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif p.result == TradeResult.LOSS:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    async def _get_daily_returns(
        self, session: AsyncSession, days: int,
    ) -> list[dict]:
        """Get daily P&L returns."""
        result = await session.execute(
            select(DailyPnL).order_by(desc(DailyPnL.date)).limit(days)
        )
        records = result.scalars().all()
        return [
            {
                "date": r.date,
                "pnl": r.total_pnl,
                "trades": r.trade_count,
                "wins": r.wins,
                "losses": r.losses,
            }
            for r in reversed(records)
        ]

    @staticmethod
    def _aggregate_monthly(daily_returns: list[dict]) -> list[dict]:
        """Aggregate daily returns into monthly."""
        monthly: dict[str, dict] = {}
        for d in daily_returns:
            month = d["date"][:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = {"month": month, "pnl": 0, "trades": 0, "wins": 0, "losses": 0}
            monthly[month]["pnl"] += d["pnl"]
            monthly[month]["trades"] += d["trades"]
            monthly[month]["wins"] += d["wins"]
            monthly[month]["losses"] += d["losses"]

        return list(monthly.values())

    async def generate_equity_chart(
        self, session: AsyncSession, days: int = 90,
    ) -> Optional[str]:
        """
        Generate an equity curve chart as base64-encoded PNG.
        Returns base64 string or None if insufficient data.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        metrics = await self.calculate_metrics(session, days)
        if not metrics.equity_curve:
            return None

        dates = [datetime.fromisoformat(e["date"]) for e in metrics.equity_curve]
        equities = [e["equity"] for e in metrics.equity_curve]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#1a1a2e")

        # Equity curve
        ax1.set_facecolor("#16213e")
        color = "#00ff88" if equities[-1] >= 0 else "#ff4444"
        ax1.plot(dates, equities, color=color, linewidth=2)
        ax1.fill_between(dates, equities, alpha=0.1, color=color)
        ax1.set_title("Equity Curve", color="white", fontsize=14)
        ax1.set_ylabel("Cumulative P&L", color="white")
        ax1.tick_params(colors="white")
        ax1.grid(True, alpha=0.2)
        ax1.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)

        # Trade P&L bars
        ax2.set_facecolor("#16213e")
        pnls = [e["pnl"] for e in metrics.equity_curve]
        colors = ["#00ff88" if p >= 0 else "#ff4444" for p in pnls]
        ax2.bar(dates, pnls, color=colors, alpha=0.7, width=0.8)
        ax2.set_title("Trade P&L", color="white", fontsize=12)
        ax2.set_ylabel("P&L", color="white")
        ax2.tick_params(colors="white")
        ax2.grid(True, alpha=0.2)
        ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)

        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.spines["bottom"].set_color("#333")
            ax.spines["left"].set_color("#333")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
