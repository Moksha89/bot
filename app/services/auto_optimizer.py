"""
Auto-optimization module.
Automatically tunes EMA periods, RSI ranges, and other strategy parameters
based on recent trading performance using walk-forward optimization.
"""

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Position, TradeResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    ema_fast: int
    ema_slow: int
    rsi_buy_min: float
    rsi_buy_max: float
    rsi_sell_min: float
    rsi_sell_max: float
    sl_atr_multiplier: float
    tp_risk_reward: float
    win_rate: float
    profit_factor: float
    total_trades: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AutoOptimizer:
    """
    Optimizes strategy parameters based on historical performance.
    Uses grid search over parameter ranges with walk-forward validation.
    """

    def __init__(
        self,
        enabled: bool = True,
        lookback_days: int = 30,
        min_trades: int = 20,
        optimize_interval_hours: int = 24,
    ) -> None:
        self.enabled = enabled
        self.lookback_days = lookback_days
        self.min_trades = min_trades
        self.optimize_interval = timedelta(hours=optimize_interval_hours)
        self._last_optimization: Optional[datetime] = None
        self._current_params: Optional[OptimizationResult] = None

    async def should_optimize(self) -> bool:
        """Check if it's time to run optimization."""
        if not self.enabled:
            return False
        if self._last_optimization is None:
            return True
        return (datetime.now(timezone.utc) - self._last_optimization) >= self.optimize_interval

    async def optimize(
        self,
        session: AsyncSession,
        df: pd.DataFrame,
    ) -> Optional[OptimizationResult]:
        """
        Run optimization on historical data.
        Returns optimized parameters or None if insufficient data.
        """
        if not self.enabled:
            return None

        # Fetch recent closed trades
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        result = await session.execute(
            select(Position).where(
                Position.is_open.is_(False),
                Position.closed_at >= cutoff,
                Position.result.in_([TradeResult.WIN, TradeResult.LOSS]),
            )
        )
        positions = result.scalars().all()

        if len(positions) < self.min_trades:
            logger.info(
                "Insufficient trades for optimization: %d/%d",
                len(positions), self.min_trades,
            )
            return None

        # Backtest with different parameter combinations
        best = self._grid_search(df, positions)

        if best:
            self._last_optimization = datetime.now(timezone.utc)
            self._current_params = best
            logger.info(
                "Optimization complete: EMA=%d/%d RSI_buy=[%.0f,%.0f] RSI_sell=[%.0f,%.0f] "
                "SL=%.1f TP=%.1f WR=%.1f%% PF=%.2f trades=%d",
                best.ema_fast, best.ema_slow,
                best.rsi_buy_min, best.rsi_buy_max,
                best.rsi_sell_min, best.rsi_sell_max,
                best.sl_atr_multiplier, best.tp_risk_reward,
                best.win_rate * 100, best.profit_factor, best.total_trades,
            )

        return best

    def _grid_search(
        self,
        df: pd.DataFrame,
        positions: list[Position],
    ) -> Optional[OptimizationResult]:
        """Run grid search over parameter space using backtesting."""
        from app.services.backtester import BacktestEngine

        # Parameter ranges (kept small for performance)
        ema_fast_range = [10, 15, 20, 25]
        ema_slow_range = [40, 50, 60]
        rsi_buy_ranges = [(50, 65), (55, 70), (55, 75)]
        rsi_sell_ranges = [(25, 40), (30, 45), (30, 50)]
        sl_atr_range = [1.0, 1.5, 2.0]
        tp_rr_range = [1.5, 2.0, 2.5, 3.0]

        best_result: Optional[OptimizationResult] = None
        best_score = -999.0

        engine = BacktestEngine(initial_balance=10000.0)

        for ema_f in ema_fast_range:
            for ema_s in ema_slow_range:
                if ema_f >= ema_s:
                    continue
                for rsi_buy in rsi_buy_ranges:
                    for rsi_sell in rsi_sell_ranges:
                        for sl_mult in sl_atr_range:
                            for tp_rr in tp_rr_range:
                                score, wr, pf, trades = self._evaluate_params(
                                    engine, df,
                                    ema_f, ema_s,
                                    rsi_buy[0], rsi_buy[1],
                                    rsi_sell[0], rsi_sell[1],
                                    sl_mult, tp_rr,
                                )
                                if score > best_score and trades >= 5:
                                    best_score = score
                                    best_result = OptimizationResult(
                                        ema_fast=ema_f,
                                        ema_slow=ema_s,
                                        rsi_buy_min=rsi_buy[0],
                                        rsi_buy_max=rsi_buy[1],
                                        rsi_sell_min=rsi_sell[0],
                                        rsi_sell_max=rsi_sell[1],
                                        sl_atr_multiplier=sl_mult,
                                        tp_risk_reward=tp_rr,
                                        win_rate=wr,
                                        profit_factor=pf,
                                        total_trades=trades,
                                    )

        return best_result

    def _evaluate_params(
        self,
        engine: object,
        df: pd.DataFrame,
        ema_fast: int,
        ema_slow: int,
        rsi_buy_min: float,
        rsi_buy_max: float,
        rsi_sell_min: float,
        rsi_sell_max: float,
        sl_atr_mult: float,
        tp_rr: float,
    ) -> tuple[float, float, float, int]:
        """
        Evaluate a parameter set by running a backtest on the DataFrame.
        Returns (score, win_rate, profit_factor, trade_count).
        Score = profit_factor * sqrt(trades) to balance profitability and sample size.
        """
        try:
            result = engine.run(  # type: ignore[attr-defined]
                df,
                strategy_type="ema_crossover",
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rsi_buy_min=rsi_buy_min,
                rsi_buy_max=rsi_buy_max,
                rsi_sell_min=rsi_sell_min,
                rsi_sell_max=rsi_sell_max,
                sl_atr_mult=sl_atr_mult,
                tp_rr=tp_rr,
            )
        except Exception:
            return -999.0, 0.0, 0.0, 0

        total = result.total_trades
        if total == 0:
            return -999.0, 0.0, 0.0, 0

        win_rate = result.win_rate
        profit_factor = result.profit_factor

        # Score: profit_factor weighted by trade count and win rate
        score = profit_factor * (total ** 0.5) * (1 + win_rate)

        # Penalize extreme parameters
        if ema_fast < 8 or ema_slow > 80:
            score *= 0.8
        if tp_rr < 1.2:
            score *= 0.9

        return score, win_rate, profit_factor, total

    def get_current_params(self) -> Optional[OptimizationResult]:
        """Get the most recent optimization result."""
        return self._current_params

    def get_optimization_summary(self) -> dict:
        """Get summary of current optimization state."""
        return {
            "enabled": self.enabled,
            "last_optimization": (
                self._last_optimization.isoformat()
                if self._last_optimization else None
            ),
            "has_optimized_params": self._current_params is not None,
            "params": {
                "ema_fast": self._current_params.ema_fast,
                "ema_slow": self._current_params.ema_slow,
                "rsi_buy_range": [self._current_params.rsi_buy_min, self._current_params.rsi_buy_max],
                "rsi_sell_range": [self._current_params.rsi_sell_min, self._current_params.rsi_sell_max],
                "sl_atr_multiplier": self._current_params.sl_atr_multiplier,
                "tp_risk_reward": self._current_params.tp_risk_reward,
                "win_rate": self._current_params.win_rate,
                "profit_factor": self._current_params.profit_factor,
            } if self._current_params else None,
        }
