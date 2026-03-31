"""
Backtesting engine module.
Tests trading strategies on historical data before going live.
Supports multiple strategies, configurable parameters, and detailed reporting.
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.strategy.indicators import add_all_indicators
from app.strategy.rules import MarketSnapshot
from app.strategy.strategies import (
    StrategySelector,
    StrategyResult,
    StrategyType,
    EmaCrossoverStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A simulated trade from backtesting."""
    entry_index: int
    exit_index: int
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    size: float
    pnl: float
    pnl_pct: float
    result: str  # "WIN" or "LOSS"
    strategy: str
    entry_time: str
    exit_time: str
    bars_held: int


@dataclass
class BacktestResult:
    """Complete backtesting result."""
    symbol: str
    timeframe: str
    strategy: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    avg_bars_held: float
    expectancy: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine that simulates trading on historical data.
    Supports all available strategies with configurable parameters.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        max_open_trades: int = 1,
        commission_pct: float = 0.001,  # 0.1% commission
    ) -> None:
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_open_trades = max_open_trades
        self.commission_pct = commission_pct

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "XAUUSD",
        timeframe: str = "HOUR",
        strategy_type: str = "ema_crossover",
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        rsi_buy_min: float = 55,
        rsi_buy_max: float = 70,
        rsi_sell_min: float = 30,
        rsi_sell_max: float = 45,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 2.0,
        max_spread: float = 5.0,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        df must have columns: open, high, low, close, volume (optional).
        """
        # Add indicators
        df = add_all_indicators(df, ema_fast, ema_slow, rsi_period)
        df = df.dropna().reset_index(drop=True)

        if len(df) < 10:
            return BacktestResult(
                symbol=symbol, timeframe=timeframe, strategy=strategy_type,
                start_date="", end_date="", initial_balance=self.initial_balance,
                final_balance=self.initial_balance, total_trades=0,
                winning_trades=0, losing_trades=0, win_rate=0,
                total_pnl=0, total_return_pct=0, profit_factor=0,
                sharpe_ratio=0, max_drawdown=0, max_drawdown_pct=0,
                avg_win=0, avg_loss=0, avg_bars_held=0, expectancy=0,
            )

        # Select strategy
        strategy = self._get_strategy(strategy_type)
        selector = StrategySelector()

        balance = self.initial_balance
        trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        open_trade: Optional[dict] = None

        for i in range(ema_slow + 5, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            current_price = float(row["close"])

            date_str = str(row.get("datetime", i))

            # Track equity
            unrealized = 0.0
            if open_trade:
                if open_trade["direction"] == "BUY":
                    unrealized = (current_price - open_trade["entry"]) * open_trade["size"]
                else:
                    unrealized = (open_trade["entry"] - current_price) * open_trade["size"]

            equity_curve.append({
                "bar": i,
                "date": date_str,
                "equity": round(balance + unrealized, 2),
                "balance": round(balance, 2),
            })

            # Check if open trade hits SL/TP
            if open_trade:
                high = float(row["high"])
                low = float(row["low"])
                closed = False

                if open_trade["direction"] == "BUY":
                    if low <= open_trade["sl"]:
                        # Stop loss hit
                        exit_price = open_trade["sl"]
                        closed = True
                    elif high >= open_trade["tp"]:
                        # Take profit hit
                        exit_price = open_trade["tp"]
                        closed = True
                elif open_trade["direction"] == "SELL":
                    if high >= open_trade["sl"]:
                        # Stop loss hit
                        exit_price = open_trade["sl"]
                        closed = True
                    elif low <= open_trade["tp"]:
                        # Take profit hit
                        exit_price = open_trade["tp"]
                        closed = True

                if closed:
                    size = open_trade["size"]
                    if open_trade["direction"] == "BUY":
                        pnl = (exit_price - open_trade["entry"]) * size
                    else:
                        pnl = (open_trade["entry"] - exit_price) * size

                    # Apply commission
                    commission = abs(exit_price * size * self.commission_pct)
                    pnl -= commission

                    balance += pnl
                    pnl_pct = pnl / self.initial_balance * 100

                    trade = BacktestTrade(
                        entry_index=open_trade["bar"],
                        exit_index=i,
                        direction=open_trade["direction"],
                        entry_price=open_trade["entry"],
                        exit_price=exit_price,
                        stop_loss=open_trade["sl"],
                        take_profit=open_trade["tp"],
                        size=size,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 4),
                        result="WIN" if pnl > 0 else "LOSS",
                        strategy=strategy_type,
                        entry_time=open_trade["date"],
                        exit_time=date_str,
                        bars_held=i - open_trade["bar"],
                    )
                    trades.append(trade)
                    open_trade = None

            # Generate signal if no open trade
            if open_trade is None:
                snapshot = MarketSnapshot(
                    close=current_price,
                    ema_fast=float(row["ema_fast"]),
                    ema_slow=float(row["ema_slow"]),
                    rsi=float(row["rsi"]),
                    atr=float(row["atr"]),
                    prev_high=float(prev_row["high"]),
                    prev_low=float(prev_row["low"]),
                    spread=0.5,  # Assume typical spread for backtest
                )

                if strategy_type == "auto":
                    result = selector.evaluate_all(df.iloc[:i + 1], snapshot)
                else:
                    result = strategy.evaluate(
                        df.iloc[:i + 1], snapshot,
                        **self._build_strategy_kwargs(
                            strategy_type, rsi_buy_min, rsi_buy_max,
                            rsi_sell_min, rsi_sell_max, max_spread,
                            sl_atr_mult, tp_rr,
                        ),
                    )

                if result.direction in ("BUY", "SELL") and result.stop_loss and result.take_profit:
                    # Calculate position size
                    risk_amount = balance * self.risk_per_trade
                    sl_distance = abs(current_price - result.stop_loss)
                    if sl_distance > 0:
                        size = risk_amount / sl_distance
                    else:
                        continue

                    # Apply commission on entry
                    commission = abs(current_price * size * self.commission_pct)
                    balance -= commission

                    open_trade = {
                        "direction": result.direction,
                        "entry": current_price,
                        "sl": result.stop_loss,
                        "tp": result.take_profit,
                        "size": size,
                        "bar": i,
                        "date": date_str,
                    }

        # Close any remaining open trade at last price
        if open_trade:
            last_price = float(df.iloc[-1]["close"])
            size = open_trade["size"]
            if open_trade["direction"] == "BUY":
                pnl = (last_price - open_trade["entry"]) * size
            else:
                pnl = (open_trade["entry"] - last_price) * size
            balance += pnl

        # Calculate metrics
        return self._calculate_result(
            symbol, timeframe, strategy_type, df, trades, equity_curve,
            balance, ema_fast, ema_slow, rsi_buy_min, rsi_buy_max,
            rsi_sell_min, rsi_sell_max, sl_atr_mult, tp_rr,
        )

    def _get_strategy(self, strategy_type: str) -> object:
        """Get strategy instance by type."""
        strategies = {
            "ema_crossover": EmaCrossoverStrategy(),
            "breakout": BreakoutStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }
        return strategies.get(strategy_type, EmaCrossoverStrategy())

    @staticmethod
    def _build_strategy_kwargs(
        strategy_type: str,
        rsi_buy_min: float, rsi_buy_max: float,
        rsi_sell_min: float, rsi_sell_max: float,
        max_spread: float, sl_atr_mult: float, tp_rr: float,
    ) -> dict:
        """Build kwargs for strategy evaluate method."""
        if strategy_type == "ema_crossover":
            return {
                "rsi_buy_min": rsi_buy_min, "rsi_buy_max": rsi_buy_max,
                "rsi_sell_min": rsi_sell_min, "rsi_sell_max": rsi_sell_max,
                "max_spread": max_spread, "sl_atr_mult": sl_atr_mult, "tp_rr": tp_rr,
            }
        elif strategy_type == "breakout":
            return {"max_spread": max_spread, "sl_atr_mult": sl_atr_mult, "tp_rr": tp_rr}
        elif strategy_type == "mean_reversion":
            return {"max_spread": max_spread, "sl_atr_mult": sl_atr_mult, "tp_rr": tp_rr}
        return {}

    def _calculate_result(
        self,
        symbol: str, timeframe: str, strategy_type: str,
        df: pd.DataFrame, trades: list[BacktestTrade],
        equity_curve: list[dict], final_balance: float,
        ema_fast: int, ema_slow: int,
        rsi_buy_min: float, rsi_buy_max: float,
        rsi_sell_min: float, rsi_sell_max: float,
        sl_atr_mult: float, tp_rr: float,
    ) -> BacktestResult:
        """Calculate backtest result metrics."""
        wins = [t for t in trades if t.result == "WIN"]
        losses = [t for t in trades if t.result == "LOSS"]

        win_pnls = [t.pnl for t in wins]
        loss_pnls = [t.pnl for t in losses]
        all_pnls = [t.pnl for t in trades]

        total_win = sum(win_pnls) if win_pnls else 0
        total_loss = abs(sum(loss_pnls)) if loss_pnls else 0

        # Sharpe ratio
        sharpe = 0.0
        if len(all_pnls) >= 5:
            returns = np.array(all_pnls)
            if returns.std() > 0:
                sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Max drawdown
        max_dd = 0.0
        max_dd_pct = 0.0
        if equity_curve:
            equities = [e["equity"] for e in equity_curve]
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd / peak if peak > 0 else 0

        start_date = str(df.iloc[0].get("datetime", "")) if len(df) > 0 else ""
        end_date = str(df.iloc[-1].get("datetime", "")) if len(df) > 0 else ""

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy_type,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=round(final_balance, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=round(len(wins) / len(trades), 4) if trades else 0,
            total_pnl=round(final_balance - self.initial_balance, 2),
            total_return_pct=round((final_balance - self.initial_balance) / self.initial_balance * 100, 2),
            profit_factor=round(total_win / total_loss, 2) if total_loss > 0 else (
                999.0 if total_win > 0 else 0
            ),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            avg_win=round(np.mean(win_pnls), 2) if win_pnls else 0,
            avg_loss=round(np.mean(loss_pnls), 2) if loss_pnls else 0,
            avg_bars_held=round(np.mean([t.bars_held for t in trades]), 1) if trades else 0,
            expectancy=round(np.mean(all_pnls), 2) if all_pnls else 0,
            trades=trades,
            equity_curve=equity_curve,
            parameters={
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi_buy_range": [rsi_buy_min, rsi_buy_max],
                "rsi_sell_range": [rsi_sell_min, rsi_sell_max],
                "sl_atr_multiplier": sl_atr_mult,
                "tp_risk_reward": tp_rr,
                "risk_per_trade": self.risk_per_trade,
                "commission": self.commission_pct,
            },
        )
