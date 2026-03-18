"""
Signal generator that combines indicators and rules to produce trading signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from app.config import settings
from app.strategy.indicators import add_all_indicators
from app.strategy.rules import (
    MarketSnapshot,
    RuleResult,
    evaluate_buy_rules,
    evaluate_sell_rules,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """A generated trading signal with full context."""
    timestamp: datetime
    symbol: str
    timeframe: str
    direction: str  # BUY, SELL, NO_TRADE
    close_price: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    prev_high: float
    prev_low: float
    spread: float
    reasons: list[str] = field(default_factory=list)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class SignalGenerator:
    """
    Generates trading signals based on configured strategy parameters.
    """

    def __init__(self) -> None:
        self.ema_fast = settings.strategy.ema_fast
        self.ema_slow = settings.strategy.ema_slow
        self.rsi_period = settings.strategy.rsi_period
        self.rsi_buy_min = settings.strategy.rsi_buy_min
        self.rsi_buy_max = settings.strategy.rsi_buy_max
        self.rsi_sell_min = settings.strategy.rsi_sell_min
        self.rsi_sell_max = settings.strategy.rsi_sell_max
        self.max_spread = settings.strategy.max_spread
        self.sl_atr_mult = settings.risk.sl_atr_multiplier
        self.tp_rr = settings.risk.tp_risk_reward
        self.require_breakout = settings.strategy.require_breakout

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        spread: float,
        has_open_long: bool = False,
        has_open_short: bool = False,
    ) -> TradeSignal:
        """
        Generate a signal from OHLCV data.
        df must have columns: open, high, low, close (and optionally volume).
        """
        # Add indicators
        df = add_all_indicators(
            df,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            rsi_period=self.rsi_period,
        )

        if len(df) < self.ema_slow + 1:
            return TradeSignal(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                direction="NO_TRADE",
                close_price=0,
                ema_fast=0,
                ema_slow=0,
                rsi=0,
                atr=0,
                prev_high=0,
                prev_low=0,
                spread=spread,
                reasons=["Insufficient data for indicators"],
            )

        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        snapshot = MarketSnapshot(
            close=float(latest["close"]),
            ema_fast=float(latest["ema_fast"]),
            ema_slow=float(latest["ema_slow"]),
            rsi=float(latest["rsi"]),
            atr=float(latest["atr"]),
            prev_high=float(prev["high"]),
            prev_low=float(prev["low"]),
            spread=spread,
            has_open_long=has_open_long,
            has_open_short=has_open_short,
        )

        # Evaluate buy rules
        buy_pass, buy_results = evaluate_buy_rules(
            snapshot,
            rsi_min=self.rsi_buy_min,
            rsi_max=self.rsi_buy_max,
            max_spread=self.max_spread,
            require_breakout=self.require_breakout,
        )
        if buy_pass:
            sl = snapshot.close - (snapshot.atr * self.sl_atr_mult)
            risk = snapshot.close - sl
            tp = snapshot.close + (risk * self.tp_rr)
            return TradeSignal(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                direction="BUY",
                close_price=snapshot.close,
                ema_fast=snapshot.ema_fast,
                ema_slow=snapshot.ema_slow,
                rsi=snapshot.rsi,
                atr=snapshot.atr,
                prev_high=snapshot.prev_high,
                prev_low=snapshot.prev_low,
                spread=spread,
                reasons=[r.reason for r in buy_results],
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
            )

        # Evaluate sell rules
        sell_pass, sell_results = evaluate_sell_rules(
            snapshot,
            rsi_min=self.rsi_sell_min,
            rsi_max=self.rsi_sell_max,
            max_spread=self.max_spread,
            require_breakout=self.require_breakout,
        )
        if sell_pass:
            sl = snapshot.close + (snapshot.atr * self.sl_atr_mult)
            risk = sl - snapshot.close
            tp = snapshot.close - (risk * self.tp_rr)
            return TradeSignal(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                timeframe=timeframe,
                direction="SELL",
                close_price=snapshot.close,
                ema_fast=snapshot.ema_fast,
                ema_slow=snapshot.ema_slow,
                rsi=snapshot.rsi,
                atr=snapshot.atr,
                prev_high=snapshot.prev_high,
                prev_low=snapshot.prev_low,
                spread=spread,
                reasons=[r.reason for r in sell_results],
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
            )

        # No trade
        all_reasons = [r.reason for r in buy_results if not r.passed]
        all_reasons += [r.reason for r in sell_results if not r.passed]
        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            direction="NO_TRADE",
            close_price=snapshot.close,
            ema_fast=snapshot.ema_fast,
            ema_slow=snapshot.ema_slow,
            rsi=snapshot.rsi,
            atr=snapshot.atr,
            prev_high=snapshot.prev_high,
            prev_low=snapshot.prev_low,
            spread=spread,
            reasons=all_reasons,
        )
