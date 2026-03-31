"""
Multiple trading strategies module.
Supports EMA crossover, breakout, and mean-reversion strategies.
Automatically selects the best strategy based on market conditions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

from app.strategy.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
)
from app.strategy.rules import MarketSnapshot, RuleResult

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    EMA_CROSSOVER = "ema_crossover"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    SWING = "swing"
    SCALPING = "scalping"


@dataclass
class StrategyResult:
    """Result from a strategy evaluation."""
    strategy: StrategyType
    direction: str  # BUY, SELL, NO_TRADE
    confidence: float  # 0.0 to 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: list[str] = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []


class EmaCrossoverStrategy:
    """Original EMA crossover strategy with RSI confirmation."""

    name = StrategyType.EMA_CROSSOVER

    def evaluate(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        rsi_buy_min: float = 55,
        rsi_buy_max: float = 70,
        rsi_sell_min: float = 30,
        rsi_sell_max: float = 45,
        max_spread: float = 5.0,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 2.0,
    ) -> StrategyResult:
        """Evaluate EMA crossover strategy."""
        reasons: list[str] = []
        confidence = 0.0

        # Buy conditions
        buy_conditions = [
            snapshot.ema_fast > snapshot.ema_slow,
            rsi_buy_min <= snapshot.rsi <= rsi_buy_max,
            snapshot.close > snapshot.prev_high,
            snapshot.spread <= max_spread,
            not snapshot.has_open_long,
        ]

        if all(buy_conditions):
            sl = snapshot.close - (snapshot.atr * sl_atr_mult)
            risk = snapshot.close - sl
            tp = snapshot.close + (risk * tp_rr)
            # Confidence based on EMA separation and RSI position
            ema_sep = (snapshot.ema_fast - snapshot.ema_slow) / snapshot.ema_slow
            confidence = min(0.5 + ema_sep * 100 + (snapshot.rsi - 50) / 100, 1.0)
            reasons.append("EMA crossover BUY confirmed")
            return StrategyResult(
                strategy=self.name,
                direction="BUY",
                confidence=confidence,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                reasons=reasons,
            )

        # Sell conditions
        sell_conditions = [
            snapshot.ema_fast < snapshot.ema_slow,
            rsi_sell_min <= snapshot.rsi <= rsi_sell_max,
            snapshot.close < snapshot.prev_low,
            snapshot.spread <= max_spread,
            not snapshot.has_open_short,
        ]

        if all(sell_conditions):
            sl = snapshot.close + (snapshot.atr * sl_atr_mult)
            risk = sl - snapshot.close
            tp = snapshot.close - (risk * tp_rr)
            ema_sep = (snapshot.ema_slow - snapshot.ema_fast) / snapshot.ema_slow
            confidence = min(0.5 + ema_sep * 100 + (50 - snapshot.rsi) / 100, 1.0)
            reasons.append("EMA crossover SELL confirmed")
            return StrategyResult(
                strategy=self.name,
                direction="SELL",
                confidence=confidence,
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                reasons=reasons,
            )

        return StrategyResult(
            strategy=self.name, direction="NO_TRADE", confidence=0.0,
            reasons=["EMA crossover conditions not met"],
        )


class BreakoutStrategy:
    """Breakout strategy: detects range-bound markets and trades breakouts."""

    name = StrategyType.BREAKOUT

    def evaluate(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        lookback: int = 20,
        atr_multiplier: float = 1.5,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 2.5,
        max_spread: float = 5.0,
    ) -> StrategyResult:
        """Evaluate breakout strategy."""
        if len(df) < lookback + 5:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Insufficient data for breakout"],
            )

        recent = df.iloc[-lookback - 1:-1]
        resistance = float(recent["high"].max())
        support = float(recent["low"].min())
        price_range = resistance - support
        avg_atr = float(df["atr"].iloc[-lookback:].mean()) if "atr" in df.columns else snapshot.atr

        # Range must be at least 2x ATR to be meaningful
        if price_range < avg_atr * 2:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Range too narrow for breakout"],
            )

        reasons: list[str] = []

        # Bullish breakout: close above resistance
        if snapshot.close > resistance and snapshot.spread <= max_spread and not snapshot.has_open_long:
            breakout_strength = (snapshot.close - resistance) / avg_atr
            confidence = min(0.5 + breakout_strength * 0.2, 1.0)
            sl = snapshot.close - (avg_atr * sl_atr_mult)
            risk = snapshot.close - sl
            tp = snapshot.close + (risk * tp_rr)
            reasons.append(f"Bullish breakout above {resistance:.5f}")
            return StrategyResult(
                strategy=self.name, direction="BUY", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        # Bearish breakout: close below support
        if snapshot.close < support and snapshot.spread <= max_spread and not snapshot.has_open_short:
            breakout_strength = (support - snapshot.close) / avg_atr
            confidence = min(0.5 + breakout_strength * 0.2, 1.0)
            sl = snapshot.close + (avg_atr * sl_atr_mult)
            risk = sl - snapshot.close
            tp = snapshot.close - (risk * tp_rr)
            reasons.append(f"Bearish breakout below {support:.5f}")
            return StrategyResult(
                strategy=self.name, direction="SELL", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        return StrategyResult(
            strategy=self.name, direction="NO_TRADE", confidence=0.0,
            reasons=["No breakout detected"],
        )


class MeanReversionStrategy:
    """Mean reversion strategy: trades when price deviates significantly from the mean."""

    name = StrategyType.MEAN_REVERSION

    def evaluate(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 1.5,
        max_spread: float = 5.0,
    ) -> StrategyResult:
        """Evaluate mean reversion strategy using Bollinger Bands + RSI."""
        if len(df) < bb_period + 5:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Insufficient data for mean reversion"],
            )

        closes = df["close"].astype(float)
        sma = closes.rolling(bb_period).mean()
        std = closes.rolling(bb_period).std()
        upper_band = float(sma.iloc[-1] + bb_std * std.iloc[-1])
        lower_band = float(sma.iloc[-1] - bb_std * std.iloc[-1])
        middle_band = float(sma.iloc[-1])

        # Guard against zero band width (all prices identical)
        band_width = upper_band - lower_band
        if band_width < 1e-10:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Bollinger Band width is zero — insufficient volatility"],
            )

        reasons: list[str] = []

        # Buy: price at/below lower band + RSI oversold
        if (
            snapshot.close <= lower_band
            and snapshot.rsi <= rsi_oversold
            and snapshot.spread <= max_spread
            and not snapshot.has_open_long
        ):
            distance_from_band = (lower_band - snapshot.close) / band_width
            confidence = min(0.5 + distance_from_band + (rsi_oversold - snapshot.rsi) / 100, 1.0)
            sl = snapshot.close - (snapshot.atr * sl_atr_mult)
            tp = middle_band  # Target the mean
            reasons.append(f"Mean reversion BUY: price below lower band ({lower_band:.5f})")
            return StrategyResult(
                strategy=self.name, direction="BUY", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        # Sell: price at/above upper band + RSI overbought
        if (
            snapshot.close >= upper_band
            and snapshot.rsi >= rsi_overbought
            and snapshot.spread <= max_spread
            and not snapshot.has_open_short
        ):
            distance_from_band = (snapshot.close - upper_band) / band_width
            confidence = min(0.5 + distance_from_band + (snapshot.rsi - rsi_overbought) / 100, 1.0)
            sl = snapshot.close + (snapshot.atr * sl_atr_mult)
            tp = middle_band
            reasons.append(f"Mean reversion SELL: price above upper band ({upper_band:.5f})")
            return StrategyResult(
                strategy=self.name, direction="SELL", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        return StrategyResult(
            strategy=self.name, direction="NO_TRADE", confidence=0.0,
            reasons=["Price within normal range"],
        )


class SwingTradingStrategy:
    """
    Swing trading strategy: captures multi-day moves using higher-timeframe
    trend alignment with lower-timeframe entries. Uses EMA 50/200 for trend,
    RSI for momentum, and support/resistance for entries.
    """

    name = StrategyType.SWING

    def evaluate(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        ema_medium: int = 50,
        ema_long: int = 200,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        sl_atr_mult: float = 2.0,
        tp_rr: float = 3.0,
        max_spread: float = 5.0,
    ) -> StrategyResult:
        """Evaluate swing trading strategy."""
        if len(df) < ema_long + 5:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Insufficient data for swing strategy"],
            )

        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)

        # Calculate EMAs for swing
        ema_50 = calculate_ema(closes, ema_medium)
        ema_200 = calculate_ema(closes, ema_long)

        current_ema50 = float(ema_50.iloc[-1])
        current_ema200 = float(ema_200.iloc[-1])
        prev_ema50 = float(ema_50.iloc[-2])
        prev_ema200 = float(ema_200.iloc[-2])

        # Detect swing levels (recent 20-bar highs/lows)
        lookback = min(20, len(df) - 1)
        recent_high = float(highs.iloc[-lookback:].max())
        recent_low = float(lows.iloc[-lookback:].min())

        reasons: list[str] = []

        # Bullish swing: EMA50 > EMA200 (uptrend), price pulls back to EMA50 zone,
        # RSI showing momentum recovery
        bullish_trend = current_ema50 > current_ema200
        price_near_ema50 = snapshot.close <= current_ema50 * 1.005  # Within 0.5% of EMA50
        price_above_ema200 = snapshot.close > current_ema200
        rsi_recovery = snapshot.rsi > rsi_oversold and snapshot.rsi < 55
        ema50_rising = current_ema50 > prev_ema50

        if (
            bullish_trend
            and price_near_ema50
            and price_above_ema200
            and rsi_recovery
            and ema50_rising
            and snapshot.spread <= max_spread
            and not snapshot.has_open_long
        ):
            sl = snapshot.close - (snapshot.atr * sl_atr_mult)
            risk = snapshot.close - sl
            tp = snapshot.close + (risk * tp_rr)
            # Confidence based on trend strength and pullback quality
            trend_strength = (current_ema50 - current_ema200) / current_ema200
            confidence = min(0.5 + trend_strength * 50 + (snapshot.rsi - rsi_oversold) / 100, 1.0)
            reasons.append(f"Swing BUY: uptrend pullback to EMA50 ({current_ema50:.2f})")
            reasons.append(f"EMA50={current_ema50:.2f} > EMA200={current_ema200:.2f}")
            return StrategyResult(
                strategy=self.name, direction="BUY", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        # Bearish swing: EMA50 < EMA200 (downtrend), price rallies to EMA50 zone,
        # RSI showing momentum fading
        bearish_trend = current_ema50 < current_ema200
        price_near_ema50_sell = snapshot.close >= current_ema50 * 0.995
        price_below_ema200 = snapshot.close < current_ema200
        rsi_fading = snapshot.rsi < rsi_overbought and snapshot.rsi > 45
        ema50_falling = current_ema50 < prev_ema50

        if (
            bearish_trend
            and price_near_ema50_sell
            and price_below_ema200
            and rsi_fading
            and ema50_falling
            and snapshot.spread <= max_spread
            and not snapshot.has_open_short
        ):
            sl = snapshot.close + (snapshot.atr * sl_atr_mult)
            risk = sl - snapshot.close
            tp = snapshot.close - (risk * tp_rr)
            trend_strength = (current_ema200 - current_ema50) / current_ema200
            confidence = min(0.5 + trend_strength * 50 + (rsi_overbought - snapshot.rsi) / 100, 1.0)
            reasons.append(f"Swing SELL: downtrend rally to EMA50 ({current_ema50:.2f})")
            reasons.append(f"EMA50={current_ema50:.2f} < EMA200={current_ema200:.2f}")
            return StrategyResult(
                strategy=self.name, direction="SELL", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        return StrategyResult(
            strategy=self.name, direction="NO_TRADE", confidence=0.0,
            reasons=["No swing trade setup detected"],
        )


class ScalpingStrategy:
    """
    Scalping strategy for quick 1-5 minute trades.
    Uses fast EMA crossover (5/13), RSI extremes, and momentum for rapid entries/exits.
    Targets small moves with tight stops for high win rate.
    """

    name = StrategyType.SCALPING

    def evaluate(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        ema_fast_period: int = 5,
        ema_slow_period: int = 13,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        sl_atr_mult: float = 1.5,
        tp_rr: float = 1.5,
        max_spread: float = 3.0,
    ) -> StrategyResult:
        """Evaluate scalping strategy — fast in-and-out trades."""
        if len(df) < ema_slow_period + 5:
            return StrategyResult(
                strategy=self.name, direction="NO_TRADE", confidence=0.0,
                reasons=["Insufficient data for scalping"],
            )

        closes = df["close"].astype(float)
        ema_fast = calculate_ema(closes, ema_fast_period)
        ema_slow = calculate_ema(closes, ema_slow_period)

        curr_fast = float(ema_fast.iloc[-1])
        curr_slow = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])

        # Detect fresh crossover (happened in the last bar)
        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        # Momentum: price velocity over last 3 bars
        if len(closes) >= 4:
            momentum = (float(closes.iloc[-1]) - float(closes.iloc[-4])) / float(closes.iloc[-4]) * 100
        else:
            momentum = 0.0

        reasons: list[str] = []

        # Scalp BUY: fresh bullish crossover + RSI confirms direction + strong momentum
        if (
            bullish_cross
            and snapshot.rsi < rsi_overbought
            and snapshot.rsi > 50  # RSI must confirm bullish bias
            and momentum > 0.05  # Require meaningful momentum (0.05%), not just noise
            and snapshot.spread <= max_spread
            and not snapshot.has_open_long
        ):
            sl = snapshot.close - (snapshot.atr * sl_atr_mult)
            risk = snapshot.close - sl
            tp = snapshot.close + (risk * tp_rr)
            confidence = min(0.6 + abs(momentum) * 0.1 + (snapshot.rsi - 50) / 200, 1.0)
            reasons.append(f"Scalp BUY: EMA5/13 bullish cross, momentum={momentum:.2f}%")
            return StrategyResult(
                strategy=self.name, direction="BUY", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        # Scalp SELL: fresh bearish crossover + RSI confirms direction + strong momentum
        if (
            bearish_cross
            and snapshot.rsi > rsi_oversold
            and snapshot.rsi < 50  # RSI must confirm bearish bias
            and momentum < -0.05  # Require meaningful momentum, not noise
            and snapshot.spread <= max_spread
            and not snapshot.has_open_short
        ):
            sl = snapshot.close + (snapshot.atr * sl_atr_mult)
            risk = sl - snapshot.close
            tp = snapshot.close - (risk * tp_rr)
            confidence = min(0.6 + abs(momentum) * 0.1 + (50 - snapshot.rsi) / 200, 1.0)
            reasons.append(f"Scalp SELL: EMA5/13 bearish cross, momentum={momentum:.2f}%")
            return StrategyResult(
                strategy=self.name, direction="SELL", confidence=confidence,
                stop_loss=round(sl, 5), take_profit=round(tp, 5), reasons=reasons,
            )

        return StrategyResult(
            strategy=self.name, direction="NO_TRADE", confidence=0.0,
            reasons=["No scalping setup detected"],
        )


class StrategySelector:
    """
    Selects the best strategy based on market conditions.
    Evaluates all strategies and picks the one with the highest confidence signal.
    Can also detect market regime (trending vs. ranging) to prioritize strategies.
    Requires minimum confidence of 0.55 to filter out weak signals.
    """

    def __init__(self) -> None:
        self.ema_crossover = EmaCrossoverStrategy()
        self.breakout = BreakoutStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.swing = SwingTradingStrategy()
        self.scalping = ScalpingStrategy()

    def detect_market_regime(self, df: pd.DataFrame, lookback: int = 30) -> str:
        """
        Detect if market is trending or ranging.
        Uses ADX-like calculation and price range analysis.
        Returns: 'trending', 'ranging', or 'volatile'
        """
        if len(df) < lookback + 5:
            return "unknown"

        recent = df.iloc[-lookback:]
        closes = recent["close"].astype(float)

        # Measure trend strength via linear regression R-squared
        x = np.arange(len(closes))
        if closes.std() == 0:
            return "ranging"

        correlation = np.corrcoef(x, closes.values)[0, 1]
        r_squared = correlation ** 2

        # Measure volatility
        returns = closes.pct_change().dropna()
        volatility = returns.std()
        avg_return = abs(returns.mean())

        if r_squared > 0.6:
            return "trending"
        elif volatility > avg_return * 3:
            return "volatile"
        else:
            return "ranging"

    def evaluate_all(
        self,
        df: pd.DataFrame,
        snapshot: MarketSnapshot,
        **kwargs: float,
    ) -> StrategyResult:
        """
        Evaluate all strategies and return the best signal.
        Market regime affects strategy weighting.
        """
        regime = self.detect_market_regime(df)
        logger.info("Market regime detected: %s", regime)

        results = [
            self.ema_crossover.evaluate(df, snapshot, **{
                k: v for k, v in kwargs.items()
                if k in ("rsi_buy_min", "rsi_buy_max", "rsi_sell_min",
                          "rsi_sell_max", "max_spread", "sl_atr_mult", "tp_rr")
            }),
            self.breakout.evaluate(df, snapshot, **{
                k: v for k, v in kwargs.items()
                if k in ("max_spread", "sl_atr_mult", "tp_rr")
            }),
            self.mean_reversion.evaluate(df, snapshot, **{
                k: v for k, v in kwargs.items()
                if k in ("max_spread", "sl_atr_mult", "tp_rr")
            }),
            self.swing.evaluate(df, snapshot, **{
                k: v for k, v in kwargs.items()
                if k in ("max_spread", "sl_atr_mult", "tp_rr")
            }),
            self.scalping.evaluate(df, snapshot, **{
                k: v for k, v in kwargs.items()
                if k in ("max_spread", "sl_atr_mult", "tp_rr")
            }),
        ]

        # Apply regime weighting
        regime_weights = {
            "trending": {StrategyType.EMA_CROSSOVER: 1.3, StrategyType.BREAKOUT: 1.1, StrategyType.MEAN_REVERSION: 0.5, StrategyType.SWING: 1.2, StrategyType.SCALPING: 1.0},
            "ranging": {StrategyType.EMA_CROSSOVER: 0.7, StrategyType.BREAKOUT: 0.8, StrategyType.MEAN_REVERSION: 1.3, StrategyType.SWING: 0.6, StrategyType.SCALPING: 1.2},
            "volatile": {StrategyType.EMA_CROSSOVER: 0.8, StrategyType.BREAKOUT: 1.2, StrategyType.MEAN_REVERSION: 0.9, StrategyType.SWING: 0.7, StrategyType.SCALPING: 1.3},
            "unknown": {StrategyType.EMA_CROSSOVER: 1.0, StrategyType.BREAKOUT: 1.0, StrategyType.MEAN_REVERSION: 1.0, StrategyType.SWING: 1.0, StrategyType.SCALPING: 1.0},
        }
        weights = regime_weights.get(regime, regime_weights["unknown"])

        # Score and pick best
        best: Optional[StrategyResult] = None
        best_score = -1.0

        for r in results:
            if r.direction == "NO_TRADE":
                continue
            weighted_score = r.confidence * weights.get(r.strategy, 1.0)
            if weighted_score > best_score:
                best_score = weighted_score
                best = r

        # Require minimum confidence to avoid weak signals that lead to losses.
        # Lowered from 0.65 to 0.55 because the ADX regime filter now handles
        # choppy-market rejection upstream, so we can let more signals through.
        min_confidence = 0.55
        if best is not None and best.confidence < min_confidence:
            logger.info(
                "Strategy %s signal rejected: confidence %.3f < %.2f minimum",
                best.strategy.value, best.confidence, min_confidence,
            )
            best = None

        if best is None:
            # Combine all reasons for no trade
            all_reasons = []
            for r in results:
                all_reasons.extend(r.reasons)
            return StrategyResult(
                strategy=StrategyType.EMA_CROSSOVER,
                direction="NO_TRADE",
                confidence=0.0,
                reasons=all_reasons,
            )

        best.reasons.append(f"Market regime: {regime}")
        best.reasons.append(f"Weighted score: {best_score:.3f}")
        logger.info(
            "Best strategy: %s (%s) confidence=%.3f weighted=%.3f",
            best.strategy.value, best.direction, best.confidence, best_score,
        )
        return best
