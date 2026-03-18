"""
Unit tests for signal generator.
"""

import numpy as np
import pandas as pd
import pytest

from app.strategy.signals import SignalGenerator


def _make_bullish_df(n: int = 100) -> pd.DataFrame:
    """Create OHLCV data with a clear uptrend."""
    np.random.seed(42)
    base = 1900.0
    trend = np.linspace(0, 50, n)
    noise = np.random.randn(n) * 0.5
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    opens = closes - np.random.rand(n) * 0.2
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(1000, 10000, n),
    })


def _make_bearish_df(n: int = 100) -> pd.DataFrame:
    """Create OHLCV data with a clear downtrend."""
    np.random.seed(42)
    base = 2000.0
    trend = np.linspace(0, -50, n)
    noise = np.random.randn(n) * 0.5
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    opens = closes + np.random.rand(n) * 0.2
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(1000, 10000, n),
    })


class TestSignalGenerator:
    def test_insufficient_data(self) -> None:
        gen = SignalGenerator()
        df = pd.DataFrame({
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
        })
        signal = gen.generate(df, "TEST", "HOUR", 0.5)
        assert signal.direction == "NO_TRADE"
        assert "Insufficient" in signal.reasons[0]

    def test_signal_has_required_fields(self) -> None:
        gen = SignalGenerator()
        df = _make_bullish_df()
        signal = gen.generate(df, "XAUUSD", "HOUR", 1.0)
        assert signal.symbol == "XAUUSD"
        assert signal.timeframe == "HOUR"
        assert signal.direction in ("BUY", "SELL", "NO_TRADE")
        assert signal.close_price > 0
        assert signal.ema_fast > 0
        assert signal.ema_slow > 0
        assert 0 <= signal.rsi <= 100

    def test_buy_signal_has_sl_tp(self) -> None:
        gen = SignalGenerator()
        df = _make_bullish_df()
        signal = gen.generate(df, "XAUUSD", "HOUR", 0.5)
        if signal.direction == "BUY":
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.stop_loss < signal.close_price
            assert signal.take_profit > signal.close_price

    def test_no_trade_with_high_spread(self) -> None:
        gen = SignalGenerator()
        df = _make_bullish_df()
        signal = gen.generate(df, "XAUUSD", "HOUR", spread=999.0)
        assert signal.direction == "NO_TRADE"

    def test_no_trade_with_open_position(self) -> None:
        gen = SignalGenerator()
        df = _make_bullish_df()
        signal = gen.generate(
            df, "XAUUSD", "HOUR", 0.5,
            has_open_long=True, has_open_short=True,
        )
        assert signal.direction == "NO_TRADE"
