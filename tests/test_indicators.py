"""
Unit tests for technical indicators.
"""

import pandas as pd
import numpy as np
import pytest

from app.strategy.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    detect_breakout_high,
    detect_breakout_low,
    add_all_indicators,
)


def _make_ohlcv(n: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    changes = np.random.randn(n) * 0.5
    closes = base_price + np.cumsum(changes)
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    opens = closes + np.random.randn(n) * 0.1
    volumes = np.random.randint(1000, 10000, n)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestEMA:
    def test_ema_length(self) -> None:
        df = _make_ohlcv()
        ema = calculate_ema(df["close"], 20)
        assert len(ema) == len(df)

    def test_ema_values_differ_by_period(self) -> None:
        df = _make_ohlcv()
        ema_fast = calculate_ema(df["close"], 10)
        ema_slow = calculate_ema(df["close"], 50)
        assert not ema_fast.equals(ema_slow)

    def test_ema_follows_price(self) -> None:
        prices = pd.Series([10.0] * 50 + [20.0] * 50)
        ema = calculate_ema(prices, 10)
        assert ema.iloc[-1] > 19.0


class TestRSI:
    def test_rsi_range(self) -> None:
        df = _make_ohlcv()
        rsi = calculate_rsi(df["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_length(self) -> None:
        df = _make_ohlcv()
        rsi = calculate_rsi(df["close"], 14)
        assert len(rsi) == len(df)

    def test_rsi_uptrend_high(self) -> None:
        # Use a series with small noise to avoid NaN from zero-loss division
        np.random.seed(0)
        prices = pd.Series(range(1, 101), dtype=float) + np.random.rand(100) * 0.01
        rsi = calculate_rsi(prices, 14)
        valid = rsi.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 80


class TestATR:
    def test_atr_positive(self) -> None:
        df = _make_ohlcv()
        atr = calculate_atr(df["high"], df["low"], df["close"], 14)
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_length(self) -> None:
        df = _make_ohlcv()
        atr = calculate_atr(df["high"], df["low"], df["close"], 14)
        assert len(atr) == len(df)


class TestBreakout:
    def test_breakout_high(self) -> None:
        close = pd.Series([10.0, 11.0, 12.0, 9.0])
        high = pd.Series([10.5, 11.5, 12.5, 9.5])
        result = detect_breakout_high(close, high)
        assert result.iloc[2] is np.True_  # 12 > 11.5

    def test_breakout_low(self) -> None:
        close = pd.Series([10.0, 9.0, 8.0, 11.0])
        low = pd.Series([9.5, 8.5, 7.5, 10.5])
        result = detect_breakout_low(close, low)
        assert result.iloc[2] is np.True_  # 8 < 8.5


class TestAddAllIndicators:
    def test_columns_added(self) -> None:
        df = _make_ohlcv()
        result = add_all_indicators(df)
        expected_cols = {"ema_fast", "ema_slow", "rsi", "atr", "breakout_high", "breakout_low"}
        assert expected_cols.issubset(set(result.columns))

    def test_no_mutation(self) -> None:
        df = _make_ohlcv()
        original_cols = set(df.columns)
        add_all_indicators(df)
        assert set(df.columns) == original_cols
