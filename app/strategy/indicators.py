"""
Technical indicators for trading strategy.
Implements EMA, RSI, ATR, and breakout detection.
"""

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    Uses the standard Wilder smoothing method.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # When avg_loss is 0, RSI is 100 (pure uptrend); avoid division by zero
    rsi = pd.Series(index=series.index, dtype=float)
    zero_loss = avg_loss == 0
    nonzero_loss = ~zero_loss

    rs = avg_gain[nonzero_loss] / avg_loss[nonzero_loss]
    rsi[nonzero_loss] = 100.0 - (100.0 / (1.0 + rs))
    rsi[zero_loss & avg_gain.notna()] = 100.0
    # Keep NaN where min_periods not yet met
    rsi[avg_gain.isna()] = np.nan
    return rsi


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average Directional Index (ADX).

    ADX measures trend strength regardless of direction:
    - ADX < 20: weak/no trend (choppy market — avoid trading)
    - ADX 20-40: developing/moderate trend
    - ADX > 40: strong trend

    Uses Wilder smoothing (same as RSI/ATR).
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # +DM and -DM
    plus_dm = high - prev_high
    minus_dm = prev_low - low

    # Only keep positive values where +DM > -DM (and vice versa)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing (EWM with alpha=1/period)
    smoothed_tr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr

    # DX = |+DI - -DI| / (+DI + -DI) * 100
    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.where(di_sum != 0, 1.0) * 100

    # ADX = smoothed DX
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx


def detect_breakout_high(close: pd.Series, high: pd.Series) -> pd.Series:
    """
    Detect if the current candle closes above the previous candle's high.
    Returns a boolean Series.
    """
    prev_high = high.shift(1)
    return close > prev_high


def detect_breakout_low(close: pd.Series, low: pd.Series) -> pd.Series:
    """
    Detect if the current candle closes below the previous candle's low.
    Returns a boolean Series.
    """
    prev_low = low.shift(1)
    return close < prev_low


def add_all_indicators(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Add all required indicators to an OHLCV DataFrame.
    Expected columns: open, high, low, close, volume (optional).
    """
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], ema_fast)
    df["ema_slow"] = calculate_ema(df["close"], ema_slow)
    df["rsi"] = calculate_rsi(df["close"], rsi_period)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], atr_period)
    df["adx"] = calculate_adx(df["high"], df["low"], df["close"])
    df["breakout_high"] = detect_breakout_high(df["close"], df["high"])
    df["breakout_low"] = detect_breakout_low(df["close"], df["low"])
    return df
