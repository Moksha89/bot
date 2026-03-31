"""
Unit tests for trading rules.
"""

import pytest

from app.strategy.rules import (
    MarketSnapshot,
    check_ema_bullish,
    check_ema_bearish,
    check_rsi_buy_zone,
    check_rsi_sell_zone,
    check_breakout_high,
    check_breakout_low,
    check_spread,
    check_no_open_long,
    check_no_open_short,
    evaluate_buy_rules,
    evaluate_sell_rules,
)


def _buy_snapshot() -> MarketSnapshot:
    """Create a snapshot that should pass all buy rules."""
    return MarketSnapshot(
        close=1950.0,
        ema_fast=1945.0,
        ema_slow=1940.0,
        rsi=60.0,
        atr=5.0,
        prev_high=1948.0,
        prev_low=1935.0,
        spread=1.0,
        has_open_long=False,
        has_open_short=False,
    )


def _sell_snapshot() -> MarketSnapshot:
    """Create a snapshot that should pass all sell rules."""
    return MarketSnapshot(
        close=1930.0,
        ema_fast=1935.0,
        ema_slow=1940.0,
        rsi=38.0,
        atr=5.0,
        prev_high=1945.0,
        prev_low=1932.0,
        spread=1.0,
        has_open_long=False,
        has_open_short=False,
    )


class TestEMARule:
    def test_bullish_when_fast_above_slow(self) -> None:
        snap = _buy_snapshot()
        result = check_ema_bullish(snap)
        assert result.passed is True

    def test_not_bullish_when_fast_below_slow(self) -> None:
        snap = _sell_snapshot()
        result = check_ema_bullish(snap)
        assert result.passed is False

    def test_bearish_when_fast_below_slow(self) -> None:
        snap = _sell_snapshot()
        result = check_ema_bearish(snap)
        assert result.passed is True

    def test_not_bearish_when_fast_above_slow(self) -> None:
        snap = _buy_snapshot()
        result = check_ema_bearish(snap)
        assert result.passed is False


class TestRSIRule:
    def test_rsi_in_buy_zone(self) -> None:
        snap = _buy_snapshot()
        result = check_rsi_buy_zone(snap, 55, 70)
        assert result.passed is True

    def test_rsi_below_buy_zone(self) -> None:
        snap = _buy_snapshot()
        snap.rsi = 50.0
        result = check_rsi_buy_zone(snap, 55, 70)
        assert result.passed is False

    def test_rsi_above_buy_zone(self) -> None:
        snap = _buy_snapshot()
        snap.rsi = 75.0
        result = check_rsi_buy_zone(snap, 55, 70)
        assert result.passed is False

    def test_rsi_in_sell_zone(self) -> None:
        snap = _sell_snapshot()
        result = check_rsi_sell_zone(snap, 30, 45)
        assert result.passed is True

    def test_rsi_outside_sell_zone(self) -> None:
        snap = _sell_snapshot()
        snap.rsi = 50.0
        result = check_rsi_sell_zone(snap, 30, 45)
        assert result.passed is False


class TestBreakoutRule:
    def test_breakout_high(self) -> None:
        snap = _buy_snapshot()
        result = check_breakout_high(snap)
        assert result.passed is True

    def test_no_breakout_high(self) -> None:
        snap = _buy_snapshot()
        snap.close = 1947.0
        result = check_breakout_high(snap)
        assert result.passed is False

    def test_breakout_low(self) -> None:
        snap = _sell_snapshot()
        result = check_breakout_low(snap)
        assert result.passed is True

    def test_no_breakout_low(self) -> None:
        snap = _sell_snapshot()
        snap.close = 1933.0
        result = check_breakout_low(snap)
        assert result.passed is False


class TestSpreadRule:
    def test_acceptable_spread(self) -> None:
        snap = _buy_snapshot()
        result = check_spread(snap, max_spread=5.0)
        assert result.passed is True

    def test_excessive_spread(self) -> None:
        snap = _buy_snapshot()
        snap.spread = 10.0
        result = check_spread(snap, max_spread=5.0)
        assert result.passed is False


class TestOpenPositionRules:
    def test_no_open_long_passes(self) -> None:
        snap = _buy_snapshot()
        assert check_no_open_long(snap).passed is True

    def test_has_open_long_fails(self) -> None:
        snap = _buy_snapshot()
        snap.has_open_long = True
        assert check_no_open_long(snap).passed is False

    def test_no_open_short_passes(self) -> None:
        snap = _sell_snapshot()
        assert check_no_open_short(snap).passed is True

    def test_has_open_short_fails(self) -> None:
        snap = _sell_snapshot()
        snap.has_open_short = True
        assert check_no_open_short(snap).passed is False


class TestEvaluateBuyRules:
    def test_all_buy_rules_pass(self) -> None:
        snap = _buy_snapshot()
        passed, results = evaluate_buy_rules(snap)
        assert passed is True
        assert all(r.passed for r in results)

    def test_buy_rejected_with_open_long(self) -> None:
        snap = _buy_snapshot()
        snap.has_open_long = True
        passed, results = evaluate_buy_rules(snap)
        assert passed is False

    def test_buy_rejected_with_bad_rsi(self) -> None:
        snap = _buy_snapshot()
        snap.rsi = 80.0
        passed, results = evaluate_buy_rules(snap)
        assert passed is False


class TestEvaluateSellRules:
    def test_all_sell_rules_pass(self) -> None:
        snap = _sell_snapshot()
        passed, results = evaluate_sell_rules(snap)
        assert passed is True
        assert all(r.passed for r in results)

    def test_sell_rejected_with_open_short(self) -> None:
        snap = _sell_snapshot()
        snap.has_open_short = True
        passed, results = evaluate_sell_rules(snap)
        assert passed is False
