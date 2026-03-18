"""
Trading rules that evaluate conditions for buy/sell signals.
Each rule returns True/False based on indicator values.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    """A snapshot of the current market state with indicators."""
    close: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    prev_high: float
    prev_low: float
    spread: float
    has_open_long: bool = False
    has_open_short: bool = False


@dataclass
class RuleResult:
    """Result of evaluating trading rules."""
    passed: bool
    reason: str


def check_ema_bullish(snapshot: MarketSnapshot) -> RuleResult:
    """Check if fast EMA is above slow EMA (bullish trend)."""
    passed = snapshot.ema_fast > snapshot.ema_slow
    reason = (
        f"EMA{20} ({snapshot.ema_fast:.5f}) > EMA{50} ({snapshot.ema_slow:.5f})"
        if passed
        else f"EMA{20} ({snapshot.ema_fast:.5f}) <= EMA{50} ({snapshot.ema_slow:.5f})"
    )
    return RuleResult(passed=passed, reason=reason)


def check_ema_bearish(snapshot: MarketSnapshot) -> RuleResult:
    """Check if fast EMA is below slow EMA (bearish trend)."""
    passed = snapshot.ema_fast < snapshot.ema_slow
    reason = (
        f"EMA{20} ({snapshot.ema_fast:.5f}) < EMA{50} ({snapshot.ema_slow:.5f})"
        if passed
        else f"EMA{20} ({snapshot.ema_fast:.5f}) >= EMA{50} ({snapshot.ema_slow:.5f})"
    )
    return RuleResult(passed=passed, reason=reason)


def check_rsi_buy_zone(
    snapshot: MarketSnapshot,
    rsi_min: float = 55,
    rsi_max: float = 70,
) -> RuleResult:
    """Check if RSI is in the buy zone."""
    passed = rsi_min <= snapshot.rsi <= rsi_max
    reason = (
        f"RSI ({snapshot.rsi:.2f}) in buy zone [{rsi_min}, {rsi_max}]"
        if passed
        else f"RSI ({snapshot.rsi:.2f}) outside buy zone [{rsi_min}, {rsi_max}]"
    )
    return RuleResult(passed=passed, reason=reason)


def check_rsi_sell_zone(
    snapshot: MarketSnapshot,
    rsi_min: float = 30,
    rsi_max: float = 45,
) -> RuleResult:
    """Check if RSI is in the sell zone."""
    passed = rsi_min <= snapshot.rsi <= rsi_max
    reason = (
        f"RSI ({snapshot.rsi:.2f}) in sell zone [{rsi_min}, {rsi_max}]"
        if passed
        else f"RSI ({snapshot.rsi:.2f}) outside sell zone [{rsi_min}, {rsi_max}]"
    )
    return RuleResult(passed=passed, reason=reason)


def check_breakout_high(snapshot: MarketSnapshot) -> RuleResult:
    """Check if current close is above previous high."""
    passed = snapshot.close > snapshot.prev_high
    reason = (
        f"Close ({snapshot.close:.5f}) > prev high ({snapshot.prev_high:.5f})"
        if passed
        else f"Close ({snapshot.close:.5f}) <= prev high ({snapshot.prev_high:.5f})"
    )
    return RuleResult(passed=passed, reason=reason)


def check_breakout_low(snapshot: MarketSnapshot) -> RuleResult:
    """Check if current close is below previous low."""
    passed = snapshot.close < snapshot.prev_low
    reason = (
        f"Close ({snapshot.close:.5f}) < prev low ({snapshot.prev_low:.5f})"
        if passed
        else f"Close ({snapshot.close:.5f}) >= prev low ({snapshot.prev_low:.5f})"
    )
    return RuleResult(passed=passed, reason=reason)


def check_spread(snapshot: MarketSnapshot, max_spread: float = 5.0) -> RuleResult:
    """Check if spread is within acceptable range."""
    passed = snapshot.spread <= max_spread
    reason = (
        f"Spread ({snapshot.spread:.5f}) <= max ({max_spread})"
        if passed
        else f"Spread ({snapshot.spread:.5f}) > max ({max_spread})"
    )
    return RuleResult(passed=passed, reason=reason)


def check_no_open_long(snapshot: MarketSnapshot) -> RuleResult:
    """Check that there is no existing long position."""
    passed = not snapshot.has_open_long
    reason = "No open long position" if passed else "Already has open long position"
    return RuleResult(passed=passed, reason=reason)


def check_no_open_short(snapshot: MarketSnapshot) -> RuleResult:
    """Check that there is no existing short position."""
    passed = not snapshot.has_open_short
    reason = "No open short position" if passed else "Already has open short position"
    return RuleResult(passed=passed, reason=reason)


def evaluate_buy_rules(
    snapshot: MarketSnapshot,
    rsi_min: float = 55,
    rsi_max: float = 70,
    max_spread: float = 5.0,
) -> tuple[bool, list[RuleResult]]:
    """
    Evaluate all buy conditions.
    Returns (all_passed, list_of_results).
    """
    results = [
        check_ema_bullish(snapshot),
        check_rsi_buy_zone(snapshot, rsi_min, rsi_max),
        check_breakout_high(snapshot),
        check_spread(snapshot, max_spread),
        check_no_open_long(snapshot),
    ]
    all_passed = all(r.passed for r in results)
    return all_passed, results


def evaluate_sell_rules(
    snapshot: MarketSnapshot,
    rsi_min: float = 30,
    rsi_max: float = 45,
    max_spread: float = 5.0,
) -> tuple[bool, list[RuleResult]]:
    """
    Evaluate all sell conditions.
    Returns (all_passed, list_of_results).
    """
    results = [
        check_ema_bearish(snapshot),
        check_rsi_sell_zone(snapshot, rsi_min, rsi_max),
        check_breakout_low(snapshot),
        check_spread(snapshot, max_spread),
        check_no_open_short(snapshot),
    ]
    all_passed = all(r.passed for r in results)
    return all_passed, results
