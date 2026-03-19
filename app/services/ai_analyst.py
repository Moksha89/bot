"""
AI-powered market analysis module using OpenRouter API.
Analyzes chart data, market conditions, and provides trade recommendations
before every trade decision.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd

from app.config import settings
from app.strategy.indicators import calculate_ema, calculate_rsi, calculate_atr

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Default model - can be overridden via env
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


class AIAnalyst:
    """
    Uses OpenRouter LLM API to analyze market data and provide
    AI-powered trade confirmation before execution.
    Supports both text-based and vision-based analysis.
    """

    def __init__(self) -> None:
        self.api_key = settings.openrouter_api_key
        self.model = settings.ai_model or DEFAULT_MODEL
        self.vision_model = settings.ai_vision_model or self.model
        self.enabled = bool(self.api_key)

    async def analyze_market(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        signal_direction: str,
        indicators: dict,
        spread: float,
        account_balance: float,
    ) -> dict:
        """
        Send market data to AI for comprehensive analysis before trading.

        Returns:
            dict with keys:
                - recommendation: 'CONFIRM', 'REJECT', or 'HOLD'
                - confidence: float 0-100
                - analysis: str (detailed reasoning)
                - risk_notes: str (any risk warnings)
        """
        if not self.enabled:
            logger.debug("AI analyst disabled (no API key)")
            return {
                "recommendation": "CONFIRM",
                "confidence": 50.0,
                "analysis": "AI analysis disabled - proceeding with rule-based signal only",
                "risk_notes": "No AI validation performed",
            }

        # Prepare market data summary for the AI
        market_summary = self._prepare_market_summary(
            symbol, timeframe, df, signal_direction, indicators, spread, account_balance
        )

        prompt = self._build_analysis_prompt(market_summary)

        try:
            response = await self._call_openrouter(prompt)
            parsed = self._parse_ai_response(response)
            logger.info(
                "AI analysis for %s %s: %s (confidence: %.1f%%)",
                signal_direction,
                symbol,
                parsed["recommendation"],
                parsed["confidence"],
            )
            return parsed
        except Exception as e:
            logger.error("AI analysis failed: %s", e)
            return {
                "recommendation": "CONFIRM",
                "confidence": 50.0,
                "analysis": f"AI analysis error: {e}. Falling back to rule-based signal.",
                "risk_notes": "AI analysis unavailable",
            }

    def _prepare_market_summary(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        signal_direction: str,
        indicators: dict,
        spread: float,
        account_balance: float,
    ) -> dict:
        """Prepare a deep-research market data summary for the AI."""
        recent = df.tail(50)
        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)

        # Price action — last 10 candles for detail
        candles = []
        for _, row in df.tail(10).iterrows():
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            body = "GREEN" if c >= o else "RED"
            candles.append({
                "open": round(o, 5), "high": round(h, 5),
                "low": round(l, 5), "close": round(c, 5),
                "body": body,
            })

        current_close = float(df.iloc[-1]["close"])
        prev_close = float(df.iloc[-2]["close"])

        # ---- Deep indicators ----
        # Bollinger Bands (20-period, 2 std)
        sma20 = closes.rolling(20).mean()
        std20 = closes.rolling(20).std()
        bb_upper = float((sma20 + 2 * std20).iloc[-1]) if len(df) >= 20 else None
        bb_lower = float((sma20 - 2 * std20).iloc[-1]) if len(df) >= 20 else None
        bb_mid = float(sma20.iloc[-1]) if len(df) >= 20 else None
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 3) if bb_mid and bb_mid > 0 else None

        # MACD (12, 26, 9)
        ema12 = calculate_ema(closes, 12)
        ema26 = calculate_ema(closes, 26)
        macd_line = ema12 - ema26
        macd_signal = calculate_ema(macd_line, 9)
        macd_hist = macd_line - macd_signal
        macd_data = {
            "macd": round(float(macd_line.iloc[-1]), 5),
            "signal": round(float(macd_signal.iloc[-1]), 5),
            "histogram": round(float(macd_hist.iloc[-1]), 5),
            "histogram_prev": round(float(macd_hist.iloc[-2]), 5),
            "crossover": "bullish" if float(macd_hist.iloc[-1]) > 0 > float(macd_hist.iloc[-2]) else
                         "bearish" if float(macd_hist.iloc[-1]) < 0 < float(macd_hist.iloc[-2]) else "none",
        }

        # Multi-period RSI
        rsi_14 = calculate_rsi(closes, 14)
        rsi_7 = calculate_rsi(closes, 7)
        rsi_data = {
            "rsi_14": round(float(rsi_14.iloc[-1]), 2),
            "rsi_7": round(float(rsi_7.iloc[-1]), 2),
            "rsi_14_prev": round(float(rsi_14.iloc[-2]), 2),
            "divergence": self._detect_rsi_divergence(closes, rsi_14),
        }

        # ATR and volatility
        atr = calculate_atr(highs, lows, closes)
        atr_val = float(atr.iloc[-1])
        atr_pct = round(atr_val / current_close * 100, 3) if current_close > 0 else 0

        # Support / Resistance (pivot points from last 50 candles)
        last_50_high = float(recent["high"].max())
        last_50_low = float(recent["low"].min())
        pivot = (last_50_high + last_50_low + current_close) / 3
        r1 = 2 * pivot - last_50_low
        s1 = 2 * pivot - last_50_high
        r2 = pivot + (last_50_high - last_50_low)
        s2 = pivot - (last_50_high - last_50_low)

        # Fibonacci retracement levels (from recent swing)
        fib_data = self._calculate_fibonacci(highs, lows, signal_direction)

        # Candlestick pattern detection
        patterns = self._detect_candlestick_patterns(df)

        # Trend analysis
        ema_20 = calculate_ema(closes, 20)
        ema_50 = calculate_ema(closes, 50)
        ema_200 = calculate_ema(closes, 200) if len(df) >= 200 else None
        trend_data = {
            "ema_20": round(float(ema_20.iloc[-1]), 5),
            "ema_50": round(float(ema_50.iloc[-1]), 5),
            "ema_200": round(float(ema_200.iloc[-1]), 5) if ema_200 is not None else "N/A",
            "price_vs_ema20": "above" if current_close > float(ema_20.iloc[-1]) else "below",
            "price_vs_ema50": "above" if current_close > float(ema_50.iloc[-1]) else "below",
            "ema20_vs_ema50": "bullish" if float(ema_20.iloc[-1]) > float(ema_50.iloc[-1]) else "bearish",
            "trend_strength": round(abs(float(ema_20.iloc[-1]) - float(ema_50.iloc[-1])) / current_close * 100, 3) if current_close > 0 else 0,
        }

        # Volume analysis
        volume_data = None
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            avg_vol_20 = float(vol.tail(20).mean())
            latest_vol = float(vol.iloc[-1])
            volume_data = {
                "latest": latest_vol,
                "average_20": round(avg_vol_20, 0),
                "ratio": round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
                "trend": "increasing" if float(vol.tail(5).mean()) > avg_vol_20 else "decreasing",
            }

        # Price change stats — multiple timeframes
        changes = {}
        for n, label in [(1, "1_candle"), (5, "5_candle"), (10, "10_candle"), (20, "20_candle")]:
            if len(df) > n:
                ref = float(df.iloc[-(n + 1)]["close"])
                changes[f"{label}_pct"] = round((current_close - ref) / ref * 100, 4)

        # Higher-highs / lower-lows structure (last 10 candles)
        hh_ll = self._analyze_structure(highs.tail(10), lows.tail(10))

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_direction": signal_direction,
            "current_price": current_close,
            "spread": spread,
            "spread_vs_atr_pct": round(spread / atr_val * 100, 1) if atr_val > 0 else 999,
            "account_balance": account_balance,
            "last_10_candles": candles,
            "indicators": indicators,
            "trend": trend_data,
            "macd": macd_data,
            "rsi": rsi_data,
            "bollinger_bands": {
                "upper": round(bb_upper, 5) if bb_upper else None,
                "middle": round(bb_mid, 5) if bb_mid else None,
                "lower": round(bb_lower, 5) if bb_lower else None,
                "width_pct": bb_width,
                "price_position": "above_upper" if bb_upper and current_close > bb_upper else
                                  "below_lower" if bb_lower and current_close < bb_lower else
                                  "upper_half" if bb_mid and current_close > bb_mid else "lower_half",
            },
            "volatility": {"atr": round(atr_val, 5), "atr_pct": atr_pct},
            "support_resistance": {
                "pivot": round(pivot, 5),
                "r1": round(r1, 5), "r2": round(r2, 5),
                "s1": round(s1, 5), "s2": round(s2, 5),
                "recent_high_50": round(last_50_high, 5),
                "recent_low_50": round(last_50_low, 5),
            },
            "fibonacci": fib_data,
            "candlestick_patterns": patterns,
            "price_structure": hh_ll,
            "price_change": changes,
            "volume": volume_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ---- Deep research helper methods ----

    def _detect_rsi_divergence(
        self, closes: pd.Series, rsi: pd.Series, lookback: int = 14,
    ) -> str:
        """Detect bullish/bearish RSI divergence."""
        if len(closes) < lookback + 2:
            return "insufficient_data"
        price_tail = closes.iloc[-lookback:]
        rsi_tail = rsi.iloc[-lookback:].dropna()
        if len(rsi_tail) < lookback:
            return "insufficient_data"
        price_making_lower_low = float(price_tail.iloc[-1]) < float(price_tail.iloc[:-1].min())
        rsi_making_higher_low = float(rsi_tail.iloc[-1]) > float(rsi_tail.iloc[:-1].min())
        if price_making_lower_low and rsi_making_higher_low:
            return "bullish_divergence"
        price_making_higher_high = float(price_tail.iloc[-1]) > float(price_tail.iloc[:-1].max())
        rsi_making_lower_high = float(rsi_tail.iloc[-1]) < float(rsi_tail.iloc[:-1].max())
        if price_making_higher_high and rsi_making_lower_high:
            return "bearish_divergence"
        return "none"

    def _calculate_fibonacci(
        self, highs: pd.Series, lows: pd.Series, direction: str,
    ) -> dict:
        """Calculate Fibonacci retracement levels from recent swing."""
        swing_high = float(highs.tail(50).max())
        swing_low = float(lows.tail(50).min())
        diff = swing_high - swing_low
        if diff <= 0:
            return {}
        levels = {}
        for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
            if direction == "BUY":  # retracement from high
                levels[f"fib_{ratio}"] = round(swing_high - diff * ratio, 5)
            else:
                levels[f"fib_{ratio}"] = round(swing_low + diff * ratio, 5)
        levels["swing_high"] = round(swing_high, 5)
        levels["swing_low"] = round(swing_low, 5)
        return levels

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> list[str]:
        """Detect common candlestick patterns in the last 3 candles."""
        patterns: list[str] = []
        if len(df) < 3:
            return patterns
        c = df.iloc[-1]
        p = df.iloc[-2]
        pp = df.iloc[-3]
        o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        po, ph, pl, pcl = float(p["open"]), float(p["high"]), float(p["low"]), float(p["close"])
        body = abs(cl - o)
        full_range = h - l if h != l else 0.0001
        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - l

        # Doji
        if body / full_range < 0.1:
            patterns.append("doji")
        # Hammer (bullish)
        if lower_wick > 2 * body and upper_wick < body and cl > o:
            patterns.append("hammer")
        # Shooting star (bearish)
        if upper_wick > 2 * body and lower_wick < body and cl < o:
            patterns.append("shooting_star")
        # Bullish engulfing
        if pcl < po and cl > o and cl > po and o < pcl:
            patterns.append("bullish_engulfing")
        # Bearish engulfing
        if pcl > po and cl < o and cl < po and o > pcl:
            patterns.append("bearish_engulfing")
        # Morning star
        if float(pp["close"]) < float(pp["open"]) and abs(pcl - po) / (ph - pl if ph != pl else 0.0001) < 0.3 and cl > o:
            patterns.append("morning_star")
        # Evening star
        if float(pp["close"]) > float(pp["open"]) and abs(pcl - po) / (ph - pl if ph != pl else 0.0001) < 0.3 and cl < o:
            patterns.append("evening_star")

        return patterns

    def _analyze_structure(
        self, highs: pd.Series, lows: pd.Series,
    ) -> dict:
        """Analyze higher-highs/lower-lows structure."""
        h_list = highs.tolist()
        l_list = lows.tolist()
        hh = sum(1 for i in range(1, len(h_list)) if h_list[i] > h_list[i - 1])
        ll = sum(1 for i in range(1, len(l_list)) if l_list[i] < l_list[i - 1])
        hl = sum(1 for i in range(1, len(l_list)) if l_list[i] > l_list[i - 1])
        lh = sum(1 for i in range(1, len(h_list)) if h_list[i] < h_list[i - 1])
        total = len(h_list) - 1 if len(h_list) > 1 else 1
        if hh / total >= 0.6 and hl / total >= 0.6:
            structure = "uptrend"
        elif ll / total >= 0.6 and lh / total >= 0.6:
            structure = "downtrend"
        else:
            structure = "ranging"
        return {
            "structure": structure,
            "higher_highs": hh,
            "lower_lows": ll,
            "higher_lows": hl,
            "lower_highs": lh,
        }

    def _build_analysis_prompt(self, market_summary: dict) -> str:
        """Build the deep-research analysis prompt for the AI."""
        signal_dir = market_summary.get("signal_direction", "N/A")
        data_json = json.dumps(market_summary, indent=2)
        return f"""You are a balanced professional trading analyst. Your job is to evaluate trade setups fairly — not too strict, not too loose.

MARKET DATA:
{data_json}

SCORE EACH FACTOR (0-10 points each). IMPORTANT: 5 = neutral/unclear. Only score below 5 if there is CLEAR evidence AGAINST the trade. Score above 5 if there is evidence FOR the trade.

1. TREND (0-10): Is the proposed {signal_dir} aligned with EMA 20/50? Score 5 if EMAs are close together (flat/unclear). Score 7-10 if EMAs clearly support direction. Score 0-3 only if EMAs clearly oppose direction.
2. MOMENTUM (0-10): RSI-14 and RSI-7 confirm direction? Score 5 if RSI is near 50 (neutral). Score 7+ if RSI clearly supports direction. Score below 3 only if RSI is extreme against the trade.
3. VOLATILITY (0-10): Is spread vs ATR acceptable? Score 5 if spread/ATR is 30-50%. Score 7+ if below 30%. Score below 3 only if spread/ATR > 60%.
4. SUPPORT/RESISTANCE (0-10): Room to move toward target? Score 5 if no clear S/R nearby. Score 7+ if entry is near favorable S/R level.
5. PATTERNS (0-10): Any confirming candlestick patterns? Score 5 if no clear patterns (neutral). Score 7+ for confirming patterns. Score below 3 only for strong reversal patterns against trade.
6. PRICE STRUCTURE (0-10): Higher-highs/lows for BUY? Lower-highs/lows for SELL? Score 5 if ranging/mixed.
7. RISK-REWARD (0-10): Reasonable reward-to-risk? Score 5 if ~1.5:1. Score 7+ if 2:1 or better.

RULES:
- Total score out of 70. Percentage = total/70*100.
- If most factors are neutral (5/10), total should be ~35/70 = 50%. This is a CONFIRM.
- CONFIRM if score >= 43% (30+ points). Most trade setups with neutral-to-positive signals should CONFIRM.
- HOLD if 30-43% (21-29 points). Mixed signals.
- REJECT only if score < 30% (below 21 points) — meaning multiple factors clearly oppose the trade.
- Do NOT default to any fixed number. Calculate honestly from the data.

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{{{{
    "recommendation": "CONFIRM" or "REJECT" or "HOLD",
    "confidence": <calculated percentage 0-100>,
    "analysis": "<2-3 sentences explaining your scoring>",
    "risk_notes": "<specific risk warnings>",
    "factor_scores": "T:<t>/10 M:<m>/10 V:<v>/10 SR:<sr>/10 P:<p>/10 PS:<ps>/10 RR:<rr>/10 = <total>/70",
    "trade_quality": "A+" or "A" or "B" or "C" or "D"
}}}}

GRADING:
- A+ (CONFIRM, 85-100%): 60-70 points — exceptional setup
- A (CONFIRM, 70-85%): 49-59 points — strong setup
- B (CONFIRM, 43-70%): 30-48 points — decent setup, trade it
- C (HOLD, 30-43%): 21-29 points — weak, borderline
- D (REJECT, 0-30%): 0-20 points — clearly bad, do not trade"""

    async def _call_openrouter(self, prompt: str, model: str = "") -> str:
        """Call OpenRouter API and return the response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://capital-trading-bot.local",
            "X-Title": "Capital Trading Bot",
        }
        payload = {
            "model": model or self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a balanced professional trading analyst. "
                        "You score each factor 0-10 where 5 means neutral/unclear. "
                        "Only score below 5 when evidence clearly opposes the trade. "
                        "A setup with mostly neutral factors (5/10 each) scores ~50% and should CONFIRM. "
                        "You REJECT only when multiple factors clearly oppose the trade (score < 30%). "
                        "Calculate the actual score from data. Do not default to any fixed number. "
                        "Always respond in the exact JSON format requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1000,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                OPENROUTER_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"OpenRouter API error {resp.status}: {body}")
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No response from OpenRouter API")
                return choices[0]["message"]["content"]

    async def analyze_with_chart(self, symbol: str, timeframe: str, df: pd.DataFrame,
                                  signal_direction: str, indicators: dict) -> dict:
        """
        Advanced AI analysis that generates a text-based chart representation
        and sends it to the vision model for pattern recognition.
        """
        if not self.enabled:
            return {
                "recommendation": "CONFIRM", "confidence": 50.0,
                "analysis": "Vision analysis disabled", "risk_notes": "None",
            }

        chart_description = self._build_chart_description(df)
        prompt = f"""You are an expert chart analyst. Analyze this {symbol} {timeframe} chart data and the proposed {signal_direction} trade.

CHART DATA (last 30 candles):
{chart_description}

INDICATORS:
- EMA Fast: {indicators.get('ema_fast', 'N/A')}
- EMA Slow: {indicators.get('ema_slow', 'N/A')}
- RSI: {indicators.get('rsi', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}

Look for:
1. Chart patterns (head & shoulders, double top/bottom, triangles, flags)
2. Support/resistance levels being tested
3. Candlestick patterns (engulfing, doji, hammer, shooting star)
4. Trend structure (higher highs/lows or lower highs/lows)
5. Divergences between price and RSI

RESPOND IN EXACTLY THIS JSON FORMAT:
{{
    "recommendation": "CONFIRM" or "REJECT" or "HOLD",
    "confidence": <number 0-100>,
    "analysis": "<analysis of chart patterns and structure>",
    "risk_notes": "<pattern-based risk warnings>",
    "patterns_detected": ["<list of chart patterns found>"]
}}"""

        try:
            response = await self._call_openrouter(prompt, model=self.vision_model)
            parsed = self._parse_ai_response(response)
            return parsed
        except Exception as e:
            logger.error("Vision analysis failed: %s", e)
            return {
                "recommendation": "CONFIRM", "confidence": 50.0,
                "analysis": f"Vision analysis error: {e}",
                "risk_notes": "Vision analysis unavailable",
            }

    def _build_chart_description(self, df: pd.DataFrame) -> str:
        """Build a text-based chart representation for AI analysis."""
        recent = df.tail(30)
        lines = []
        for _, row in recent.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            body = 'GREEN' if c >= o else 'RED'
            body_size = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            dt = row.get('datetime', '')
            lines.append(
                f"{dt} | O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} | "
                f"{body} body:{body_size:.2f} upper_wick:{upper_wick:.2f} lower_wick:{lower_wick:.2f}"
            )
        return "\n".join(lines)

    def _parse_ai_response(self, response_text: str) -> dict:
        """Parse the AI response into a structured dict."""
        # Try to extract JSON from the response
        text = response_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)

        try:
            result = json.loads(text)
            return {
                "recommendation": result.get("recommendation", "HOLD"),
                "confidence": float(result.get("confidence", 50)),
                "analysis": result.get("analysis", "No analysis provided"),
                "risk_notes": result.get("risk_notes", "None"),
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON: %s", text[:200])
            # Try to extract recommendation from text
            recommendation = "HOLD"
            if "CONFIRM" in text.upper():
                recommendation = "CONFIRM"
            elif "REJECT" in text.upper():
                recommendation = "REJECT"
            return {
                "recommendation": recommendation,
                "confidence": 50.0,
                "analysis": text[:500],
                "risk_notes": "Response parsing failed, using text extraction",
            }
