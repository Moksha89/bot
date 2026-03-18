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
import pandas as pd

from app.config import settings

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
        """Prepare a concise market data summary for the AI."""
        recent = df.tail(20)

        # Price action summary
        candles = []
        for _, row in recent.iterrows():
            candles.append({
                "open": round(float(row["open"]), 5),
                "high": round(float(row["high"]), 5),
                "low": round(float(row["low"]), 5),
                "close": round(float(row["close"]), 5),
            })

        # Calculate key levels
        last_20_high = float(recent["high"].max())
        last_20_low = float(recent["low"].min())
        current_close = float(df.iloc[-1]["close"])
        prev_close = float(df.iloc[-2]["close"])

        # Volume analysis if available
        volume_data = None
        if "volume" in df.columns:
            avg_volume = float(recent["volume"].mean())
            latest_volume = float(df.iloc[-1]["volume"])
            volume_data = {
                "latest": latest_volume,
                "average_20": round(avg_volume, 0),
                "ratio": round(latest_volume / avg_volume, 2) if avg_volume > 0 else 0,
            }

        # Price change stats
        price_change_1 = round(
            ((current_close - prev_close) / prev_close) * 100, 4
        )
        price_change_5 = round(
            ((current_close - float(df.iloc[-6]["close"])) / float(df.iloc[-6]["close"])) * 100, 4
        ) if len(df) >= 6 else 0

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal_direction": signal_direction,
            "current_price": current_close,
            "spread": spread,
            "account_balance": account_balance,
            "recent_candles_count": len(candles),
            "last_5_candles": candles[-5:],
            "indicators": indicators,
            "support_resistance": {
                "recent_high_20": round(last_20_high, 5),
                "recent_low_20": round(last_20_low, 5),
                "range": round(last_20_high - last_20_low, 5),
            },
            "price_change": {
                "1_candle_pct": price_change_1,
                "5_candle_pct": price_change_5,
            },
            "volume": volume_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_analysis_prompt(self, market_summary: dict) -> str:
        """Build the analysis prompt for the AI."""
        return f"""You are an expert trading analyst. Analyze the following market data and provide your assessment.

MARKET DATA:
{json.dumps(market_summary, indent=2)}

ANALYSIS REQUIREMENTS:
1. Evaluate the current market conditions based on the provided data
2. Assess whether the {market_summary['signal_direction']} signal aligns with the overall market structure
3. Check for potential risks: overextension, divergences, key levels, spread concerns
4. Consider volume confirmation if available
5. Evaluate trend strength from the indicator values
6. Look for any red flags that would warrant rejecting this trade

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{{
    "recommendation": "CONFIRM" or "REJECT" or "HOLD",
    "confidence": <number 0-100>,
    "analysis": "<2-3 sentence analysis of market conditions and why you recommend this action>",
    "risk_notes": "<any specific risk warnings or concerns, or 'None' if no concerns>"
}}

Rules:
- CONFIRM: The signal looks good, proceed with the trade
- REJECT: Market conditions don't support this trade, skip it
- HOLD: Wait for better conditions or more confirmation
- Be conservative - when in doubt, recommend HOLD or REJECT
- Consider the spread relative to expected move
- Factor in trend alignment with the signal direction"""

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
                        "You are a professional trading analyst AI. "
                        "You analyze market data and provide precise, "
                        "actionable trading recommendations. "
                        "Always respond in the exact JSON format requested. "
                        "Be conservative and prioritize capital preservation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
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
