"""
Sentiment analysis module.
Pulls news/social sentiment as an additional trading filter.
Uses external APIs and OpenRouter LLM for sentiment scoring.
"""

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    score: float  # -1.0 (very bearish) to 1.0 (very bullish)
    label: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    sources: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources:
    1. News headlines via RSS/API
    2. LLM-based sentiment scoring via OpenRouter
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[str, SentimentResult] = {}
        self._cache_ttl = timedelta(minutes=30)

    async def analyze(self, symbol: str) -> SentimentResult:
        """
        Get sentiment for a symbol.
        Returns cached result if fresh enough.
        """
        if not self.enabled:
            return SentimentResult(
                score=0.0, label="neutral", confidence=0.0,
                sources=["disabled"],
            )

        # Check cache
        cached = self._cache.get(symbol)
        if cached and (datetime.now(timezone.utc) - cached.timestamp) < self._cache_ttl:
            return cached

        # Fetch headlines
        headlines = await self._fetch_headlines(symbol)

        if not headlines:
            result = SentimentResult(
                score=0.0, label="neutral", confidence=0.0,
                sources=["no_data"],
            )
            self._cache[symbol] = result
            return result

        # Score with LLM
        result = await self._score_with_llm(symbol, headlines)
        self._cache[symbol] = result
        return result

    async def _fetch_headlines(self, symbol: str) -> list[str]:
        """Fetch recent news headlines for a symbol."""
        headlines: list[str] = []

        # Map symbol to search terms
        search_terms = self._symbol_to_search_terms(symbol)

        # Try fetching from free news APIs
        try:
            async with aiohttp.ClientSession() as session:
                # Use DuckDuckGo news as a free source
                for term in search_terms[:2]:
                    url = f"https://api.duckduckgo.com/?q={term}+market+news&format=json&no_redirect=1"
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            for topic in data.get("RelatedTopics", [])[:5]:
                                text = topic.get("Text", "")
                                if text:
                                    headlines.append(text)
        except Exception as e:
            logger.warning("Failed to fetch headlines: %s", e)

        return headlines[:10]

    async def _score_with_llm(
        self, symbol: str, headlines: list[str],
    ) -> SentimentResult:
        """Use OpenRouter LLM to score sentiment from headlines."""
        api_key = settings.openrouter_api_key
        if not api_key:
            return self._simple_keyword_score(headlines)

        prompt = (
            f"Analyze the following news headlines for {symbol} market sentiment.\n\n"
            f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float from -1.0 bearish to 1.0 bullish>, '
            '"label": "<bullish|bearish|neutral>", '
            '"confidence": <float 0.0-1.0>, '
            '"summary": "<one sentence summary>"}'
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.ai_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return self._parse_sentiment_response(content, headlines)
        except Exception as e:
            logger.warning("LLM sentiment scoring failed: %s", e)

        return self._simple_keyword_score(headlines)

    def _parse_sentiment_response(
        self, content: str, headlines: list[str],
    ) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        import json
        try:
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                score = max(-1.0, min(1.0, float(data.get("score", 0))))
                return SentimentResult(
                    score=score,
                    label=data.get("label", "neutral"),
                    confidence=max(0, min(1.0, float(data.get("confidence", 0.5)))),
                    sources=["llm"],
                    headlines=headlines,
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse sentiment response: %s", e)

        return self._simple_keyword_score(headlines)

    @staticmethod
    def _simple_keyword_score(headlines: list[str]) -> SentimentResult:
        """Fallback: simple keyword-based sentiment scoring."""
        bullish_words = {"rally", "surge", "bullish", "gain", "rise", "high", "up",
                         "growth", "positive", "strong", "recover", "boom", "buy"}
        bearish_words = {"crash", "drop", "bearish", "fall", "decline", "low", "down",
                         "recession", "negative", "weak", "sell", "plunge", "fear"}

        bull_count = 0
        bear_count = 0
        text = " ".join(headlines).lower()
        for word in bullish_words:
            bull_count += text.count(word)
        for word in bearish_words:
            bear_count += text.count(word)

        total = bull_count + bear_count
        if total == 0:
            return SentimentResult(
                score=0.0, label="neutral", confidence=0.3,
                sources=["keyword"], headlines=headlines,
            )

        score = (bull_count - bear_count) / total
        label = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")
        return SentimentResult(
            score=round(score, 3),
            label=label,
            confidence=min(total / 20, 0.7),
            sources=["keyword"],
            headlines=headlines,
        )

    @staticmethod
    def _symbol_to_search_terms(symbol: str) -> list[str]:
        """Convert a trading symbol to search-friendly terms."""
        symbol = symbol.upper()
        mappings = {
            "XAUUSD": ["gold", "XAUUSD"],
            "EURUSD": ["EUR/USD", "euro dollar"],
            "GBPUSD": ["GBP/USD", "pound dollar"],
            "USDJPY": ["USD/JPY", "dollar yen"],
            "BTCUSD": ["bitcoin", "BTC"],
            "US100": ["NASDAQ", "US100"],
            "US500": ["S&P 500", "SPX"],
            "US30": ["Dow Jones", "DJIA"],
        }
        return mappings.get(symbol, [symbol])

    def is_sentiment_favorable(
        self, result: SentimentResult, direction: str,
        min_score: float = 0.1, min_confidence: float = 0.3,
    ) -> tuple[bool, str]:
        """Check if sentiment supports the trade direction."""
        if result.confidence < min_confidence:
            return True, f"Sentiment confidence too low ({result.confidence:.2f}), allowing trade"

        if direction == "BUY" and result.score < -min_score:
            return False, f"Bearish sentiment ({result.score:.3f}) conflicts with BUY"
        elif direction == "SELL" and result.score > min_score:
            return False, f"Bullish sentiment ({result.score:.3f}) conflicts with SELL"

        return True, f"Sentiment {result.label} ({result.score:.3f}) supports {direction}"
