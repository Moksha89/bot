"""
News filter module.
Skips trading during high-impact news events (NFP, FOMC, CPI, etc.).
Uses a configurable calendar of known events and optionally fetches
from external APIs.
"""

import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """A scheduled high-impact news event."""
    name: str
    timestamp: datetime
    currency: str  # e.g., "USD", "EUR", "ALL"
    impact: str  # "high", "medium", "low"
    buffer_minutes_before: int = 30
    buffer_minutes_after: int = 30


# Known recurring high-impact events (approximate schedule)
KNOWN_HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "FOMC Rate Decision",
    "FOMC Minutes",
    "CPI",
    "Core CPI",
    "GDP",
    "Retail Sales",
    "PMI",
    "ECB Rate Decision",
    "BOE Rate Decision",
    "BOJ Rate Decision",
    "Unemployment Rate",
    "ISM Manufacturing",
    "ISM Services",
    "Trade Balance",
    "Consumer Confidence",
    "Durable Goods Orders",
    "Jackson Hole",
]


class NewsFilter:
    """
    Filters trades during high-impact news events.
    Uses a combination of:
    1. Manually configured events
    2. External API for economic calendar (optional)
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_before: int = 30,
        buffer_after: int = 30,
        api_url: str = "",
    ) -> None:
        self.enabled = enabled
        self.buffer_before = buffer_before
        self.buffer_after = buffer_after
        self.api_url = api_url
        self._events: list[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(hours=6)

    def add_event(self, event: NewsEvent) -> None:
        """Manually add a news event."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

    def add_events_from_list(self, events: list[dict]) -> None:
        """Add events from a list of dicts."""
        for e in events:
            try:
                ts = e.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                self._events.append(NewsEvent(
                    name=e.get("name", "Unknown"),
                    timestamp=ts,
                    currency=e.get("currency", "ALL"),
                    impact=e.get("impact", "high"),
                    buffer_minutes_before=e.get("buffer_before", self.buffer_before),
                    buffer_minutes_after=e.get("buffer_after", self.buffer_after),
                ))
            except (ValueError, TypeError) as ex:
                logger.warning("Failed to parse news event: %s (%s)", e, ex)

    async def fetch_events(self) -> None:
        """Fetch events from external economic calendar API (if configured)."""
        if not self.api_url:
            return

        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return  # Don't fetch too frequently

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = data if isinstance(data, list) else data.get("events", [])
                        # Only add high-impact events
                        high_impact = [
                            e for e in events
                            if e.get("impact", "").lower() in ("high", "critical")
                        ]
                        self.add_events_from_list(high_impact)
                        self._last_fetch = now
                        logger.info("Fetched %d high-impact events from API", len(high_impact))
                    else:
                        logger.warning("News API returned status %d", resp.status)
        except Exception as e:
            logger.error("Failed to fetch news events: %s", e)

    def is_trading_allowed(
        self,
        symbol: str = "",
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """
        Check if trading is allowed based on upcoming/active news events.
        Returns (allowed, reason).
        """
        if not self.enabled:
            return True, "News filter disabled"

        if now is None:
            now = datetime.now(timezone.utc)

        # Clean up old events
        self._events = [
            e for e in self._events
            if e.timestamp + timedelta(minutes=e.buffer_minutes_after) > now
        ]

        # Determine relevant currency from symbol
        relevant_currencies = self._get_currencies_for_symbol(symbol)

        for event in self._events:
            # Check if event is relevant to this symbol
            if event.currency != "ALL" and event.currency not in relevant_currencies:
                continue

            event_start = event.timestamp - timedelta(minutes=event.buffer_minutes_before)
            event_end = event.timestamp + timedelta(minutes=event.buffer_minutes_after)

            if event_start <= now <= event_end:
                remaining = (event_end - now).total_seconds() / 60
                return False, (
                    f"High-impact news: {event.name} ({event.impact}) "
                    f"at {event.timestamp.strftime('%H:%M UTC')}. "
                    f"Trading resumes in {remaining:.0f} min."
                )

        return True, "No active news events"

    def get_upcoming_events(
        self, hours: int = 24, now: datetime | None = None,
    ) -> list[NewsEvent]:
        """Get upcoming events within the next N hours."""
        if now is None:
            now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        return [e for e in self._events if now <= e.timestamp <= cutoff]

    @staticmethod
    def _get_currencies_for_symbol(symbol: str) -> set[str]:
        """Extract relevant currencies from a trading symbol."""
        symbol = symbol.upper().replace("/", "")
        currencies = set()
        # Common currency pairs
        known = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]
        for curr in known:
            if curr in symbol:
                currencies.add(curr)
        # Commodities
        if "XAU" in symbol or "GOLD" in symbol:
            currencies.add("USD")
            currencies.add("XAU")
        if "XAG" in symbol or "SILVER" in symbol:
            currencies.add("USD")
        if "OIL" in symbol or "WTI" in symbol or "BRENT" in symbol:
            currencies.add("USD")
        # Indices
        if any(idx in symbol for idx in ["US100", "US500", "US30", "SPX", "NDX", "DJI"]):
            currencies.add("USD")
        if "UK100" in symbol or "FTSE" in symbol:
            currencies.add("GBP")
        if "DE30" in symbol or "DAX" in symbol:
            currencies.add("EUR")
        return currencies or {"ALL"}
