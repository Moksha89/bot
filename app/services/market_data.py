"""
Market data service.
Fetches and formats OHLCV data from Capital.com into pandas DataFrames.
"""

import logging
from typing import Optional

import pandas as pd

from app.api.capital_client import CapitalClient, CapitalAPIError

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    Fetches market data from Capital.com and converts it to DataFrames.
    """

    def __init__(self, client: CapitalClient) -> None:
        self.client = client

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "HOUR",
        count: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles and return as a DataFrame.
        Returns DataFrame with columns: datetime, open, high, low, close, volume
        """
        try:
            data = await self.client.get_prices(
                epic=symbol, resolution=timeframe, max_bars=count
            )
            prices = data.get("prices", [])
            if not prices:
                logger.warning("No candle data returned for %s", symbol)
                return None

            rows = []
            for candle in prices:
                row = {
                    "datetime": candle.get("snapshotTime", ""),
                    "open": self._mid(candle.get("openPrice", {})),
                    "high": self._mid(candle.get("highPrice", {})),
                    "low": self._mid(candle.get("lowPrice", {})),
                    "close": self._mid(candle.get("closePrice", {})),
                    "volume": candle.get("lastTradedVolume", 0),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info("Fetched %d candles for %s (%s)", len(df), symbol, timeframe)
            return df

        except CapitalAPIError as e:
            logger.error("Failed to fetch candles for %s: %s", symbol, e)
            return None

    async def get_spread(self, symbol: str) -> float:
        """Get current spread for a symbol."""
        try:
            price = await self.client.get_current_price(symbol)
            return price.get("spread", 0.0)
        except CapitalAPIError as e:
            logger.error("Failed to get spread for %s: %s", symbol, e)
            return 999.0  # Return high spread to block trading

    async def get_current_prices(
        self, symbols: list[str]
    ) -> dict[str, dict[str, float]]:
        """Get current bid/ask for multiple symbols."""
        prices: dict[str, dict[str, float]] = {}
        for symbol in symbols:
            try:
                price = await self.client.get_current_price(symbol)
                prices[symbol] = price
            except CapitalAPIError as e:
                logger.error("Failed to get price for %s: %s", symbol, e)
        return prices

    @staticmethod
    def _mid(price_obj: dict) -> float:
        """Calculate mid price from bid/ask."""
        bid = float(price_obj.get("bid", 0))
        ask = float(price_obj.get("ask", 0))
        if bid and ask:
            return (bid + ask) / 2
        return bid or ask
