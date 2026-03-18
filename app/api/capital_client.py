"""
Capital.com REST API client.
Handles authentication, market data, order placement, and position management.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Capital.com API endpoints
AUTH_ENDPOINT = "/api/v1/session"
ACCOUNTS_ENDPOINT = "/api/v1/accounts"
MARKETS_ENDPOINT = "/api/v1/markets"
PRICES_ENDPOINT = "/api/v1/prices"
POSITIONS_ENDPOINT = "/api/v1/positions"
ORDERS_ENDPOINT = "/api/v1/orders"


class CapitalAPIError(Exception):
    """Custom exception for Capital.com API errors."""

    def __init__(self, message: str, status_code: int = 0, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class CapitalClient:
    """
    Async client for Capital.com API.
    Manages authentication tokens and provides methods for all trading operations.
    """

    def __init__(self) -> None:
        self.base_url = settings.capital.api_url
        self.api_key = settings.capital.api_key
        self.email = settings.capital.email
        self.password = settings.capital.password
        self.cst: Optional[str] = None
        self.security_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"X-CAP-API-KEY": self.api_key}
        if self.cst:
            headers["CST"] = self.cst
        if self.security_token:
            headers["X-SECURITY-TOKEN"] = self.security_token
        return headers

    async def authenticate(self) -> bool:
        """
        Authenticate with Capital.com and obtain session tokens.
        Returns True on success.
        """
        client = await self._get_client()
        payload = {
            "identifier": self.email,
            "password": self.password,
        }
        try:
            response = await client.post(
                AUTH_ENDPOINT,
                json=payload,
                headers={"X-CAP-API-KEY": self.api_key},
            )
            if response.status_code == 200:
                self.cst = response.headers.get("CST", "")
                self.security_token = response.headers.get("X-SECURITY-TOKEN", "")
                data = response.json()
                self.account_id = data.get("currentAccountId")
                logger.info("Authenticated with Capital.com (account: %s)", self.account_id)
                return True
            else:
                logger.error(
                    "Authentication failed: %s %s",
                    response.status_code,
                    response.text,
                )
                try:
                    auth_error_data = response.json() if response.text else {}
                except Exception:
                    auth_error_data = {}
                raise CapitalAPIError(
                    f"Auth failed: {response.status_code}",
                    status_code=response.status_code,
                    response_data=auth_error_data,
                )
        except httpx.RequestError as exc:
            logger.error("Authentication request error: %s", exc)
            raise CapitalAPIError(f"Connection error: {exc}") from exc

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        retry: bool = True,
    ) -> dict:
        """Make an authenticated API request with optional retry on auth failure."""
        client = await self._get_client()
        try:
            response = await client.request(
                method,
                endpoint,
                headers=self._auth_headers(),
                params=params,
                json=json_data,
            )
            if response.status_code == 401 and retry:
                logger.warning("Session expired, re-authenticating...")
                await self.authenticate()
                return await self._request(method, endpoint, params, json_data, retry=False)

            if response.status_code >= 400:
                try:
                    error_data = response.json() if response.text else {}
                except Exception:
                    error_data = {}
                raise CapitalAPIError(
                    f"API error {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response_data=error_data,
                )
            try:
                return response.json() if response.text else {}
            except Exception:
                return {}
        except httpx.RequestError as exc:
            logger.error("Request error on %s %s: %s", method, endpoint, exc)
            raise CapitalAPIError(f"Request error: {exc}") from exc

    # --- Account ---

    async def get_accounts(self) -> dict:
        """Get account details including balance."""
        return await self._request("GET", ACCOUNTS_ENDPOINT)

    async def get_account_balance(self) -> dict[str, float]:
        """Get current account balance, equity, and available funds."""
        data = await self.get_accounts()
        accounts = data.get("accounts", [])
        if not accounts:
            return {"balance": 0.0, "equity": 0.0, "available": 0.0, "pnl": 0.0}
        account = accounts[0]
        balance_info = account.get("balance", {})
        return {
            "balance": float(balance_info.get("balance", 0)),
            "equity": float(balance_info.get("deposit", 0)),
            "available": float(balance_info.get("available", 0)),
            "pnl": float(balance_info.get("profitLoss", 0)),
        }

    # --- Market Data ---

    async def get_market_info(self, epic: str) -> dict:
        """Get market details for a specific symbol."""
        return await self._request("GET", f"{MARKETS_ENDPOINT}/{epic}")

    async def get_prices(
        self,
        epic: str,
        resolution: str = "HOUR",
        max_bars: int = 100,
    ) -> dict:
        """
        Get historical price candles.
        resolution: MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK
        """
        params = {"resolution": resolution, "max": max_bars}
        return await self._request("GET", f"{PRICES_ENDPOINT}/{epic}", params=params)

    async def get_current_price(self, epic: str) -> dict[str, float]:
        """Get current bid/ask/spread for a symbol."""
        data = await self.get_prices(epic, max_bars=1)
        prices = data.get("prices", [])
        if not prices:
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0}
        latest = prices[-1]
        bid = float(latest.get("closePrice", {}).get("bid", 0))
        ask = float(latest.get("closePrice", {}).get("ask", 0))
        return {"bid": bid, "ask": ask, "spread": round(ask - bid, 5)}

    # --- Positions ---

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = await self._request("GET", POSITIONS_ENDPOINT)
        return data.get("positions", [])

    async def get_position_by_symbol(self, epic: str) -> Optional[dict]:
        """Get an open position for a specific symbol, if any."""
        positions = await self.get_positions()
        for pos in positions:
            market = pos.get("market", {})
            if market.get("epic") == epic:
                return pos
        return None

    async def close_position(self, deal_id: str) -> dict:
        """Close a position by deal ID."""
        return await self._request("DELETE", f"{POSITIONS_ENDPOINT}/{deal_id}")

    async def update_position(
        self,
        deal_id: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict:
        """Update stop loss and/or take profit for an existing position."""
        payload: dict = {}
        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
        if take_profit is not None:
            payload["profitLevel"] = take_profit
        if not payload:
            return {}
        return await self._request("PUT", f"{POSITIONS_ENDPOINT}/{deal_id}", json_data=payload)

    # --- Transaction History ---

    async def get_transaction_history(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        transaction_type: str = "ALL",
        max_results: int = 500,
    ) -> list[dict]:
        """
        Get account transaction history (closed trades, deposits, withdrawals).
        from_date/to_date format: 2024-01-01T00:00:00
        transaction_type: ALL, TRADE, DEPOSIT, WITHDRAWAL
        """
        params: dict = {"type": transaction_type}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._request("GET", "/api/v1/history/transactions", params=params)
        return data.get("transactions", [])

    async def get_activity_history(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        max_results: int = 500,
    ) -> list[dict]:
        """
        Get account activity history (trades, order fills, etc.).
        from_date/to_date format: 2024-01-01T00:00:00
        """
        params: dict = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._request("GET", "/api/v1/history/activity", params=params)
        return data.get("activities", [])

    async def get_market_constraints(self, epic: str) -> dict:
        """Get market constraints for order validation.

        Returns dict with keys:
            bid, ask, min_stop_pct, min_deal_size, min_size_increment
        """
        info = await self.get_market_info(epic)
        rules = info.get("dealingRules", {})
        snapshot = info.get("snapshot", {})

        min_stop = rules.get("minStopOrProfitDistance", {})
        min_stop_pct = float(min_stop.get("value", 0.01)) if min_stop.get("unit") == "PERCENTAGE" else 0.0
        # If unit is POINTS, store raw value as points distance
        min_stop_points = float(min_stop.get("value", 0)) if min_stop.get("unit") == "POINTS" else 0.0

        min_deal = rules.get("minDealSize", {})
        min_deal_size = float(min_deal.get("value", 0.01))

        min_inc = rules.get("minSizeIncrement", {})
        min_size_increment = float(min_inc.get("value", 0.01))

        max_deal = rules.get("maxDealSize", {})
        max_deal_size = float(max_deal.get("value", 0))

        return {
            "bid": float(snapshot.get("bid", 0)),
            "ask": float(snapshot.get("offer", 0)),
            "min_stop_pct": min_stop_pct,
            "min_stop_points": min_stop_points,
            "min_deal_size": min_deal_size,
            "min_size_increment": min_size_increment,
            "max_deal_size": max_deal_size,
        }

    # --- Orders ---

    async def place_order(
        self,
        epic: str,
        direction: str,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        """
        Place a market order.
        direction: 'BUY' or 'SELL'
        """
        payload: dict = {
            "epic": epic,
            "direction": direction,
            "size": size,
        }
        if stop_loss is not None:
            payload["stopLevel"] = stop_loss
        if take_profit is not None:
            payload["profitLevel"] = take_profit

        logger.info(
            "Placing %s order: %s size=%.4f SL=%s TP=%s",
            direction,
            epic,
            size,
            stop_loss,
            take_profit,
        )
        logger.info("Order payload: %s", payload)
        result = await self._request("POST", POSITIONS_ENDPOINT, json_data=payload)
        logger.info("Order result: %s", result)
        return result

    async def confirm_order(self, deal_reference: str) -> dict:
        """Confirm an order by deal reference."""
        return await self._request("GET", f"/api/v1/confirms/{deal_reference}")

    # --- Session management ---

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("Capital.com client closed")

    async def ping(self) -> bool:
        """Check if API connection is alive."""
        try:
            await self.get_accounts()
            return True
        except CapitalAPIError:
            return False
