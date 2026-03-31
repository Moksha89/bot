"""
Portfolio management module.
Manages correlations between symbols, total portfolio risk limits,
and multi-symbol position management.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Position

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Correlation data between symbols."""
    symbols: list[str]
    matrix: list[list[float]]
    timestamp: str


@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics."""
    total_exposure: float
    max_allowed_exposure: float
    exposure_pct: float
    correlated_risk: float
    positions_by_symbol: dict[str, int]
    is_within_limits: bool
    message: str


class PortfolioManager:
    """
    Manages portfolio-level risk across multiple symbols.
    Tracks correlations and enforces total portfolio risk limits.
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.10,  # 10% of account
        max_correlated_exposure: float = 0.15,  # 15% for correlated symbols
        max_positions: int = 5,
        correlation_threshold: float = 0.7,  # High correlation cutoff
    ) -> None:
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_exposure = max_correlated_exposure
        self.max_positions = max_positions
        self.correlation_threshold = correlation_threshold
        self._correlation_cache: Optional[CorrelationMatrix] = None
        self._price_history: dict[str, pd.Series] = {}

    def update_price_history(self, symbol: str, closes: pd.Series) -> None:
        """Update price history for correlation calculations."""
        self._price_history[symbol] = closes.copy()

    def calculate_correlations(self) -> Optional[CorrelationMatrix]:
        """Calculate correlation matrix between all tracked symbols."""
        if len(self._price_history) < 2:
            return None

        symbols = list(self._price_history.keys())
        # Align all series to common dates
        df = pd.DataFrame(self._price_history)
        returns = df.pct_change().dropna()

        if len(returns) < 20:
            return None

        corr = returns.corr()
        matrix = corr.values.tolist()

        result = CorrelationMatrix(
            symbols=symbols,
            matrix=matrix,
            timestamp=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        self._correlation_cache = result
        return result

    def get_correlated_symbols(self, symbol: str) -> list[tuple[str, float]]:
        """Get symbols that are highly correlated with the given symbol."""
        if self._correlation_cache is None:
            self.calculate_correlations()

        if self._correlation_cache is None:
            return []

        try:
            idx = self._correlation_cache.symbols.index(symbol)
        except ValueError:
            return []

        correlated = []
        for i, s in enumerate(self._correlation_cache.symbols):
            if i == idx:
                continue
            corr_val = self._correlation_cache.matrix[idx][i]
            if abs(corr_val) >= self.correlation_threshold:
                correlated.append((s, corr_val))

        return sorted(correlated, key=lambda x: abs(x[1]), reverse=True)

    async def check_portfolio_risk(
        self,
        session: AsyncSession,
        symbol: str,
        direction: str,
        account_balance: float,
        position_risk: float,
    ) -> PortfolioRisk:
        """
        Check if a new trade would exceed portfolio risk limits.
        position_risk: the dollar risk of the proposed new position.
        """
        # Get all open positions
        result = await session.execute(
            select(Position).where(Position.is_open.is_(True))
        )
        open_positions = result.scalars().all()

        # Calculate current exposure
        total_exposure = sum(
            abs(p.entry_price * p.size) for p in open_positions
            if p.entry_price and p.size
        )
        current_risk = sum(
            abs(p.entry_price - (p.stop_loss or p.entry_price)) * p.size
            for p in open_positions
            if p.entry_price and p.size
        )

        # Count positions by symbol
        positions_by_symbol: dict[str, int] = {}
        for p in open_positions:
            positions_by_symbol[p.symbol] = positions_by_symbol.get(p.symbol, 0) + 1

        # Check total positions
        total_positions = len(open_positions)
        if total_positions >= self.max_positions:
            return PortfolioRisk(
                total_exposure=total_exposure,
                max_allowed_exposure=account_balance * self.max_portfolio_risk,
                exposure_pct=current_risk / account_balance if account_balance > 0 else 0,
                correlated_risk=0,
                positions_by_symbol=positions_by_symbol,
                is_within_limits=False,
                message=f"Max positions reached ({total_positions}/{self.max_positions})",
            )

        # Check total portfolio risk
        new_total_risk = current_risk + position_risk
        risk_pct = new_total_risk / account_balance if account_balance > 0 else 0

        if risk_pct > self.max_portfolio_risk:
            return PortfolioRisk(
                total_exposure=total_exposure,
                max_allowed_exposure=account_balance * self.max_portfolio_risk,
                exposure_pct=risk_pct,
                correlated_risk=0,
                positions_by_symbol=positions_by_symbol,
                is_within_limits=False,
                message=f"Portfolio risk {risk_pct:.1%} exceeds limit {self.max_portfolio_risk:.1%}",
            )

        # Check correlated exposure
        correlated_risk = self._calculate_correlated_risk(
            symbol, direction, position_risk, open_positions,
        )

        corr_pct = correlated_risk / account_balance if account_balance > 0 else 0
        if corr_pct > self.max_correlated_exposure:
            return PortfolioRisk(
                total_exposure=total_exposure,
                max_allowed_exposure=account_balance * self.max_portfolio_risk,
                exposure_pct=risk_pct,
                correlated_risk=correlated_risk,
                positions_by_symbol=positions_by_symbol,
                is_within_limits=False,
                message=(
                    f"Correlated exposure {corr_pct:.1%} exceeds limit "
                    f"{self.max_correlated_exposure:.1%}"
                ),
            )

        return PortfolioRisk(
            total_exposure=total_exposure + position_risk,
            max_allowed_exposure=account_balance * self.max_portfolio_risk,
            exposure_pct=risk_pct,
            correlated_risk=correlated_risk,
            positions_by_symbol=positions_by_symbol,
            is_within_limits=True,
            message="Within portfolio risk limits",
        )

    def _calculate_correlated_risk(
        self,
        symbol: str,
        direction: str,
        position_risk: float,
        open_positions: list[Position],
    ) -> float:
        """Calculate total risk from correlated positions."""
        correlated_symbols = self.get_correlated_symbols(symbol)
        correlated_symbol_set = {s for s, _ in correlated_symbols}

        total_corr_risk = position_risk  # Include proposed trade
        for pos in open_positions:
            if pos.symbol in correlated_symbol_set or pos.symbol == symbol:
                risk = abs(pos.entry_price - (pos.stop_loss or pos.entry_price)) * pos.size
                total_corr_risk += risk

        return total_corr_risk

    def get_portfolio_summary(
        self, positions: list[Position], account_balance: float,
    ) -> dict:
        """Get a summary of the current portfolio state."""
        total_exposure = sum(
            abs(p.entry_price * p.size) for p in positions
            if p.entry_price and p.size
        )
        total_risk = sum(
            abs(p.entry_price - (p.stop_loss or p.entry_price)) * p.size
            for p in positions
            if p.entry_price and p.size
        )
        total_unrealized_pnl = 0.0

        positions_by_symbol: dict[str, list[dict]] = {}
        for p in positions:
            if p.symbol not in positions_by_symbol:
                positions_by_symbol[p.symbol] = []
            positions_by_symbol[p.symbol].append({
                "direction": p.direction,
                "size": p.size,
                "entry": p.entry_price,
                "sl": p.stop_loss,
                "tp": p.take_profit,
            })

        return {
            "total_positions": len(positions),
            "max_positions": self.max_positions,
            "total_exposure": round(total_exposure, 2),
            "total_risk": round(total_risk, 2),
            "risk_pct": round(total_risk / account_balance * 100, 2) if account_balance > 0 else 0,
            "max_risk_pct": self.max_portfolio_risk * 100,
            "positions_by_symbol": positions_by_symbol,
            "correlations": {
                "symbols": self._correlation_cache.symbols if self._correlation_cache else [],
                "matrix": self._correlation_cache.matrix if self._correlation_cache else [],
            },
        }
