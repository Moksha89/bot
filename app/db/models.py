"""
Database models for storing signals, orders, positions, trade results, errors, and balance history.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase
import enum


class Base(DeclarativeBase):
    pass


class SignalDirection(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class OrderStatus(str, enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TradeResult(str, enum.Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(20), nullable=False)
    direction = Column(SAEnum(SignalDirection), nullable=False)
    ema_fast = Column(Float)
    ema_slow = Column(Float)
    rsi = Column(Float)
    close_price = Column(Float)
    prev_high = Column(Float)
    prev_low = Column(Float)
    spread = Column(Float)
    reason = Column(Text)
    acted_on = Column(Boolean, default=False)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    signal_id = Column(Integer, nullable=True)
    symbol = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)
    size = Column(Float, nullable=False)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    status = Column(SAEnum(OrderStatus), default=OrderStatus.PENDING)
    deal_id = Column(String(100), nullable=True)
    deal_reference = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opened_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    closed_at = Column(DateTime, nullable=True)
    symbol = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float, nullable=True)
    result = Column(SAEnum(TradeResult), default=TradeResult.OPEN)
    deal_id = Column(String(100), nullable=True)
    is_open = Column(Boolean, default=True)


class ErrorLog(Base):
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    module = Column(String(100), nullable=False)
    error_type = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)


class BalanceHistory(Base):
    __tablename__ = "balance_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    available = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)


class DailyPnL(Base):
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True)
    total_pnl = Column(Float, default=0.0)
    trade_count = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    starting_balance = Column(Float)
    ending_balance = Column(Float)
