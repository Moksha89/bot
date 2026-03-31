"""
Configuration module for the trading bot.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class CapitalConfig:
    api_key: str = field(default_factory=lambda: _env("CAPITAL_API_KEY"))
    email: str = field(default_factory=lambda: _env("CAPITAL_EMAIL"))
    password: str = field(default_factory=lambda: _env("CAPITAL_PASSWORD"))
    api_url: str = field(
        default_factory=lambda: _env(
            "CAPITAL_API_URL",
            "https://demo-api-capital.backend-capital.com",
        )
    )


@dataclass
class TradingConfig:
    symbol: str = field(default_factory=lambda: _env("TRADING_SYMBOL", "XAUUSD"))
    # Symbols ranked by backtest profit factor (2026-03-19).
    # Dropped worst forex pairs (EURUSD PF=0.01, GBPUSD PF=0.01, USDCAD PF=0.00,
    # USDJPY PF=0.02, AUDUSD PF=0.05) and worst equities/indices
    # (MSFT PF=0.09, US500 PF=0.09, DE40 PF=0.07).
    symbols: list[str] = field(default_factory=lambda: [
        s.strip() for s in _env(
            "TRADING_SYMBOLS",
            # Top tier (PF > 0.7): OIL_CRUDE, DOGEUSD, AMZN, ADAUSD, DOTUSD, XRPUSD
            # Mid tier (PF 0.3-0.7): TSLA, SILVER, BTCUSD, ETHUSD, GOLD
            # Lower tier kept for diversity: US100, UK100, AAPL, NVDA, GOOGL, META
            "OIL_CRUDE,DOGEUSD,AMZN,ADAUSD,DOTUSD,XRPUSD,"
            "TSLA,SILVER,BTCUSD,ETHUSD,GOLD,"
            "US100,UK100,AAPL,NVDA,GOOGL,META"
        ).split(",") if s.strip()
    ])
    timeframe: str = field(default_factory=lambda: _env("TRADING_TIMEFRAME", "HOUR"))
    mode: str = field(default_factory=lambda: _env("TRADING_MODE", "demo"))
    strategy: str = field(default_factory=lambda: _env("TRADING_STRATEGY", "auto"))


@dataclass
class StrategyConfig:
    ema_fast: int = field(default_factory=lambda: _env_int("EMA_FAST", 20))
    ema_slow: int = field(default_factory=lambda: _env_int("EMA_SLOW", 50))
    rsi_period: int = field(default_factory=lambda: _env_int("RSI_PERIOD", 14))
    rsi_buy_min: float = field(default_factory=lambda: _env_float("RSI_BUY_MIN", 40))
    rsi_buy_max: float = field(default_factory=lambda: _env_float("RSI_BUY_MAX", 80))
    rsi_sell_min: float = field(default_factory=lambda: _env_float("RSI_SELL_MIN", 20))
    rsi_sell_max: float = field(default_factory=lambda: _env_float("RSI_SELL_MAX", 60))
    max_spread: float = field(default_factory=lambda: _env_float("MAX_SPREAD", 5.0))
    require_breakout: bool = field(default_factory=lambda: _env_bool("REQUIRE_BREAKOUT", False))


@dataclass
class RiskConfig:
    risk_per_trade: float = field(
        default_factory=lambda: _env_float("RISK_PER_TRADE", 0.08)
    )
    max_daily_loss: float = field(
        default_factory=lambda: _env_float("MAX_DAILY_LOSS", 0.15)
    )
    max_consecutive_losses: int = field(
        default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 5)
    )
    cooldown_minutes: int = field(
        default_factory=lambda: _env_int("COOLDOWN_MINUTES", 5)
    )
    max_open_trades: int = field(
        default_factory=lambda: _env_int("MAX_OPEN_TRADES", 999)
    )
    sl_atr_multiplier: float = field(
        default_factory=lambda: _env_float("SL_ATR_MULTIPLIER", 2.5)
    )
    tp_risk_reward: float = field(
        default_factory=lambda: _env_float("TP_RISK_REWARD", 3.5)
    )
    max_portfolio_risk: float = field(
        default_factory=lambda: _env_float("MAX_PORTFOLIO_RISK", 0.30)
    )
    max_correlated_exposure: float = field(
        default_factory=lambda: _env_float("MAX_CORRELATED_EXPOSURE", 0.25)
    )
    max_total_positions: int = field(
        default_factory=lambda: _env_int("MAX_TOTAL_POSITIONS", 999)
    )


@dataclass
class DatabaseConfig:
    url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "sqlite+aiosqlite:///./data/trading_bot.db"
        )
    )


@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


@dataclass
class Settings:
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    kill_switch: bool = field(default_factory=lambda: _env_bool("KILL_SWITCH", False))
    openrouter_api_key: str = field(
        default_factory=lambda: _env("OPENROUTER_API_KEY")
    )
    ai_model: str = field(
        default_factory=lambda: _env("AI_MODEL", "google/gemini-2.0-flash-001")
    )
    ai_analysis_enabled: bool = field(
        default_factory=lambda: _env_bool("AI_ANALYSIS_ENABLED", True)
    )
    ai_vision_model: str = field(
        default_factory=lambda: _env("AI_VISION_MODEL", "google/gemini-2.0-flash-001")
    )
    # Feature flags
    trailing_stop_enabled: bool = field(
        default_factory=lambda: _env_bool("TRAILING_STOP_ENABLED", True)
    )
    trailing_stop_activation_r: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_ACTIVATION_R", 1.0)
    )
    trailing_stop_atr_mult: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_ATR_MULT", 1.0)
    )
    session_filter_enabled: bool = field(
        default_factory=lambda: _env_bool("SESSION_FILTER_ENABLED", False)
    )
    allowed_sessions: str = field(
        default_factory=lambda: _env("ALLOWED_SESSIONS", "london,new_york")
    )
    news_filter_enabled: bool = field(
        default_factory=lambda: _env_bool("NEWS_FILTER_ENABLED", True)
    )
    news_buffer_minutes: int = field(
        default_factory=lambda: _env_int("NEWS_BUFFER_MINUTES", 30)
    )
    sentiment_enabled: bool = field(
        default_factory=lambda: _env_bool("SENTIMENT_ENABLED", True)
    )
    ml_scoring_enabled: bool = field(
        default_factory=lambda: _env_bool("ML_SCORING_ENABLED", True)
    )
    auto_optimize_enabled: bool = field(
        default_factory=lambda: _env_bool("AUTO_OPTIMIZE_ENABLED", True)
    )
    auto_optimize_interval_hours: int = field(
        default_factory=lambda: _env_int("AUTO_OPTIMIZE_INTERVAL_HOURS", 24)
    )
    # Multi-timeframe confirmation: require higher-TF trend agreement
    mtf_enabled: bool = field(
        default_factory=lambda: _env_bool("MTF_ENABLED", True)
    )
    mtf_timeframe: str = field(
        default_factory=lambda: _env("MTF_TIMEFRAME", "HOUR")
    )
    # Time-based exit: close trades that haven't moved after N minutes
    stale_trade_exit_enabled: bool = field(
        default_factory=lambda: _env_bool("STALE_TRADE_EXIT_ENABLED", True)
    )
    stale_trade_minutes: int = field(
        default_factory=lambda: _env_int("STALE_TRADE_MINUTES", 60)
    )
    # Adaptive position sizing
    adaptive_sizing_enabled: bool = field(
        default_factory=lambda: _env_bool("ADAPTIVE_SIZING_ENABLED", True)
    )
    # Breakeven trailing stop settings
    trailing_stop_breakeven_r: float = field(
        default_factory=lambda: _env_float("TRAILING_STOP_BREAKEVEN_R", 1.0)
    )
    # Daily trade limits & recovery mode
    max_daily_trades: int = field(
        default_factory=lambda: _env_int("MAX_DAILY_TRADES", 20)
    )
    recovery_extra_trades: int = field(
        default_factory=lambda: _env_int("RECOVERY_EXTRA_TRADES", 10)
    )
    daily_profit_target_pct: float = field(
        default_factory=lambda: _env_float("DAILY_PROFIT_TARGET_PCT", 10.0)
    )
    min_trade_value: float = field(
        default_factory=lambda: _env_float("MIN_TRADE_VALUE", 50.0)
    )
    # ADX trend regime filter: skip trades in choppy markets (ADX < threshold)
    adx_filter_enabled: bool = field(
        default_factory=lambda: _env_bool("ADX_FILTER_ENABLED", True)
    )
    adx_min_threshold: float = field(
        default_factory=lambda: _env_float("ADX_MIN_THRESHOLD", 15.0)
    )


settings = Settings()
