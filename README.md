# Capital.com Automated Trading Bot

A production-ready automated trading bot for Capital.com built with Python and FastAPI. Features a modular architecture with strategy engine, risk management, Telegram notifications, and an admin dashboard.

## Features

- **Capital.com API Integration**: Authentication, market data, order placement, position management
- **Strategy Engine**: EMA crossover + RSI + breakout confirmation
- **Risk Management**: Position sizing, daily loss limits, consecutive loss tracking, cooldown periods, duplicate prevention
- **Multiple Trading Modes**: Analysis, Paper, Demo, Live
- **Telegram Notifications**: Trade alerts, error alerts, daily summaries
- **Admin Dashboard**: Real-time monitoring with kill switch
- **Database Logging**: All signals, orders, positions, errors, and balance history
- **Docker Ready**: Full Docker and docker-compose support

## Architecture

```
app/
├── main.py                  # FastAPI entry point
├── config.py                # Configuration from environment
├── api/
│   ├── capital_client.py    # Capital.com REST API client
│   └── webhook.py           # External signal webhook
├── strategy/
│   ├── indicators.py        # EMA, RSI, ATR calculations
│   ├── rules.py             # Trading rule evaluators
│   └── signals.py           # Signal generator
├── execution/
│   ├── risk_manager.py      # Risk checks and position sizing
│   ├── order_manager.py     # Order lifecycle management
│   └── position_manager.py  # Position tracking and P&L
├── services/
│   ├── market_data.py       # Market data fetching
│   ├── notifier.py          # Telegram notifications
│   └── scheduler.py         # Trading cycle orchestrator
├── db/
│   ├── models.py            # SQLAlchemy models
│   └── session.py           # Database session management
├── dashboard/
│   └── routes.py            # Dashboard API and UI
└── templates/
    └── dashboard.html       # Dashboard web interface
```

## Trading Strategy

### Buy Signal
- EMA20 > EMA50 (bullish trend)
- RSI between 55 and 70
- Current candle closes above previous high (breakout)
- Spread within acceptable threshold
- No existing long position

### Sell Signal
- EMA20 < EMA50 (bearish trend)
- RSI between 30 and 45
- Current candle closes below previous low (breakout)
- Spread within acceptable threshold
- No existing short position

### Exit Rules
- Stop loss based on ATR multiplier
- Take profit at configurable risk-reward ratio (default 1:2)

## Risk Management

- Configurable risk per trade (default 1%)
- Maximum daily loss limit (default 5%)
- Maximum consecutive losses before cooldown (default 3)
- Cooldown period after consecutive losses (default 30 min)
- Maximum one open trade per symbol
- No duplicate orders
- Global kill switch

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry

### Local Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <repo-url>
   cd trading-bot
   poetry install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Capital.com API credentials
   ```

3. **Run in analysis mode** (no real trades):
   ```bash
   # Set TRADING_MODE=analysis in .env
   poetry run fastapi dev app/main.py
   ```

4. **Access the dashboard**:
   Open http://localhost:8000/dashboard

5. **Run tests**:
   ```bash
   poetry run pytest tests/ -v
   ```

### Docker Setup

1. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Build and run**:
   ```bash
   docker-compose up -d
   ```

3. **View logs**:
   ```bash
   docker-compose logs -f trading-bot
   ```

### VPS Deployment

1. **Set up Ubuntu VPS** (2-4 vCPU, 4-8 GB RAM recommended)

2. **Install Docker and Docker Compose**:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   sudo systemctl enable docker
   ```

3. **Deploy**:
   ```bash
   git clone <repo-url>
   cd trading-bot
   cp .env.example .env
   nano .env  # Configure your settings
   docker-compose up -d
   ```

4. **Optional: Set up Nginx reverse proxy** for HTTPS access to the dashboard.

## Trading Modes

| Mode | Description |
|------|-------------|
| `analysis` | Fetch data, run strategy, log signals. No execution. |
| `paper` | Simulate trades with virtual P&L tracking. |
| `demo` | Place real orders on Capital.com demo account. |
| `live` | Place real orders on Capital.com live account. |

**Recommended progression**: analysis -> paper -> demo -> live (with tiny size)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/api/config` | GET | Current bot configuration |
| `/api/run-cycle` | POST | Manually trigger trading cycle |
| `/webhook/signal` | POST | Receive external trading signal |
| `/dashboard/` | GET | Admin dashboard UI |
| `/dashboard/api/status` | GET | Bot status |
| `/dashboard/api/stats` | GET | Trading statistics |
| `/dashboard/api/positions` | GET | Position history |
| `/dashboard/api/signals` | GET | Signal history |
| `/dashboard/api/daily-pnl` | GET | Daily P&L history |
| `/dashboard/api/errors` | GET | Error log |
| `/dashboard/api/kill-switch` | POST | Toggle kill switch |

## Configuration

All settings are configurable via environment variables. See `.env.example` for the full list.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADING_SYMBOL` | XAUUSD | Trading instrument |
| `TRADING_TIMEFRAME` | HOUR | Candle timeframe |
| `TRADING_MODE` | demo | Trading mode |
| `RISK_PER_TRADE` | 0.01 | Risk 1% per trade |
| `MAX_DAILY_LOSS` | 0.05 | Stop after 5% daily loss |
| `MAX_CONSECUTIVE_LOSSES` | 3 | Cooldown after 3 losses |
| `SL_ATR_MULTIPLIER` | 1.5 | Stop loss = 1.5x ATR |
| `TP_RISK_REWARD` | 2.0 | Take profit at 1:2 R:R |

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID via [@userinfobot](https://t.me/userinfobot)
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`

## Safety Notes

- **Always start in analysis or paper mode**
- **Never risk more than you can afford to lose**
- **Test thoroughly on demo before going live**
- **Monitor the bot regularly**
- **Use the kill switch if anything looks wrong**
- **Start with a single symbol and small position sizes**

## License

MIT
