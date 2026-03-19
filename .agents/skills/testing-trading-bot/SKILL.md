# Testing the Capital.com Trading Bot

## Overview
This bot runs as a Docker container on a VPS, with a FastAPI dashboard at port 8000. Testing involves verifying unit tests locally, checking VPS logs for feature activity, and testing the dashboard UI.

## Devin Secrets Needed
- `VPS_PASSWORD` — SSH password for the VPS (administrator@93.127.138.91)
- `GITHUB_TOKEN` — GitHub PAT for pushing code

## Running Unit Tests
```bash
cd /home/ubuntu/trading-bot && python -m pytest tests/ -v
```
Expect 49 tests to pass. Tests cover indicators, rules, risk manager, and signals.

## VPS Access
```bash
sshpass -p '$VPS_PASSWORD' ssh -o StrictHostKeyChecking=no administrator@93.127.138.91 "<command>"
```

## Checking VPS Config
```bash
# Verify .env values on VPS
sshpass -p '$VPS_PASSWORD' ssh -o StrictHostKeyChecking=no administrator@93.127.138.91 \
  "cat /home/administrator/bot/.env | grep -E 'RISK_PER_TRADE|MAX_DAILY|RECOVERY|DAILY_PROFIT'"
```

## Checking Docker Logs for Feature Activity
The bot logs key events that can be grep'd to verify features:

```bash
# Symbol ranking (should appear every ~5 min cycle)
docker logs capital-trading-bot 2>&1 | grep 'Symbol ranking top 5'

# Daily P&L tracking (shows bot_trades_today counter)
docker logs capital-trading-bot 2>&1 | grep 'Daily P&L update'

# Recovery mode activation
docker logs capital-trading-bot 2>&1 | grep -iE 'Recovery|RECOVERY'

# Daily counter reset at midnight UTC
docker logs capital-trading-bot 2>&1 | grep 'Daily counters reset'

# Rate limiting errors (should be zero)
docker logs capital-trading-bot 2>&1 | grep -iE 'rate|429|throttl'

# FILLED orders (verify trade sizes)
docker logs capital-trading-bot 2>&1 | grep 'Order placed.*FILLED'
```

## Dashboard Testing
- **URL:** http://93.127.138.91:8000/dashboard/
- The dashboard auto-refreshes every 5 seconds
- Key sections to verify:
  - Today's P&L banner (top) — shows P&L, trades, wins, losses, win rate
  - Account cards — Balance, Equity, Available Funds, Unrealized P&L
  - Bot Status — Running/Stopped, mode, symbols, timeframe
  - Live Positions table — Symbol, Direction, Size, Entry, Current, Invested, P&L, SL, Suggestion
  - Tabs: Today's Trades, Trade History, Signals, Errors
  - Filter buttons on Trade History: All Trades, Profits Only, Losses Only

## API Endpoints to Verify
- `GET /dashboard/api/status` — Bot running state, mode, symbols, features
- `GET /dashboard/api/live-positions` — Open positions with invested amounts
- `GET /dashboard/api/today-pnl` — Today's closed trades P&L
- `GET /dashboard/api/stats` — Overall trading statistics

## Important Testing Notes

### Daily Counter Reset
The daily counters (trades, P&L, wins, losses) reset at midnight UTC. If testing near midnight, you may see the counters reset during your test — this is expected behavior and actually confirms the reset logic works.

### Recovery Mode
Recovery mode activates when `daily_losses > daily_wins` AND `daily_trades >= 5` (session-scoped). The `daily_trades` counter is session-only (resets on bot restart), while `daily_losses`/`daily_wins` come from the DB. This means recovery mode may not trigger immediately after a restart even if historical losses > wins, because the session trade counter starts at 0.

### Daily Profit Target
The 10% daily profit target check uses `daily_pnl / account_balance * 100`. It can only trigger when P&L is positive, so it won't activate during losing sessions.

### USDSEK Position
There may be a manually-opened USDSEK position with very large invested amount. This is NOT from the bot — the bot only trades the 27 configured symbols. The dashboard correctly flags it with "CLOSE - Loss exceeds 3% of account".

### Rebuilding Docker on VPS
```bash
cd /home/administrator/bot && docker compose down && docker compose up -d --build
```
Wait ~30 seconds for the bot to start and authenticate with Capital.com before checking logs.
