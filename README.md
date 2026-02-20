# Elite Trading System

A modular trading platform for Binance/Bitget with automated strategy testing and execution.

## Features
- **Bulletproof Pair Discovery**: Scans Binance & Bitget to dynamically find the best liquid USDT perpetuals while blocking low-liquidity coins.
- **Advanced Strategies**: Vectorized grid simulation, comprehensive scaling execution, and robust live order generation for Grid Trading and Scalping.
- **News Sentiment Guard**: Automatically pauses trading using CryptoPanic's API if market sentiment drops below -0.3.
- **Funding Rate Arbitrage**: Delta-neutral strategy for passively capturing funding fee spreads between exchanges.
- **FastAPI Dashboard**: Real-time monitoring of status, positions, priority list, and live execution logs.
- **Telegram Alerts**: Push notifications for trade executions, risk events, and daily PnL summaries.
- **Research Module**: Auto-generate and backtest trading strategies
- **Core Engine**: Execute trades based on priority list with 10x leverage and risk management built-in.
- **Robust Logging**: JSON-formatted logs for easy parsing

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
copy config\.env.example config\.env
# Edit config\.env with your Binance/Bitget, CRYPTOPANIC_API_KEY
# Add TELEGRAM_TOKEN, TELEGRAM_CHAT_ID for alerts
# Add DASHBOARD_USERNAME, DASHBOARD_PASSWORD for web auth
```

4. Run:
```bash
# Quick Research Mode - 15m timeframe, 6 solid strategies (including Scalp/Grid), fast backtesting
python main.py --mode research --quick

# Trade mode - execute trades (paper trading by default)
python main.py --mode trade --paper

# Trade mode with Dashboard and Telegram enabled
python main.py --mode trade --paper --dashboard --telegram

# Full auto mode
python main.py --mode both
```

## Project Structure
```text
├── config/          # Configuration and API keys
├── core/            # Trading engine, execution, robust Grid live strategy
├── research/        # Strategy testing, fast vectorized backtesting, pair discovery
├── utils/           # Logging, sentiment guard, database utilities
├── data/            # Database, logs, and outputs
└── tests/           # Unit tests
```
