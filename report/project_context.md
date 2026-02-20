# Elite Trading System - Project Context

This document contains the full source code and structure of the Elite Trading System project for AI analysis.

## Project Structure
```text
DeadEnd/
    generate_context.py
    main.py
    README.md
    requirements.txt
    config/
        .env.example
        settings.py
        __init__.py
    core/
        engine.py
        executor.py
        risk_manager.py
        __init__.py
    research/
        backtester.py
        data_pipeline.py
        priority_list.py
        strategy_generator.py
        __init__.py
    tests/
        test_backtester.py
        test_executor.py
        test_logger.py
        __init__.py
    utils/
        database.py
        logger.py
        __init__.py
```

## File Contents

### main.py
```py
#!/usr/bin/env python3
"""
Elite Trading System - Main Entry Point

Usage:
    python main.py --mode research        # Run strategy discovery
    python main.py --mode trade --paper   # Paper trading
    python main.py --mode trade --live    # Live trading (careful!)
    python main.py --mode both            # Research then trade
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.engine import TradingEngine, EngineMode
from config.settings import ExchangeType
from utils.logger import get_logger, setup_logging

logger = get_logger('main')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Elite Trading System - Automated crypto trading with strategy discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run research to find best strategies
    python main.py --mode research --pairs BTC/USDT ETH/USDT
    
    # Paper trading with top strategies
    python main.py --mode trade --paper --interval 60
    
    # Full auto mode (research then trade)
    python main.py --mode both --paper
    
    # Quick test with minimal strategies
    python main.py --mode research --quick
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['research', 'trade', 'both', 'status'],
        default='research',
        help='Operating mode (default: research)'
    )
    
    parser.add_argument(
        '--pairs', '-p',
        nargs='+',
        default=None,
        help='Trading pairs (default: BTC/USDT ETH/USDT SOL/USDT)'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Enable paper trading mode (simulated)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading mode (real money!)'
    )
    
    parser.add_argument(
        '--exchange', '-e',
        choices=['binance', 'bitget'],
        default='binance',
        help='Exchange to use (default: binance)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Trading loop interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode with minimal strategy set'
    )
    
    parser.add_argument(
        '--max-strategies',
        type=int,
        default=None,
        help='Maximum strategies to test per pair'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           ELITE TRADING SYSTEM v1.0                           ║
║           Automated Strategy Discovery & Execution            ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status():
    """Print current system status"""
    from utils.database import get_database
    from research.priority_list import PriorityListManager
    
    settings = get_settings()
    db = get_database()
    priority_manager = PriorityListManager()
    
    print("\n=== SYSTEM STATUS ===\n")
    
    # Settings
    print(f"Trading Mode: {'PAPER' if settings.is_paper_mode else 'LIVE'}")
    print(f"Starting Balance: {settings.starting_balance} USDT")
    print(f"Futures Target: {settings.futures_target} USDT")
    print(f"Risk Per Trade: {settings.risk_per_trade:.1%}")
    print(f"Max Drawdown: {settings.max_drawdown:.1%}")
    print(f"Default Leverage: {settings.default_leverage}x")
    
    # Balance
    current = db.get_current_balance()
    if current:
        print(f"\nCurrent Balance: {current:.2f} USDT")
        drawdown = db.calculate_drawdown()
        print(f"Current Drawdown: {drawdown['current_drawdown']:.1%}")
        print(f"Peak Balance: {drawdown['peak_balance']:.2f} USDT")
    
    # Strategies
    strategies = db.get_top_strategies(limit=5)
    if strategies:
        print(f"\nTop 5 Strategies:")
        for s in strategies:
            print(f"  - {s['name']} on {s['pair']}: ROI={s['roi']*100:.1f}%, Sharpe={s['sharpe_ratio']:.2f}")
    
    # Priority list
    entries = priority_manager.load()
    if entries:
        print(f"\nPriority List ({len(entries)} entries):")
        priority_manager.print_summary(entries[:5])
    
    # Recent trades
    trades = db.get_recent_trades(limit=5)
    if trades:
        print(f"\nRecent Trades:")
        for t in trades:
            print(f"  - {t['side'].upper()} {t['amount']:.4f} {t['pair']} @ {t['price']:.2f}")


def run_research_mode(args, engine: TradingEngine):
    """Run research/backtesting mode"""
    logger.info("Starting Research Mode")
    
    if args.quick:
        results = engine.run_quick_research(args.pairs)
    else:
        results = engine.run_research(
            pairs=args.pairs,
            max_strategies=args.max_strategies
        )
    
    print(f"\n✅ Research complete: {len(results)} strategies tested")
    print("Priority list saved to data/priority_list.json")


def run_trading_mode(args, engine: TradingEngine):
    """Run trading mode"""
    if args.live and not args.paper:
        print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("You are about to trade with REAL MONEY!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return
    
    logger.info(f"Starting Trading Mode ({'PAPER' if engine.paper_mode else 'LIVE'})")
    engine.run_trading_loop(interval_seconds=args.interval)


def main():
    """Main entry point"""
    args = parse_args()
    
    print_banner()
    
    # Setup logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    # Determine paper mode
    paper_mode = True  # Default to paper
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    
    # Handle status mode separately
    if args.mode == 'status':
        print_status()
        return
    
    # Create engine
    exchange_type = ExchangeType.BINANCE if args.exchange == 'binance' else ExchangeType.BITGET
    
    engine = TradingEngine(
        mode=EngineMode(args.mode) if args.mode != 'both' else EngineMode.BOTH,
        exchange_type=exchange_type,
        paper_mode=paper_mode
    )
    
    print(f"\n📊 Mode: {args.mode.upper()}")
    print(f"💰 Trading: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"🏦 Exchange: {args.exchange.upper()}")
    print(f"📈 Pairs: {args.pairs or 'default'}\n")
    
    try:
        if args.mode == 'research':
            run_research_mode(args, engine)
            
        elif args.mode == 'trade':
            run_trading_mode(args, engine)
            
        elif args.mode == 'both':
            run_research_mode(args, engine)
            print("\n" + "="*50)
            print("Research complete. Starting trading...")
            print("="*50 + "\n")
            run_trading_mode(args, engine)
            
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
```

### README.md
```md
# Elite Trading System

A modular trading platform for Binance/Bitget with automated strategy testing and execution.

## Features
- **Research Module**: Auto-generate and backtest trading strategies
- **Core Engine**: Execute trades based on priority list
- **Risk Manager**: Position sizing and drawdown protection
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
# Edit config\.env with your API keys
```

4. Run:
```bash
# Research mode - find best strategies
python main.py --mode research

# Trade mode - execute trades (paper trading by default)
python main.py --mode trade --paper

# Full auto mode
python main.py --mode both
```

## Project Structure
```
├── config/          # Configuration and API keys
├── core/            # Trading engine and execution
├── research/        # Strategy testing and backtesting
├── utils/           # Logging and database utilities
├── data/            # Database, logs, and outputs
└── tests/           # Unit tests
```
```

### requirements.txt
```txt
# Core dependencies
ccxt>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
ta>=0.10.0
python-dotenv>=1.0.0

# Backtesting
backtrader>=1.9.78

# Database
sqlalchemy>=2.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Utilities
requests>=2.28.0
aiohttp>=3.8.0
```

### config\.env.example
```env
# Binance API
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Bitget API
BITGET_API_KEY=your_bitget_api_key_here
BITGET_API_SECRET=your_bitget_api_secret_here
BITGET_PASSPHRASE=your_bitget_passphrase_here

# Trading Configuration
STARTING_BALANCE=50
RISK_PER_TRADE=0.02
MAX_DRAWDOWN=0.20
FUTURES_TARGET=2000
DEFAULT_LEVERAGE=5

# Mode: paper or live
TRADING_MODE=paper

# Logging
LOG_LEVEL=DEBUG
```

### config\settings.py
```py
"""
Configuration settings for Elite Trading System
Loads from .env file and provides exchange connections
"""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
import ccxt

# Load .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try example file for structure reference
    example_path = Path(__file__).parent / '.env.example'
    if example_path.exists():
        load_dotenv(example_path)


class ExchangeType(Enum):
    BINANCE = "binance"
    BITGET = "bitget"


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class Settings:
    """Application settings loaded from environment"""
    
    # Binance
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_api_secret: str = os.getenv('BINANCE_API_SECRET', '')
    
    # Bitget
    bitget_api_key: str = os.getenv('BITGET_API_KEY', '')
    bitget_api_secret: str = os.getenv('BITGET_API_SECRET', '')
    bitget_passphrase: str = os.getenv('BITGET_PASSPHRASE', '')
    
    # Trading params
    starting_balance: float = float(os.getenv('STARTING_BALANCE', '50'))
    risk_per_trade: float = float(os.getenv('RISK_PER_TRADE', '0.02'))
    max_drawdown: float = float(os.getenv('MAX_DRAWDOWN', '0.20'))
    futures_target: float = float(os.getenv('FUTURES_TARGET', '2000'))
    default_leverage: int = int(os.getenv('DEFAULT_LEVERAGE', '5'))
    
    # Mode
    trading_mode: str = os.getenv('TRADING_MODE', 'paper')
    log_level: str = os.getenv('LOG_LEVEL', 'DEBUG')
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / 'data'
    logs_dir: Path = data_dir / 'logs'
    db_path: Path = data_dir / 'strategies.db'
    priority_list_path: Path = data_dir / 'priority_list.json'
    
    # Default trading pairs
    default_pairs: list = None
    
    def __post_init__(self):
        if self.default_pairs is None:
            self.default_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @property
    def is_paper_mode(self) -> bool:
        return self.trading_mode.lower() == 'paper'
    
    def has_binance_keys(self) -> bool:
        return bool(self.binance_api_key and self.binance_api_secret 
                   and not self.binance_api_key.startswith('your_'))
    
    def has_bitget_keys(self) -> bool:
        return bool(self.bitget_api_key and self.bitget_api_secret 
                   and not self.bitget_api_key.startswith('your_'))


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_exchange(exchange_type: ExchangeType, futures: bool = True) -> ccxt.Exchange:
    """
    Create and return a configured exchange instance
    
    Args:
        exchange_type: Which exchange to connect to
        futures: If True, use futures/swap markets
    
    Returns:
        Configured ccxt Exchange instance
    """
    settings = get_settings()
    
    if exchange_type == ExchangeType.BINANCE:
        exchange_class = ccxt.binance
        config = {
            'apiKey': settings.binance_api_key,
            'secret': settings.binance_api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if futures else 'spot',
                'adjustForTimeDifference': True
            }
        }
    elif exchange_type == ExchangeType.BITGET:
        exchange_class = ccxt.bitget
        config = {
            'apiKey': settings.bitget_api_key,
            'secret': settings.bitget_api_secret,
            'password': settings.bitget_passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap' if futures else 'spot'
            }
        }
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")
    
    # Create exchange
    exchange = exchange_class(config)
    
    # Enable sandbox/testnet for paper trading
    if settings.is_paper_mode:
        exchange.set_sandbox_mode(True)
    
    return exchange


def get_public_exchange(exchange_type: ExchangeType = ExchangeType.BINANCE) -> ccxt.Exchange:
    """Get exchange instance for public data only (no auth required)"""
    if exchange_type == ExchangeType.BINANCE:
        return ccxt.binance({'enableRateLimit': True})
    elif exchange_type == ExchangeType.BITGET:
        return ccxt.bitget({'enableRateLimit': True})
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")
```

### config\__init__.py
```py
"""
Elite Trading System - Configuration Module
"""
from config.settings import (
    Settings,
    get_settings,
    get_exchange,
    ExchangeType
)

__all__ = ['Settings', 'get_settings', 'get_exchange', 'ExchangeType']
```

### core\engine.py
```py
"""
Main Trading Engine for Elite Trading System

Orchestrates strategy execution, risk management, and order flow.
"""

import time
import signal
import sys
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import threading

from config.settings import get_settings, get_exchange, ExchangeType
from core.executor import OrderExecutor, OrderStatus
from core.risk_manager import RiskManager, RiskLevel, TradingPhase
from research.priority_list import PriorityListManager, PriorityEntry
from research.backtester import Backtester
from research.strategy_generator import StrategyGenerator, StrategyParams, StrategyType
from research.data_pipeline import get_data_pipeline
from utils.logger import get_logger, setup_logging
from utils.database import get_database

logger = get_logger('engine')


class EngineMode(Enum):
    """Engine operating modes"""
    IDLE = "idle"
    RESEARCH = "research"
    TRADING = "trading"
    BOTH = "both"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Trading signal from strategy"""
    pair: str
    strategy: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime


class TradingEngine:
    """
    Main trading engine that orchestrates the entire system.
    
    Responsibilities:
    - Run research/backtesting
    - Generate and execute signals
    - Manage risk
    - Handle graceful shutdown
    """
    
    def __init__(
        self,
        mode: EngineMode = EngineMode.TRADING,
        exchange_type: ExchangeType = ExchangeType.BINANCE,
        paper_mode: bool = None
    ):
        # Setup
        self.settings = get_settings()
        setup_logging()
        
        self.mode = mode
        self.exchange_type = exchange_type
        self.paper_mode = paper_mode if paper_mode is not None else self.settings.is_paper_mode
        
        # Components
        self.executor = OrderExecutor(exchange_type, self.paper_mode)
        self.risk_manager = RiskManager()
        self.priority_manager = PriorityListManager()
        self.backtester = Backtester()
        self.pipeline = get_data_pipeline()
        self.db = get_database()
        
        # State
        self._running = False
        self._shutdown_event = threading.Event()
        self._current_signals: List[TradingSignal] = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Trading Engine initialized | Mode: {mode.value} | Paper: {self.paper_mode}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping gracefully...")
        self._shutdown_event.set()
        self._running = False
    
    # ==================== Research Methods ====================
    
    def run_research(
        self,
        pairs: List[str] = None,
        strategy_types: List[StrategyType] = None,
        timeframe: str = '1h',
        limit: int = 1000,
        max_strategies: int = None
    ) -> List[Dict]:
        """
        Run strategy discovery and backtesting.
        
        Args:
            pairs: Trading pairs to test
            strategy_types: Types of strategies to generate
            timeframe: Candle timeframe for backtesting
            limit: Number of candles per backtest
            max_strategies: Maximum strategies to test per pair
        
        Returns:
            List of backtest results
        """
        pairs = pairs or self.settings.default_pairs
        
        logger.info(f"Starting research: {len(pairs)} pairs")
        
        # Generate strategies
        generator = StrategyGenerator()
        strategies = list(generator.generate_all(strategy_types, limit=max_strategies))
        
        logger.info(f"Generated {len(strategies)} strategy combinations")
        
        # Run backtests
        results = self.backtester.run_batch(pairs, strategies, timeframe, limit)
        
        # Generate priority list
        entries = self.priority_manager.generate()
        self.priority_manager.save(entries)
        self.priority_manager.print_summary(entries)
        
        logger.info(f"Research complete: {len(results)} backtests, {len(entries)} in priority list")
        
        return [r.to_dict() for r in results]
    
    def run_quick_research(self, pairs: List[str] = None) -> List[Dict]:
        """Run a quick research with minimal strategy set"""
        generator = StrategyGenerator(StrategyGenerator.get_quick_test_config())
        strategies = list(generator.generate_all())
        
        return self.run_research(
            pairs=pairs,
            max_strategies=len(strategies)
        )
    
    # ==================== Signal Generation ====================
    
    def _generate_signal(
        self,
        pair: str,
        strategy: StrategyParams
    ) -> Optional[TradingSignal]:
        """Generate trading signal for a pair using given strategy"""
        try:
            import ta
            import numpy as np
            
            # Fetch recent data
            df = self.pipeline.fetch_ohlcv(pair, '1h', 100)
            
            if df.empty or len(df) < 50:
                return None
            
            params = strategy.params
            signal_type = SignalType.HOLD
            
            # Calculate indicators and generate signal based on strategy type
            if strategy.strategy_type == StrategyType.SMA_CROSSOVER:
                df['ma_short'] = ta.trend.sma_indicator(df['close'], window=params['short_period'])
                df['ma_long'] = ta.trend.sma_indicator(df['close'], window=params['long_period'])
                
                # Check for crossover on latest candle
                if df['ma_short'].iloc[-1] > df['ma_long'].iloc[-1] and df['ma_short'].iloc[-2] <= df['ma_long'].iloc[-2]:
                    signal_type = SignalType.BUY
                elif df['ma_short'].iloc[-1] < df['ma_long'].iloc[-1] and df['ma_short'].iloc[-2] >= df['ma_long'].iloc[-2]:
                    signal_type = SignalType.SELL
            
            elif strategy.strategy_type == StrategyType.RSI_REVERSAL:
                df['rsi'] = ta.momentum.rsi(df['close'], window=params['period'])
                
                if df['rsi'].iloc[-1] > params['oversold'] and df['rsi'].iloc[-2] <= params['oversold']:
                    signal_type = SignalType.BUY
                elif df['rsi'].iloc[-1] < params['overbought'] and df['rsi'].iloc[-2] >= params['overbought']:
                    signal_type = SignalType.SELL
            
            # Add more strategy types as needed...
            
            if signal_type != SignalType.HOLD:
                return TradingSignal(
                    pair=pair,
                    strategy=strategy.name,
                    signal_type=signal_type,
                    confidence=0.75,  # Could be calculated based on indicator values
                    price=df['close'].iloc[-1],
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generation failed for {pair}: {e}")
            return None
    
    def _process_signal(self, signal: TradingSignal, entry: PriorityEntry) -> bool:
        """Process a trading signal and execute if appropriate"""
        balance = self.executor.get_balance()
        current_balance = balance.get('USDT', 0)
        
        # Check risk
        can_trade, reason = self.risk_manager.can_open_position(signal.pair, current_balance)
        if not can_trade:
            logger.warning(f"Cannot open position: {reason}")
            return False
        
        # Calculate position size
        leverage = self.settings.default_leverage if self.risk_manager.get_status(current_balance).phase == TradingPhase.FUTURES_GROWTH else 1
        position_value = self.risk_manager.calculate_position_size_simple(current_balance, leverage)
        position_size = position_value / signal.price
        
        # Execute order
        if signal.signal_type == SignalType.BUY:
            order = self.executor.execute(
                pair=signal.pair,
                side='buy',
                amount=position_size,
                order_type='market',
                strategy=signal.strategy
            )
            
            if order.status == OrderStatus.FILLED:
                self.risk_manager.open_position(
                    pair=signal.pair,
                    size=order.filled_amount,
                    entry_price=order.average_price,
                    side='buy'
                )
                return True
        
        elif signal.signal_type == SignalType.SELL:
            # Close existing position
            position = self.executor.get_position(signal.pair)
            if position and position.get('amount', 0) > 0:
                order = self.executor.execute(
                    pair=signal.pair,
                    side='sell',
                    amount=position['amount'],
                    order_type='market',
                    strategy=signal.strategy
                )
                
                if order.status == OrderStatus.FILLED:
                    self.risk_manager.close_position(signal.pair, order.average_price)
                    return True
        
        return False
    
    # ==================== Trading Loop ====================
    
    def run_trading_loop(
        self,
        interval_seconds: int = 60,
        max_iterations: int = None
    ):
        """
        Main trading loop.
        
        Args:
            interval_seconds: Time between iterations
            max_iterations: Maximum iterations (None = infinite)
        """
        # Load priority list
        entries = self.priority_manager.load()
        
        if not entries:
            logger.warning("No entries in priority list. Run research first!")
            return
        
        logger.info(f"Starting trading loop with {len(entries)} strategies")
        
        self._running = True
        iteration = 0
        
        while self._running:
            if max_iterations and iteration >= max_iterations:
                break
            
            if self._shutdown_event.is_set():
                break
            
            try:
                iteration += 1
                logger.info(f"Trading iteration {iteration}")
                
                # Update balance and check risk
                balance = self.executor.get_balance()
                current_balance = balance.get('USDT', 0)
                status = self.risk_manager.update_balance(current_balance, "trading_loop")
                
                if not status.can_trade:
                    logger.warning(f"Cannot trade: {status.message}")
                    if status.level == RiskLevel.CRITICAL:
                        logger.error("Critical risk level - stopping trading")
                        break
                    time.sleep(interval_seconds)
                    continue
                
                # Check for phase transition
                if self.risk_manager.should_switch_to_spot(current_balance):
                    logger.info("🎉 Target reached! Switching to spot mode")
                    self.executor.futures = False
                
                # Generate signals from priority list
                for entry in entries[:3]:  # Top 3 strategies
                    strategy_params = StrategyParams(
                        strategy_type=StrategyType.SMA_CROSSOVER,  # Would need to map from entry
                        name=entry.strategy,
                        params=entry.params
                    )
                    
                    signal = self._generate_signal(entry.pair, strategy_params)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        logger.info(f"Signal: {signal.signal_type.value.upper()} {signal.pair} ({signal.strategy})")
                        self._process_signal(signal, entry)
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval_seconds)
        
        logger.info("Trading loop stopped")
    
    # ==================== Entry Points ====================
    
    def run(
        self,
        mode: EngineMode = None,
        pairs: List[str] = None,
        interval: int = 60
    ):
        """
        Main entry point.
        
        Args:
            mode: Operating mode (research, trading, both)
            pairs: Trading pairs
            interval: Trading loop interval in seconds
        """
        mode = mode or self.mode
        
        logger.info(f"Starting Trading Engine in {mode.value} mode")
        
        try:
            if mode == EngineMode.RESEARCH:
                self.run_research(pairs)
                
            elif mode == EngineMode.TRADING:
                self.run_trading_loop(interval)
                
            elif mode == EngineMode.BOTH:
                # Run research first, then trade
                self.run_research(pairs)
                self.run_trading_loop(interval)
            
        except KeyboardInterrupt:
            logger.info("Engine stopped by user")
        
        except Exception as e:
            logger.error(f"Engine error: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup on shutdown"""
        logger.info("Cleaning up...")
        
        # Log final status
        balance = self.executor.get_balance()
        logger.balance_update(balance.get('USDT', 0), source="shutdown")
        
        logger.info("Engine shutdown complete")
```

### core\executor.py
```py
"""
Order Executor for Elite Trading System

Handles order execution via CCXT with retry logic,
paper trading mode, and comprehensive logging.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

import ccxt

from config.settings import get_settings, get_exchange, ExchangeType
from utils.logger import get_logger
from utils.database import get_database

logger = get_logger('executor')


class OrderStatus(Enum):
    """Order status codes"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Order representation"""
    id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution details
    exchange_order_id: Optional[str] = None
    filled_amount: float = 0.0
    average_price: Optional[float] = None
    cost: float = 0.0
    fee: float = 0.0
    
    # Metadata
    strategy: Optional[str] = None
    is_paper: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pair': self.pair,
            'side': self.side.value,
            'type': self.order_type.value,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'exchange_order_id': self.exchange_order_id,
            'filled_amount': self.filled_amount,
            'average_price': self.average_price,
            'cost': self.cost,
            'fee': self.fee,
            'strategy': self.strategy,
            'is_paper': self.is_paper,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None
        }


class OrderExecutor:
    """
    Executes orders on exchanges with paper trading support.
    
    Features:
    - Paper trading mode (simulated execution)
    - Retry logic with exponential backoff
    - Logging of all order activities
    - Database persistence
    """
    
    def __init__(
        self,
        exchange_type: ExchangeType = ExchangeType.BINANCE,
        paper_mode: bool = None,
        futures: bool = True,
        leverage: int = None
    ):
        self.settings = get_settings()
        self.exchange_type = exchange_type
        self.paper_mode = paper_mode if paper_mode is not None else self.settings.is_paper_mode
        self.futures = futures
        self.leverage = leverage or self.settings.default_leverage
        
        self.db = get_database()
        self._exchange: Optional[ccxt.Exchange] = None
        
        # Paper trading simulated balance
        self._paper_balance = self.settings.starting_balance
        self._paper_positions: Dict[str, float] = {}
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._executed_orders: List[Order] = []
    
    @property
    def exchange(self) -> ccxt.Exchange:
        """Get or create exchange connection"""
        if self._exchange is None:
            self._exchange = get_exchange(self.exchange_type, self.futures)
            logger.info(f"Connected to {self.exchange_type.value} ({'futures' if self.futures else 'spot'})")
        return self._exchange
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"order_{uuid.uuid4().hex[:12]}"
    
    def _get_current_price(self, pair: str) -> float:
        """Get current market price for a pair"""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get price for {pair}: {e}")
            raise
    
    def _execute_paper_order(self, order: Order) -> Order:
        """Execute order in paper trading mode"""
        try:
            # Get current price
            price = self._get_current_price(order.pair)
            
            # Calculate cost
            cost = order.amount * price
            fee = cost * 0.001  # 0.1% fee simulation
            
            if order.side == OrderSide.BUY:
                # Check balance
                if cost + fee > self._paper_balance:
                    order.status = OrderStatus.FAILED
                    logger.trade_failed(order.pair, order.side.value, "Insufficient balance", order.strategy)
                    return order
                
                # Execute
                self._paper_balance -= (cost + fee)
                base_asset = order.pair.split('/')[0]
                self._paper_positions[base_asset] = self._paper_positions.get(base_asset, 0) + order.amount
                
            else:  # SELL
                base_asset = order.pair.split('/')[0]
                position = self._paper_positions.get(base_asset, 0)
                
                if order.amount > position:
                    order.status = OrderStatus.FAILED
                    logger.trade_failed(order.pair, order.side.value, "Insufficient position", order.strategy)
                    return order
                
                # Execute
                self._paper_positions[base_asset] -= order.amount
                self._paper_balance += (cost - fee)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.exchange_order_id = f"paper_{uuid.uuid4().hex[:8]}"
            order.filled_amount = order.amount
            order.average_price = price
            order.cost = cost
            order.fee = fee
            order.executed_at = datetime.now()
            
            logger.trade_executed(
                pair=order.pair,
                side=order.side.value,
                amount=order.amount,
                price=price,
                order_id=order.exchange_order_id,
                strategy=order.strategy
            )
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"Paper order execution failed: {e}")
            return order
    
    def _execute_live_order(self, order: Order, max_retries: int = 3) -> Order:
        """Execute order on live exchange with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Set leverage for futures
                if self.futures:
                    try:
                        self.exchange.set_leverage(self.leverage, order.pair)
                    except Exception as e:
                        logger.warning(f"Could not set leverage: {e}")
                
                # Execute order
                if order.order_type == OrderType.MARKET:
                    if order.side == OrderSide.BUY:
                        result = self.exchange.create_market_buy_order(order.pair, order.amount)
                    else:
                        result = self.exchange.create_market_sell_order(order.pair, order.amount)
                else:
                    result = self.exchange.create_limit_order(
                        order.pair,
                        order.side.value,
                        order.amount,
                        order.price
                    )
                
                # Update order from result
                order.exchange_order_id = result.get('id')
                order.status = OrderStatus.FILLED if result.get('status') == 'closed' else OrderStatus.SUBMITTED
                order.filled_amount = result.get('filled', order.amount)
                order.average_price = result.get('average', result.get('price'))
                order.cost = result.get('cost', 0)
                order.fee = result.get('fee', {}).get('cost', 0)
                order.executed_at = datetime.now()
                
                logger.trade_executed(
                    pair=order.pair,
                    side=order.side.value,
                    amount=order.amount,
                    price=order.average_price,
                    order_id=order.exchange_order_id,
                    strategy=order.strategy
                )
                
                return order
                
            except ccxt.InsufficientFunds as e:
                order.status = OrderStatus.FAILED
                logger.trade_failed(order.pair, order.side.value, f"Insufficient funds: {e}", order.strategy)
                return order
                
            except ccxt.InvalidOrder as e:
                order.status = OrderStatus.FAILED
                logger.trade_failed(order.pair, order.side.value, f"Invalid order: {e}", order.strategy)
                return order
                
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                last_error = e
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                
            except Exception as e:
                order.status = OrderStatus.FAILED
                logger.error(f"Order execution failed: {e}")
                return order
        
        # Max retries exceeded
        order.status = OrderStatus.FAILED
        logger.trade_failed(order.pair, order.side.value, f"Max retries exceeded: {last_error}", order.strategy)
        return order
    
    def execute(
        self,
        pair: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: float = None,
        strategy: str = None
    ) -> Order:
        """
        Execute a trading order.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            strategy: Strategy name for logging
        
        Returns:
            Order object with execution details
        """
        # Create order
        order = Order(
            id=self._generate_order_id(),
            pair=pair,
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            amount=amount,
            price=price,
            strategy=strategy,
            is_paper=self.paper_mode
        )
        
        logger.info(f"Executing {'PAPER' if self.paper_mode else 'LIVE'} order: "
                   f"{side.upper()} {amount} {pair} @ {order_type}")
        
        # Execute
        if self.paper_mode:
            order = self._execute_paper_order(order)
        else:
            order = self._execute_live_order(order)
        
        # Save to database
        if order.status == OrderStatus.FILLED:
            self.db.save_trade(
                pair=order.pair,
                side=order.side.value,
                amount=order.filled_amount,
                price=order.average_price,
                order_id=order.exchange_order_id,
                cost=order.cost,
                fee=order.fee,
                strategy=order.strategy,
                exchange=self.exchange_type.value,
                is_paper=order.is_paper
            )
            self._executed_orders.append(order)
        
        return order
    
    def get_balance(self) -> Dict[str, float]:
        """Get current balance"""
        if self.paper_mode:
            return {
                'USDT': self._paper_balance,
                'positions': self._paper_positions.copy()
            }
        
        try:
            balance = self.exchange.fetch_balance()
            return {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'total': balance.get('total', {})
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {'USDT': 0, 'error': str(e)}
    
    def get_position(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get current position for a pair"""
        if self.paper_mode:
            base_asset = pair.split('/')[0]
            return {
                'amount': self._paper_positions.get(base_asset, 0),
                'pair': pair
            }
        
        try:
            if self.futures:
                positions = self.exchange.fetch_positions([pair])
                return positions[0] if positions else None
            else:
                balance = self.exchange.fetch_balance()
                base_asset = pair.split('/')[0]
                return {
                    'amount': balance.get(base_asset, {}).get('free', 0),
                    'pair': pair
                }
        except Exception as e:
            logger.error(f"Failed to fetch position for {pair}: {e}")
            return None
    
    def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel a pending order"""
        if self.paper_mode:
            logger.info(f"Paper order {order_id} cancelled")
            return True
        
        try:
            self.exchange.cancel_order(order_id, pair)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
```

### core\risk_manager.py
```py
"""
Risk Manager for Elite Trading System

Enforces position sizing, drawdown limits, and phase transitions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from config.settings import get_settings
from utils.logger import get_logger
from utils.database import get_database

logger = get_logger('risk_manager')


class RiskLevel(Enum):
    """Risk alert levels"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class TradingPhase(Enum):
    """Trading phases"""
    FUTURES_GROWTH = "futures_growth"  # 50 -> 2000 USDT
    SPOT_ELITE = "spot_elite"          # 2000+ USDT


@dataclass
class RiskStatus:
    """Current risk status snapshot"""
    level: RiskLevel
    current_drawdown: float
    max_drawdown_limit: float
    current_balance: float
    peak_balance: float
    daily_pnl: float
    open_positions: int
    phase: TradingPhase
    can_trade: bool
    message: str


class RiskManager:
    """
    Risk management system.
    
    Features:
    - Position sizing based on risk per trade
    - Maximum drawdown enforcement
    - Phase transition (futures -> spot at target balance)
    - Daily loss limits
    - Maximum position limits
    """
    
    def __init__(
        self,
        max_drawdown: float = None,
        risk_per_trade: float = None,
        max_positions: int = 3,
        daily_loss_limit: float = 0.10  # 10% daily loss limit
    ):
        self.settings = get_settings()
        self.db = get_database()
        
        self.max_drawdown = max_drawdown or self.settings.max_drawdown
        self.risk_per_trade = risk_per_trade or self.settings.risk_per_trade
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        
        self._peak_balance = self.settings.starting_balance
        self._daily_start_balance = self.settings.starting_balance
        self._open_positions: Dict[str, Dict] = {}
        self._last_daily_reset: Optional[datetime] = None
        self._trading_halted = False
        self._halt_reason = ""
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Formula: Position = (Balance * Risk%) / (Entry - StopLoss) / Entry
        
        Args:
            balance: Current account balance
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            leverage: Leverage multiplier
        
        Returns:
            Position size in base currency
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        # Calculate risk amount in USDT
        risk_amount = balance * self.risk_per_trade
        
        # Calculate price risk (distance to stop loss)
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        
        if price_risk == 0:
            return 0.0
        
        # Position size without leverage
        position_value = risk_amount / price_risk
        
        # Apply leverage
        position_value *= leverage
        
        # Convert to base currency amount
        position_size = position_value / entry_price
        
        logger.debug(
            f"Position sizing: balance={balance}, risk={risk_amount:.2f}, "
            f"price_risk={price_risk:.2%}, size={position_size:.6f}"
        )
        
        return position_size
    
    def calculate_position_size_simple(
        self,
        balance: float,
        leverage: int = 1
    ) -> float:
        """
        Simple position size based on risk per trade without stop loss.
        Uses a default 2% stop loss distance.
        
        Args:
            balance: Current account balance
            leverage: Leverage multiplier
        
        Returns:
            Maximum position value in USDT
        """
        # Risk amount
        risk_amount = balance * self.risk_per_trade
        
        # Assume 2% price movement for stop
        default_stop_distance = 0.02
        
        # Position value
        position_value = (risk_amount / default_stop_distance) * leverage
        
        # Cap at percentage of balance
        max_position = balance * 0.25 * leverage  # Max 25% per position
        
        return min(position_value, max_position)
    
    def update_balance(self, new_balance: float, source: str = None) -> RiskStatus:
        """
        Update balance and check risk levels.
        
        Args:
            new_balance: Current balance
            source: Source of balance update
        
        Returns:
            Current RiskStatus
        """
        # Update peak
        if new_balance > self._peak_balance:
            self._peak_balance = new_balance
        
        # Check for daily reset
        now = datetime.now()
        if self._last_daily_reset is None or now.date() > self._last_daily_reset.date():
            self._daily_start_balance = new_balance
            self._last_daily_reset = now
        
        # Calculate drawdown
        current_drawdown = (self._peak_balance - new_balance) / self._peak_balance if self._peak_balance > 0 else 0
        
        # Calculate daily PnL
        daily_pnl = (new_balance - self._daily_start_balance) / self._daily_start_balance if self._daily_start_balance > 0 else 0
        
        # Log balance update
        change = new_balance - (self.db.get_current_balance() or self.settings.starting_balance)
        logger.balance_update(new_balance, change, source)
        
        # Save to database
        self.db.save_balance(new_balance, change, source)
        
        # Determine phase
        phase = TradingPhase.SPOT_ELITE if new_balance >= self.settings.futures_target else TradingPhase.FUTURES_GROWTH
        
        # Determine risk level and check limits
        status = self._evaluate_risk(current_drawdown, daily_pnl, new_balance, phase)
        
        return status
    
    def _evaluate_risk(
        self,
        drawdown: float,
        daily_pnl: float,
        balance: float,
        phase: TradingPhase
    ) -> RiskStatus:
        """Evaluate current risk level and determine if trading should continue"""
        
        can_trade = True
        message = "Trading normal"
        level = RiskLevel.NORMAL
        
        # Check drawdown thresholds
        if drawdown >= self.max_drawdown:
            level = RiskLevel.CRITICAL
            can_trade = False
            message = f"Max drawdown exceeded: {drawdown:.1%} >= {self.max_drawdown:.1%}"
            self._trading_halted = True
            self._halt_reason = message
            logger.risk_alert("MAX_DRAWDOWN", drawdown, self.max_drawdown, "HALT_TRADING")
            
        elif drawdown >= self.max_drawdown * 0.8:
            level = RiskLevel.HIGH
            message = f"Approaching max drawdown: {drawdown:.1%}"
            logger.risk_alert("HIGH_DRAWDOWN", drawdown, self.max_drawdown * 0.8, "REDUCE_RISK")
            
        elif drawdown >= self.max_drawdown * 0.5:
            level = RiskLevel.ELEVATED
            message = f"Elevated drawdown: {drawdown:.1%}"
        
        # Check daily loss limit
        if daily_pnl <= -self.daily_loss_limit:
            level = RiskLevel.CRITICAL
            can_trade = False
            message = f"Daily loss limit hit: {daily_pnl:.1%}"
            logger.risk_alert("DAILY_LOSS_LIMIT", abs(daily_pnl), self.daily_loss_limit, "HALT_TRADING")
        
        # Check position limits
        if len(self._open_positions) >= self.max_positions:
            if level != RiskLevel.CRITICAL:
                level = RiskLevel.HIGH
            message = f"Max positions reached: {len(self._open_positions)}"
        
        return RiskStatus(
            level=level,
            current_drawdown=drawdown,
            max_drawdown_limit=self.max_drawdown,
            current_balance=balance,
            peak_balance=self._peak_balance,
            daily_pnl=daily_pnl,
            open_positions=len(self._open_positions),
            phase=phase,
            can_trade=can_trade,
            message=message
        )
    
    def can_open_position(self, pair: str, balance: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Returns:
            Tuple of (can_open, reason)
        """
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        if pair in self._open_positions:
            return False, f"Position already open for {pair}"
        
        if len(self._open_positions) >= self.max_positions:
            return False, "Maximum positions reached"
        
        # Check current risk status
        status = self.get_status(balance)
        if not status.can_trade:
            return False, status.message
        
        return True, "OK"
    
    def open_position(
        self,
        pair: str,
        size: float,
        entry_price: float,
        side: str,
        stop_loss: float = None
    ):
        """Record an opened position"""
        self._open_positions[pair] = {
            'size': size,
            'entry_price': entry_price,
            'side': side,
            'stop_loss': stop_loss,
            'opened_at': datetime.now()
        }
        logger.info(f"Position opened: {side.upper()} {size} {pair} @ {entry_price}")
    
    def close_position(self, pair: str, exit_price: float = None) -> Optional[Dict]:
        """Record a closed position and return PnL"""
        if pair not in self._open_positions:
            return None
        
        position = self._open_positions.pop(pair)
        
        if exit_price:
            if position['side'] == 'buy':
                pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
            pnl_value = position['size'] * position['entry_price'] * pnl
            
            logger.info(f"Position closed: {pair} @ {exit_price}, PnL: {pnl:.2%} ({pnl_value:+.2f} USDT)")
            
            return {
                **position,
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'pnl_value': pnl_value
            }
        
        return position
    
    def get_status(self, current_balance: float = None) -> RiskStatus:
        """Get current risk status"""
        if current_balance is None:
            current_balance = self.db.get_current_balance() or self.settings.starting_balance
        
        drawdown_data = self.db.calculate_drawdown()
        daily_pnl = (current_balance - self._daily_start_balance) / self._daily_start_balance if self._daily_start_balance > 0 else 0
        
        phase = TradingPhase.SPOT_ELITE if current_balance >= self.settings.futures_target else TradingPhase.FUTURES_GROWTH
        
        return self._evaluate_risk(
            drawdown_data.get('current_drawdown', 0),
            daily_pnl,
            current_balance,
            phase
        )
    
    def should_switch_to_spot(self, balance: float) -> bool:
        """Check if should switch from futures to spot mode"""
        return balance >= self.settings.futures_target
    
    def reset_daily(self, current_balance: float):
        """Manual reset of daily tracking"""
        self._daily_start_balance = current_balance
        self._last_daily_reset = datetime.now()
        logger.info(f"Daily tracking reset at balance: {current_balance}")
    
    def resume_trading(self):
        """Resume trading after halt"""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
```

### core\__init__.py
```py
"""
Elite Trading System - Core Module
"""
from core.executor import OrderExecutor, Order, OrderStatus
from core.risk_manager import RiskManager, RiskLevel
from core.engine import TradingEngine

__all__ = [
    'OrderExecutor', 'Order', 'OrderStatus',
    'RiskManager', 'RiskLevel',
    'TradingEngine'
]
```

### research\backtester.py
```py
"""
Vectorized Backtester for Elite Trading System

Uses Pandas for fast, vectorized strategy testing.
No loops over individual candles - all calculations done on arrays.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from research.strategy_generator import StrategyParams, StrategyType
from research.data_pipeline import get_data_pipeline
from utils.logger import get_logger
from utils.database import get_database

# Suppress ta library warnings
warnings.filterwarnings('ignore')

try:
    import ta
except ImportError:
    ta = None
    print("Warning: ta library not installed. Install with: pip install ta")

logger = get_logger('backtester')


@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    pair: str
    params: Dict[str, Any]
    
    # Performance metrics
    roi: float  # Return on Investment
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    
    # Additional info
    test_start: str
    test_end: str
    timeframe: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'pair': self.pair,
            'params': self.params,
            'roi': self.roi,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'profit_factor': self.profit_factor,
            'test_start': self.test_start,
            'test_end': self.test_end,
            'timeframe': self.timeframe
        }


class Backtester:
    """
    Vectorized backtesting engine.
    
    All indicator calculations and signal generation are done using
    Pandas vectorized operations for maximum performance.
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005   # 0.05% slippage
    ):
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.pipeline = get_data_pipeline()
        self.db = get_database()
    
    def _add_indicators(self, df: pd.DataFrame, strategy: StrategyParams) -> pd.DataFrame:
        """Add technical indicators to dataframe based on strategy type"""
        if ta is None:
            raise ImportError("ta library required for indicators")
        
        df = df.copy()
        params = strategy.params
        
        if strategy.strategy_type == StrategyType.SMA_CROSSOVER:
            df['ma_short'] = ta.trend.sma_indicator(df['close'], window=params['short_period'])
            df['ma_long'] = ta.trend.sma_indicator(df['close'], window=params['long_period'])
        
        elif strategy.strategy_type == StrategyType.EMA_CROSSOVER:
            df['ma_short'] = ta.trend.ema_indicator(df['close'], window=params['short_period'])
            df['ma_long'] = ta.trend.ema_indicator(df['close'], window=params['long_period'])
        
        elif strategy.strategy_type == StrategyType.RSI_REVERSAL:
            df['rsi'] = ta.momentum.rsi(df['close'], window=params['period'])
        
        elif strategy.strategy_type == StrategyType.MACD_SIGNAL:
            macd = ta.trend.MACD(
                df['close'],
                window_slow=params['slow_period'],
                window_fast=params['fast_period'],
                window_sign=params['signal_period']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
        
        elif strategy.strategy_type == StrategyType.BOLLINGER_BANDS:
            bb = ta.volatility.BollingerBands(
                df['close'],
                window=params['period'],
                window_dev=params['std_dev']
            )
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
        
        elif strategy.strategy_type == StrategyType.COMBINED:
            df['ma_short'] = ta.trend.sma_indicator(df['close'], window=params['ma_short'])
            df['ma_long'] = ta.trend.sma_indicator(df['close'], window=params['ma_long'])
            df['rsi'] = ta.momentum.rsi(df['close'], window=params['rsi_period'])
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame, strategy: StrategyParams) -> pd.DataFrame:
        """Generate buy/sell signals based on strategy"""
        df = df.copy()
        params = strategy.params
        
        # Initialize signal column: 1 = buy, -1 = sell, 0 = hold
        df['signal'] = 0
        
        if strategy.strategy_type in (StrategyType.SMA_CROSSOVER, StrategyType.EMA_CROSSOVER):
            # Buy when short MA crosses above long MA
            # Sell when short MA crosses below long MA
            df['signal'] = np.where(
                (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1)),
                1,
                np.where(
                    (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1)),
                    -1,
                    0
                )
            )
        
        elif strategy.strategy_type == StrategyType.RSI_REVERSAL:
            oversold = params['oversold']
            overbought = params['overbought']
            
            # Buy when RSI crosses above oversold
            # Sell when RSI crosses below overbought
            df['signal'] = np.where(
                (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold),
                1,
                np.where(
                    (df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought),
                    -1,
                    0
                )
            )
        
        elif strategy.strategy_type == StrategyType.MACD_SIGNAL:
            # Buy when MACD crosses above signal line
            # Sell when MACD crosses below signal line
            df['signal'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                1,
                np.where(
                    (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                    -1,
                    0
                )
            )
        
        elif strategy.strategy_type == StrategyType.BOLLINGER_BANDS:
            # Buy when price touches lower band
            # Sell when price touches upper band
            df['signal'] = np.where(
                (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1)),
                1,
                np.where(
                    (df['close'] >= df['bb_upper']) & (df['close'].shift(1) < df['bb_upper'].shift(1)),
                    -1,
                    0
                )
            )
        
        elif strategy.strategy_type == StrategyType.COMBINED:
            # MA crossover with RSI filter
            oversold = params['rsi_oversold']
            overbought = params['rsi_overbought']
            
            # Buy: MA crossover + RSI not overbought
            # Sell: MA crossunder + RSI not oversold
            ma_cross_up = (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1))
            ma_cross_down = (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1))
            
            df['signal'] = np.where(
                ma_cross_up & (df['rsi'] < overbought),
                1,
                np.where(
                    ma_cross_down & (df['rsi'] > oversold),
                    -1,
                    0
                )
            )
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade returns with position tracking"""
        df = df.copy()
        
        # Position: 1 = long, 0 = flat
        df['position'] = 0
        position = 0
        
        for i in range(len(df)):
            if df.iloc[i]['signal'] == 1:
                position = 1
            elif df.iloc[i]['signal'] == -1:
                position = 0
            df.iloc[i, df.columns.get_loc('position')] = position
        
        # Daily returns
        df['returns'] = df['close'].pct_change()
        
        # Strategy returns (only when in position)
        df['strategy_returns'] = df['position'].shift(1) * df['returns']
        
        # Subtract costs on signal days
        df.loc[df['signal'] != 0, 'strategy_returns'] -= (self.commission + self.slippage)
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        return df
    
    def _calculate_metrics(self, df: pd.DataFrame, strategy: StrategyParams, pair: str) -> BacktestResult:
        """Calculate performance metrics from backtest results"""
        
        # Filter to valid data
        df = df.dropna()
        
        if len(df) < 2:
            return BacktestResult(
                strategy_name=strategy.name,
                pair=pair,
                params=strategy.params,
                roi=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
                test_start='',
                test_end='',
                timeframe=''
            )
        
        # ROI
        final_value = df['cumulative_returns'].iloc[-1]
        roi = final_value - 1  # -1 because we start at 1
        
        # Sharpe Ratio (annualized, assuming hourly data = 24*365 periods/year)
        if df['strategy_returns'].std() != 0:
            sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0.0
        
        # Max Drawdown
        rolling_max = df['cumulative_returns'].cummax()
        drawdown = (rolling_max - df['cumulative_returns']) / rolling_max
        max_drawdown = drawdown.max()
        
        # Trade analysis
        trades = df[df['signal'] != 0].copy()
        total_trades = len(trades)
        
        if total_trades > 0:
            # Calculate profit for each trade
            trades['trade_profit'] = trades['strategy_returns']
            winning_trades = len(trades[trades['trade_profit'] > 0])
            win_rate = winning_trades / total_trades
            
            # Profit factor
            gross_profit = trades[trades['trade_profit'] > 0]['trade_profit'].sum()
            gross_loss = abs(trades[trades['trade_profit'] < 0]['trade_profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # Date range
        test_start = datetime.fromtimestamp(df['timestamp'].iloc[0] / 1000).isoformat()
        test_end = datetime.fromtimestamp(df['timestamp'].iloc[-1] / 1000).isoformat()
        
        return BacktestResult(
            strategy_name=strategy.name,
            pair=pair,
            params=strategy.params,
            roi=roi,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=profit_factor if profit_factor != float('inf') else 999.99,
            test_start=test_start,
            test_end=test_end,
            timeframe='1h'
        )
    
    def run(
        self,
        pair: str,
        strategy: StrategyParams,
        timeframe: str = '1h',
        limit: int = 1000,
        save_to_db: bool = True
    ) -> Optional[BacktestResult]:
        """
        Run a single backtest.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            strategy: Strategy parameters to test
            timeframe: Candle timeframe
            limit: Number of candles to use
            save_to_db: Whether to save results to database
        
        Returns:
            BacktestResult or None if failed
        """
        try:
            # Fetch data
            df = self.pipeline.fetch_ohlcv(pair, timeframe, limit)
            
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {pair} ({len(df)} candles)")
                return None
            
            # Add indicators
            df = self._add_indicators(df, strategy)
            
            # Generate signals
            df = self._generate_signals(df, strategy)
            
            # Calculate returns
            df = self._calculate_returns(df)
            
            # Calculate metrics
            result = self._calculate_metrics(df, strategy, pair)
            
            # Save to database
            if save_to_db:
                self.db.save_strategy_result(
                    name=result.strategy_name,
                    pair=result.pair,
                    params=result.params,
                    roi=result.roi,
                    sharpe_ratio=result.sharpe_ratio,
                    max_drawdown=result.max_drawdown,
                    win_rate=result.win_rate,
                    total_trades=result.total_trades,
                    profit_factor=result.profit_factor,
                    timeframe=result.timeframe,
                    test_start_date=result.test_start,
                    test_end_date=result.test_end
                )
            
            logger.strategy_result(
                strategy=result.strategy_name,
                pair=result.pair,
                roi=result.roi,
                sharpe=result.sharpe_ratio,
                drawdown=result.max_drawdown,
                win_rate=result.win_rate
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed for {strategy.name} on {pair}: {e}")
            return None
    
    def run_batch(
        self,
        pairs: List[str],
        strategies: List[StrategyParams],
        timeframe: str = '1h',
        limit: int = 1000,
        parallel: bool = False
    ) -> List[BacktestResult]:
        """
        Run multiple backtests.
        
        Args:
            pairs: List of trading pairs
            strategies: List of strategy parameters
            timeframe: Candle timeframe
            limit: Number of candles to use
            parallel: Use parallel processing (experimental)
        
        Returns:
            List of BacktestResult
        """
        results = []
        total = len(pairs) * len(strategies)
        completed = 0
        
        logger.info(f"Starting batch backtest: {len(pairs)} pairs x {len(strategies)} strategies = {total} tests")
        
        for pair in pairs:
            for strategy in strategies:
                result = self.run(pair, strategy, timeframe, limit)
                if result:
                    results.append(result)
                
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
        
        logger.info(f"Batch complete: {len(results)}/{total} successful")
        return results
```

### research\data_pipeline.py
```py
"""
Data Pipeline for Elite Trading System

Fetches and caches OHLCV data from exchanges.
Supports multiple pairs and timeframes.
"""

import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import ccxt

from config.settings import get_settings, get_public_exchange, ExchangeType
from utils.logger import get_logger

logger = get_logger('data_pipeline')


@dataclass
class OHLCVData:
    """Container for OHLCV data with metadata"""
    pair: str
    timeframe: str
    data: pd.DataFrame
    exchange: str
    fetched_at: datetime
    
    @property
    def is_stale(self) -> bool:
        """Check if data is older than 1 hour"""
        return datetime.now() - self.fetched_at > timedelta(hours=1)


class DataPipeline:
    """
    Fetches and manages OHLCV data from exchanges.
    Implements caching to reduce API calls.
    """
    
    def __init__(self, cache_dir: Path = None):
        self.settings = get_settings()
        self.cache_dir = cache_dir or self.settings.data_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, OHLCVData] = {}
        
        # Exchange instance (public, no auth needed)
        self._exchange: Optional[ccxt.Exchange] = None
    
    @property
    def exchange(self) -> ccxt.Exchange:
        """Get or create exchange instance"""
        if self._exchange is None:
            self._exchange = get_public_exchange(ExchangeType.BINANCE)
        return self._exchange
    
    def _cache_key(self, pair: str, timeframe: str) -> str:
        """Generate cache key for pair/timeframe combo"""
        return f"{pair.replace('/', '_')}_{timeframe}"
    
    def _cache_path(self, pair: str, timeframe: str) -> Path:
        """Get file path for cached data"""
        return self.cache_dir / f"{self._cache_key(pair, timeframe)}.parquet"
    
    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = '1h',
        limit: int = 1000,
        since: datetime = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a trading pair.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max varies by exchange)
            since: Start time for data fetch
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = self._cache_key(pair, timeframe)
        
        # Check memory cache first
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if not cached.is_stale:
                logger.debug(f"Using memory cache for {pair} {timeframe}")
                return cached.data.copy()
        
        # Check file cache
        cache_path = self._cache_path(pair, timeframe)
        if use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                # Check if recent enough
                last_ts = df['timestamp'].max()
                if datetime.fromtimestamp(last_ts/1000) > datetime.now() - timedelta(hours=1):
                    logger.debug(f"Using file cache for {pair} {timeframe}")
                    self._cache[cache_key] = OHLCVData(
                        pair=pair, timeframe=timeframe, data=df,
                        exchange='binance', fetched_at=datetime.now()
                    )
                    return df.copy()
            except Exception as e:
                logger.warning(f"Failed to read cache file: {e}")
        
        # Fetch from exchange
        logger.info(f"Fetching OHLCV data for {pair} {timeframe} (limit={limit})")
        
        try:
            since_ts = int(since.timestamp() * 1000) if since else None
            
            # CCXT rate limit handling
            raw_data = self.exchange.fetch_ohlcv(
                pair, 
                timeframe=timeframe, 
                limit=limit,
                since=since_ts
            )
            
            if not raw_data:
                logger.warning(f"No data returned for {pair} {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            df = pd.DataFrame(raw_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Cache to file
            try:
                df.to_parquet(cache_path, index=False)
            except Exception as e:
                logger.warning(f"Failed to cache to file: {e}")
            
            # Cache to memory
            self._cache[cache_key] = OHLCVData(
                pair=pair, timeframe=timeframe, data=df,
                exchange='binance', fetched_at=datetime.now()
            )
            
            logger.info(f"Fetched {len(df)} candles for {pair} {timeframe}")
            return df
            
        except ccxt.RateLimitExceeded:
            logger.warning(f"Rate limit exceeded for {pair}, waiting...")
            time.sleep(60)
            return self.fetch_ohlcv(pair, timeframe, limit, since, use_cache)
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching {pair}: {e}")
            raise
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {pair}: {e}")
            raise
    
    def fetch_multiple_pairs(
        self,
        pairs: List[str],
        timeframe: str = '1h',
        limit: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple pairs.
        
        Args:
            pairs: List of trading pairs
            timeframe: Candle timeframe
            limit: Number of candles per pair
        
        Returns:
            Dictionary mapping pair to DataFrame
        """
        result = {}
        
        for pair in pairs:
            try:
                result[pair] = self.fetch_ohlcv(pair, timeframe, limit)
                # Small delay to avoid rate limits
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")
                continue
        
        return result
    
    def get_historical_data(
        self,
        pair: str,
        timeframe: str = '1d',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch extended historical data by making multiple requests.
        
        Args:
            pair: Trading pair
            timeframe: Candle timeframe
            days: Number of days of history to fetch
        
        Returns:
            DataFrame with historical OHLCV data
        """
        # Calculate how many candles we need
        tf_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes_per_candle = tf_minutes.get(timeframe, 60)
        candles_needed = (days * 24 * 60) // minutes_per_candle
        
        # Fetch in batches (1000 candles per request is typical limit)
        all_data = []
        since = datetime.now() - timedelta(days=days)
        batch_size = 1000
        
        logger.info(f"Fetching {candles_needed} candles for {pair} {timeframe} ({days} days)")
        
        while len(all_data) < candles_needed:
            try:
                df = self.fetch_ohlcv(pair, timeframe, batch_size, since, use_cache=False)
                
                if df.empty:
                    break
                
                all_data.append(df)
                
                # Move since to after last candle
                last_ts = df['timestamp'].max()
                since = datetime.fromtimestamp(last_ts / 1000) + timedelta(minutes=minutes_per_candle)
                
                # Small delay
                time.sleep(0.2)
                
                if len(df) < batch_size:
                    break  # No more data available
                    
            except Exception as e:
                logger.error(f"Error in batch fetch: {e}")
                break
        
        if not all_data:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"Fetched total {len(result)} candles for {pair}")
        return result
    
    def clear_cache(self, pair: str = None, timeframe: str = None):
        """Clear cached data"""
        if pair and timeframe:
            key = self._cache_key(pair, timeframe)
            self._cache.pop(key, None)
            path = self._cache_path(pair, timeframe)
            if path.exists():
                path.unlink()
        else:
            self._cache.clear()
            for f in self.cache_dir.glob('*.parquet'):
                f.unlink()
        
        logger.info("Cache cleared")


# Singleton instance
_pipeline: Optional[DataPipeline] = None


def get_data_pipeline() -> DataPipeline:
    """Get or create data pipeline singleton"""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline
```

### research\priority_list.py
```py
"""
Priority List Manager for Elite Trading System

Ranks strategies and outputs priority list for the trading engine.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from config.settings import get_settings
from utils.database import get_database
from utils.logger import get_logger

logger = get_logger('priority_list')


@dataclass
class PriorityEntry:
    """Single entry in the priority list"""
    priority: int
    pair: str
    strategy: str
    params: Dict[str, Any]
    expected_roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    confidence_score: float  # Composite score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'priority': self.priority,
            'pair': self.pair,
            'strategy': self.strategy,
            'params': self.params,
            'expected_roi': round(self.expected_roi, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'win_rate': round(self.win_rate, 4),
            'confidence_score': round(self.confidence_score, 2)
        }


class PriorityListManager:
    """
    Manages the priority list of trading strategies.
    Ranks by composite score combining multiple metrics.
    """
    
    def __init__(self, output_path: Path = None):
        self.settings = get_settings()
        self.db = get_database()
        self.output_path = output_path or self.settings.priority_list_path
    
    def _calculate_confidence_score(
        self,
        roi: float,
        sharpe: float,
        drawdown: float,
        win_rate: float,
        total_trades: int
    ) -> float:
        """
        Calculate composite confidence score.
        
        Weights:
        - Sharpe ratio: 40% (risk-adjusted returns)
        - Win rate: 25%
        - ROI: 20%
        - Drawdown: 15% (penalize high drawdown)
        """
        # Normalize each metric
        sharpe_score = min(max(sharpe, 0), 3) / 3 * 100  # Cap at 3
        win_rate_score = win_rate * 100
        roi_score = min(max(roi, -1), 2) / 2 * 100 + 50  # Normalize -100% to +200%
        drawdown_score = (1 - min(drawdown, 1)) * 100  # Lower is better
        
        # Trade count penalty for low sample size
        trade_penalty = 1.0 if total_trades >= 30 else (total_trades / 30)
        
        # Weighted average
        score = (
            sharpe_score * 0.40 +
            win_rate_score * 0.25 +
            roi_score * 0.20 +
            drawdown_score * 0.15
        ) * trade_penalty
        
        return score
    
    def generate(
        self,
        limit: int = 10,
        min_sharpe: float = 0.5,
        min_trades: int = 10,
        pairs: List[str] = None
    ) -> List[PriorityEntry]:
        """
        Generate priority list from database.
        
        Args:
            limit: Maximum number of entries
            min_sharpe: Minimum Sharpe ratio filter
            min_trades: Minimum number of trades filter
            pairs: Filter by specific pairs
        
        Returns:
            List of PriorityEntry sorted by confidence score
        """
        logger.info(f"Generating priority list (limit={limit}, min_sharpe={min_sharpe})")
        
        # Get all strategies from DB
        strategies = self.db.get_top_strategies(limit=limit * 3, min_sharpe=min_sharpe)
        
        if not strategies:
            logger.warning("No strategies found in database")
            return []
        
        # Calculate confidence scores and filter
        scored_strategies = []
        
        for s in strategies:
            # Apply pair filter
            if pairs and s['pair'] not in pairs:
                continue
            
            # Apply trade count filter
            if s['total_trades'] and s['total_trades'] < min_trades:
                continue
            
            confidence = self._calculate_confidence_score(
                roi=s['roi'] or 0,
                sharpe=s['sharpe_ratio'] or 0,
                drawdown=s['max_drawdown'] or 0,
                win_rate=s['win_rate'] or 0,
                total_trades=s['total_trades'] or 0
            )
            
            scored_strategies.append({
                **s,
                'confidence_score': confidence
            })
        
        # Sort by confidence score
        scored_strategies.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Create priority entries
        entries = []
        for i, s in enumerate(scored_strategies[:limit]):
            params = json.loads(s['params_json']) if isinstance(s['params_json'], str) else s['params_json']
            
            entry = PriorityEntry(
                priority=i + 1,
                pair=s['pair'],
                strategy=s['name'],
                params=params,
                expected_roi=s['roi'] or 0,
                sharpe_ratio=s['sharpe_ratio'] or 0,
                max_drawdown=s['max_drawdown'] or 0,
                win_rate=s['win_rate'] or 0,
                confidence_score=s['confidence_score']
            )
            entries.append(entry)
        
        logger.info(f"Generated priority list with {len(entries)} entries")
        return entries
    
    def save(self, entries: List[PriorityEntry] = None) -> Path:
        """
        Save priority list to JSON file.
        
        Args:
            entries: List of entries to save (generates if not provided)
        
        Returns:
            Path to saved file
        """
        if entries is None:
            entries = self.generate()
        
        data = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'count': len(entries),
            'strategies': [e.to_dict() for e in entries]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved priority list to {self.output_path}")
        return self.output_path
    
    def load(self) -> List[PriorityEntry]:
        """Load priority list from JSON file"""
        if not self.output_path.exists():
            logger.warning(f"Priority list not found: {self.output_path}")
            return []
        
        with open(self.output_path, 'r') as f:
            data = json.load(f)
        
        entries = []
        for s in data.get('strategies', []):
            entries.append(PriorityEntry(
                priority=s['priority'],
                pair=s['pair'],
                strategy=s['strategy'],
                params=s['params'],
                expected_roi=s['expected_roi'],
                sharpe_ratio=s['sharpe_ratio'],
                max_drawdown=s['max_drawdown'],
                win_rate=s['win_rate'],
                confidence_score=s['confidence_score']
            ))
        
        logger.info(f"Loaded {len(entries)} entries from priority list")
        return entries
    
    def get_top(self, n: int = 3) -> List[PriorityEntry]:
        """Get top N strategies from current priority list"""
        entries = self.load()
        return entries[:n]
    
    def get_by_pair(self, pair: str) -> Optional[PriorityEntry]:
        """Get best strategy for a specific pair"""
        entries = self.load()
        for e in entries:
            if e.pair == pair:
                return e
        return None
    
    def print_summary(self, entries: List[PriorityEntry] = None):
        """Print priority list summary to console"""
        if entries is None:
            entries = self.load()
        
        if not entries:
            print("Priority list is empty")
            return
        
        print("\n" + "="*80)
        print("PRIORITY LIST SUMMARY")
        print("="*80)
        print(f"{'#':<3} {'Pair':<12} {'Strategy':<25} {'ROI':<10} {'Sharpe':<8} {'Win%':<8} {'Score':<8}")
        print("-"*80)
        
        for e in entries:
            print(f"{e.priority:<3} {e.pair:<12} {e.strategy:<25} "
                  f"{e.expected_roi*100:>7.2f}% {e.sharpe_ratio:>7.2f} "
                  f"{e.win_rate*100:>6.1f}% {e.confidence_score:>7.2f}")
        
        print("="*80 + "\n")
```

### research\strategy_generator.py
```py
"""
Strategy Generator for Elite Trading System

Auto-generates trading strategy parameter combinations.
Supports multiple indicator types with configurable ranges.
"""

from enum import Enum
from typing import List, Dict, Any, Generator, Tuple
from dataclasses import dataclass, field
from itertools import product

from utils.logger import get_logger

logger = get_logger('strategy_generator')


class StrategyType(Enum):
    """Available strategy types"""
    SMA_CROSSOVER = "sma_crossover"
    EMA_CROSSOVER = "ema_crossover"
    RSI_REVERSAL = "rsi_reversal"
    MACD_SIGNAL = "macd_signal"
    BOLLINGER_BANDS = "bollinger_bands"
    COMBINED = "combined"


@dataclass
class StrategyParams:
    """Container for strategy parameters"""
    strategy_type: StrategyType
    name: str
    params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.strategy_type.value,
            'name': self.name,
            'params': self.params
        }


@dataclass
class IndicatorConfig:
    """Configuration for indicator parameter ranges"""
    
    # SMA/EMA parameters
    ma_short_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    ma_long_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # RSI parameters
    rsi_periods: List[int] = field(default_factory=lambda: [14])
    rsi_oversold: List[int] = field(default_factory=lambda: [20, 30])
    rsi_overbought: List[int] = field(default_factory=lambda: [70, 80])
    
    # MACD parameters
    macd_fast: List[int] = field(default_factory=lambda: [12])
    macd_slow: List[int] = field(default_factory=lambda: [26])
    macd_signal: List[int] = field(default_factory=lambda: [9])
    
    # Bollinger Bands parameters
    bb_periods: List[int] = field(default_factory=lambda: [20])
    bb_std: List[float] = field(default_factory=lambda: [2.0, 2.5])


class StrategyGenerator:
    """
    Generates all possible strategy combinations from indicator configs.
    Uses itertools.product for efficient permutation generation.
    """
    
    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self._generated_count = 0
    
    def generate_sma_crossover(self) -> Generator[StrategyParams, None, None]:
        """Generate SMA crossover strategy variants"""
        for short, long in product(self.config.ma_short_periods, self.config.ma_long_periods):
            if short >= long:
                continue
            
            name = f"SMA_{short}_{long}"
            yield StrategyParams(
                strategy_type=StrategyType.SMA_CROSSOVER,
                name=name,
                params={
                    'short_period': short,
                    'long_period': long
                }
            )
    
    def generate_ema_crossover(self) -> Generator[StrategyParams, None, None]:
        """Generate EMA crossover strategy variants"""
        for short, long in product(self.config.ma_short_periods, self.config.ma_long_periods):
            if short >= long:
                continue
            
            name = f"EMA_{short}_{long}"
            yield StrategyParams(
                strategy_type=StrategyType.EMA_CROSSOVER,
                name=name,
                params={
                    'short_period': short,
                    'long_period': long
                }
            )
    
    def generate_rsi_reversal(self) -> Generator[StrategyParams, None, None]:
        """Generate RSI reversal strategy variants"""
        for period, oversold, overbought in product(
            self.config.rsi_periods,
            self.config.rsi_oversold,
            self.config.rsi_overbought
        ):
            name = f"RSI_{period}_{oversold}_{overbought}"
            yield StrategyParams(
                strategy_type=StrategyType.RSI_REVERSAL,
                name=name,
                params={
                    'period': period,
                    'oversold': oversold,
                    'overbought': overbought
                }
            )
    
    def generate_macd_signal(self) -> Generator[StrategyParams, None, None]:
        """Generate MACD signal strategy variants"""
        for fast, slow, signal in product(
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        ):
            if fast >= slow:
                continue
            
            name = f"MACD_{fast}_{slow}_{signal}"
            yield StrategyParams(
                strategy_type=StrategyType.MACD_SIGNAL,
                name=name,
                params={
                    'fast_period': fast,
                    'slow_period': slow,
                    'signal_period': signal
                }
            )
    
    def generate_bollinger_bands(self) -> Generator[StrategyParams, None, None]:
        """Generate Bollinger Bands strategy variants"""
        for period, std in product(self.config.bb_periods, self.config.bb_std):
            name = f"BB_{period}_{std}"
            yield StrategyParams(
                strategy_type=StrategyType.BOLLINGER_BANDS,
                name=name,
                params={
                    'period': period,
                    'std_dev': std
                }
            )
    
    def generate_combined(self) -> Generator[StrategyParams, None, None]:
        """Generate combined strategies (MA + RSI filter)"""
        for short, long in product(self.config.ma_short_periods, self.config.ma_long_periods):
            if short >= long:
                continue
            
            for rsi_period, oversold, overbought in product(
                self.config.rsi_periods,
                self.config.rsi_oversold,
                self.config.rsi_overbought
            ):
                name = f"COMBINED_SMA_{short}_{long}_RSI_{rsi_period}"
                yield StrategyParams(
                    strategy_type=StrategyType.COMBINED,
                    name=name,
                    params={
                        'ma_short': short,
                        'ma_long': long,
                        'rsi_period': rsi_period,
                        'rsi_oversold': oversold,
                        'rsi_overbought': overbought
                    }
                )
    
    def generate_all(
        self,
        strategy_types: List[StrategyType] = None,
        limit: int = None
    ) -> Generator[StrategyParams, None, None]:
        """
        Generate all strategy combinations.
        
        Args:
            strategy_types: List of strategy types to include (None = all)
            limit: Maximum number of strategies to generate
        
        Yields:
            StrategyParams for each combination
        """
        generators = {
            StrategyType.SMA_CROSSOVER: self.generate_sma_crossover,
            StrategyType.EMA_CROSSOVER: self.generate_ema_crossover,
            StrategyType.RSI_REVERSAL: self.generate_rsi_reversal,
            StrategyType.MACD_SIGNAL: self.generate_macd_signal,
            StrategyType.BOLLINGER_BANDS: self.generate_bollinger_bands,
            StrategyType.COMBINED: self.generate_combined,
        }
        
        if strategy_types is None:
            strategy_types = list(StrategyType)
        
        self._generated_count = 0
        
        for st in strategy_types:
            if st in generators:
                for params in generators[st]():
                    self._generated_count += 1
                    
                    if limit and self._generated_count > limit:
                        return
                    
                    yield params
        
        logger.info(f"Generated {self._generated_count} strategy combinations")
    
    def get_strategy_count(self, strategy_types: List[StrategyType] = None) -> int:
        """Calculate total number of strategy combinations without generating"""
        count = 0
        
        # Count valid MA combinations
        ma_combos = sum(1 for s, l in product(
            self.config.ma_short_periods, 
            self.config.ma_long_periods
        ) if s < l)
        
        rsi_combos = len(self.config.rsi_periods) * len(self.config.rsi_oversold) * len(self.config.rsi_overbought)
        macd_combos = sum(1 for f, s, sig in product(
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        ) if f < s)
        bb_combos = len(self.config.bb_periods) * len(self.config.bb_std)
        combined_combos = ma_combos * rsi_combos
        
        counts = {
            StrategyType.SMA_CROSSOVER: ma_combos,
            StrategyType.EMA_CROSSOVER: ma_combos,
            StrategyType.RSI_REVERSAL: rsi_combos,
            StrategyType.MACD_SIGNAL: macd_combos,
            StrategyType.BOLLINGER_BANDS: bb_combos,
            StrategyType.COMBINED: combined_combos,
        }
        
        for st in (strategy_types or list(StrategyType)):
            count += counts.get(st, 0)
        
        return count
    
    @staticmethod
    def get_quick_test_config() -> 'IndicatorConfig':
        """Get a minimal config for quick testing"""
        return IndicatorConfig(
            ma_short_periods=[10],
            ma_long_periods=[50],
            rsi_periods=[14],
            rsi_oversold=[30],
            rsi_overbought=[70],
            macd_fast=[12],
            macd_slow=[26],
            macd_signal=[9],
            bb_periods=[20],
            bb_std=[2.0]
        )
    
    @staticmethod
    def get_full_config() -> 'IndicatorConfig':
        """Get comprehensive config for thorough testing"""
        return IndicatorConfig(
            ma_short_periods=[5, 10, 15, 20],
            ma_long_periods=[30, 50, 100, 200],
            rsi_periods=[7, 14, 21],
            rsi_oversold=[20, 25, 30],
            rsi_overbought=[70, 75, 80],
            macd_fast=[8, 12],
            macd_slow=[21, 26],
            macd_signal=[9],
            bb_periods=[15, 20, 25],
            bb_std=[1.5, 2.0, 2.5]
        )
```

### research\__init__.py
```py
"""
Elite Trading System - Research Module
"""
from research.data_pipeline import DataPipeline, get_data_pipeline
from research.strategy_generator import StrategyGenerator, StrategyType
from research.backtester import Backtester, BacktestResult
from research.priority_list import PriorityListManager

__all__ = [
    'DataPipeline', 'get_data_pipeline',
    'StrategyGenerator', 'StrategyType',
    'Backtester', 'BacktestResult',
    'PriorityListManager'
]
```

### tests\test_backtester.py
```py
"""
Tests for the backtesting system
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_strategy_generator():
    """Test strategy generation"""
    from research.strategy_generator import StrategyGenerator, StrategyType, IndicatorConfig
    
    # Test with minimal config
    config = IndicatorConfig(
        ma_short_periods=[10],
        ma_long_periods=[50],
        rsi_periods=[14],
        rsi_oversold=[30],
        rsi_overbought=[70]
    )
    
    generator = StrategyGenerator(config)
    
    # Count strategies
    count = generator.get_strategy_count()
    assert count > 0, "Should generate at least some strategies"
    
    # Generate and verify
    strategies = list(generator.generate_all(limit=10))
    assert len(strategies) <= 10, "Should respect limit"
    
    for s in strategies:
        assert s.name, "Strategy should have a name"
        assert s.params, "Strategy should have parameters"
    
    print(f"✅ Strategy generator test passed ({len(strategies)} strategies)")


def test_strategy_generator_types():
    """Test generating specific strategy types"""
    from research.strategy_generator import StrategyGenerator, StrategyType
    
    generator = StrategyGenerator(StrategyGenerator.get_quick_test_config())
    
    # Test SMA only
    sma_strategies = list(generator.generate_sma_crossover())
    assert all(s.strategy_type == StrategyType.SMA_CROSSOVER for s in sma_strategies)
    
    # Test RSI only
    rsi_strategies = list(generator.generate_rsi_reversal())
    assert all(s.strategy_type == StrategyType.RSI_REVERSAL for s in rsi_strategies)
    
    print("✅ Strategy types test passed")


def test_backtest_result_dataclass():
    """Test BacktestResult dataclass"""
    from research.backtester import BacktestResult
    
    result = BacktestResult(
        strategy_name='SMA_10_50',
        pair='BTC/USDT',
        params={'short_period': 10, 'long_period': 50},
        roi=0.15,
        sharpe_ratio=1.5,
        max_drawdown=0.10,
        win_rate=0.60,
        total_trades=50,
        profit_factor=1.8,
        test_start='2024-01-01',
        test_end='2024-12-31',
        timeframe='1h'
    )
    
    # Test to_dict
    d = result.to_dict()
    assert d['strategy_name'] == 'SMA_10_50'
    assert d['roi'] == 0.15
    assert d['pair'] == 'BTC/USDT'
    
    print("✅ BacktestResult test passed")


def test_priority_list_manager():
    """Test priority list generation and loading"""
    import tempfile
    from pathlib import Path
    from research.priority_list import PriorityListManager, PriorityEntry
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'priority_list.json'
        
        manager = PriorityListManager(output_path)
        
        # Create manual entries
        entries = [
            PriorityEntry(
                priority=1,
                pair='BTC/USDT',
                strategy='SMA_10_50',
                params={'short_period': 10, 'long_period': 50},
                expected_roi=0.15,
                sharpe_ratio=1.8,
                max_drawdown=0.08,
                win_rate=0.62,
                confidence_score=85.5
            ),
            PriorityEntry(
                priority=2,
                pair='ETH/USDT',
                strategy='RSI_14_30_70',
                params={'period': 14, 'oversold': 30, 'overbought': 70},
                expected_roi=0.12,
                sharpe_ratio=1.5,
                max_drawdown=0.10,
                win_rate=0.58,
                confidence_score=78.2
            )
        ]
        
        # Save
        manager.save(entries)
        assert output_path.exists(), "Priority list file should be created"
        
        # Load
        loaded = manager.load()
        assert len(loaded) == 2, "Should load 2 entries"
        assert loaded[0].pair == 'BTC/USDT'
        assert loaded[0].priority == 1
        
        # Get top
        top = manager.get_top(1)
        assert len(top) == 1
        assert top[0].strategy == 'SMA_10_50'
        
        # Get by pair
        eth = manager.get_by_pair('ETH/USDT')
        assert eth is not None
        assert eth.strategy == 'RSI_14_30_70'
        
        print("✅ Priority list manager test passed")


def test_confidence_score_calculation():
    """Test the confidence score calculation"""
    from research.priority_list import PriorityListManager
    
    manager = PriorityListManager()
    
    # High quality strategy
    high_score = manager._calculate_confidence_score(
        roi=0.20,      # 20% ROI
        sharpe=2.0,    # Good Sharpe
        drawdown=0.05, # Low drawdown
        win_rate=0.65, # 65% win rate
        total_trades=100
    )
    
    # Low quality strategy
    low_score = manager._calculate_confidence_score(
        roi=0.02,      # 2% ROI
        sharpe=0.5,    # Poor Sharpe
        drawdown=0.25, # High drawdown
        win_rate=0.45, # 45% win rate
        total_trades=100
    )
    
    assert high_score > low_score, "High quality strategy should score higher"
    
    # Test trade count penalty
    penalized_score = manager._calculate_confidence_score(
        roi=0.20,
        sharpe=2.0,
        drawdown=0.05,
        win_rate=0.65,
        total_trades=10  # Low sample
    )
    
    assert penalized_score < high_score, "Low trade count should be penalized"
    
    print(f"✅ Confidence score test passed (high={high_score:.1f}, low={low_score:.1f})")


if __name__ == '__main__':
    print("\n=== Running Backtester Tests ===\n")
    
    test_strategy_generator()
    test_strategy_generator_types()
    test_backtest_result_dataclass()
    test_priority_list_manager()
    test_confidence_score_calculation()
    
    print("\n✅ All backtester tests passed!\n")
```

### tests\test_executor.py
```py
"""
Tests for the order executor and risk manager
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_order_creation():
    """Test Order dataclass"""
    from core.executor import Order, OrderSide, OrderType, OrderStatus
    from datetime import datetime
    
    order = Order(
        id='test_001',
        pair='BTC/USDT',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount=0.01
    )
    
    assert order.status == OrderStatus.PENDING
    assert order.is_paper == True  # Default
    
    # Test to_dict
    d = order.to_dict()
    assert d['pair'] == 'BTC/USDT'
    assert d['side'] == 'buy'
    assert d['type'] == 'market'
    
    print("✅ Order creation test passed")


def test_position_sizing():
    """Test risk manager position sizing"""
    from core.risk_manager import RiskManager
    
    rm = RiskManager(
        max_drawdown=0.20,
        risk_per_trade=0.02
    )
    
    # Test standard position sizing
    size = rm.calculate_position_size(
        balance=1000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,  # 2% stop
        leverage=1
    )
    
    # Risk = 1000 * 0.02 = 20 USDT
    # Price risk = (50000 - 49000) / 50000 = 2%
    # Position = 20 / 0.02 = 1000 USDT = 0.02 BTC
    assert 0.019 <= size <= 0.021, f"Expected ~0.02 BTC, got {size}"
    
    # Test with leverage
    size_leveraged = rm.calculate_position_size(
        balance=1000.0,
        entry_price=50000.0,
        stop_loss_price=49000.0,
        leverage=5
    )
    
    assert size_leveraged > size, "Leveraged position should be larger"
    
    print(f"✅ Position sizing test passed (size={size:.4f}, leveraged={size_leveraged:.4f})")


def test_simple_position_sizing():
    """Test simplified position sizing"""
    from core.risk_manager import RiskManager
    
    rm = RiskManager(risk_per_trade=0.02)
    
    position = rm.calculate_position_size_simple(balance=1000.0, leverage=1)
    
    # Should be capped and reasonable
    assert position > 0
    assert position <= 1000 * 0.25  # Max 25% of balance
    
    # With leverage
    position_lev = rm.calculate_position_size_simple(balance=1000.0, leverage=5)
    assert position_lev > position
    
    print(f"✅ Simple position sizing test passed (no_lev={position:.2f}, lev5x={position_lev:.2f})")


def test_risk_evaluation():
    """Test risk level evaluation"""
    from core.risk_manager import RiskManager, RiskLevel, TradingPhase
    
    rm = RiskManager(max_drawdown=0.20)
    
    # Test normal conditions
    status = rm._evaluate_risk(
        drawdown=0.05,
        daily_pnl=0.02,
        balance=100.0,
        phase=TradingPhase.FUTURES_GROWTH
    )
    
    assert status.level == RiskLevel.NORMAL
    assert status.can_trade == True
    
    # Test elevated drawdown
    status = rm._evaluate_risk(
        drawdown=0.12,  # 60% of max
        daily_pnl=0.0,
        balance=88.0,
        phase=TradingPhase.FUTURES_GROWTH
    )
    
    assert status.level == RiskLevel.ELEVATED
    
    # Test critical drawdown
    status = rm._evaluate_risk(
        drawdown=0.22,  # Over max
        daily_pnl=-0.05,
        balance=78.0,
        phase=TradingPhase.FUTURES_GROWTH
    )
    
    assert status.level == RiskLevel.CRITICAL
    assert status.can_trade == False
    
    print("✅ Risk evaluation test passed")


def test_phase_detection():
    """Test phase transition detection"""
    from core.risk_manager import RiskManager
    from config.settings import get_settings
    
    settings = get_settings()
    rm = RiskManager()
    
    # Below target
    assert rm.should_switch_to_spot(500) == False
    assert rm.should_switch_to_spot(1999) == False
    
    # At or above target
    assert rm.should_switch_to_spot(2000) == True
    assert rm.should_switch_to_spot(5000) == True
    
    print("✅ Phase detection test passed")


def test_paper_executor_balance():
    """Test paper trading balance tracking"""
    from core.executor import OrderExecutor
    from config.settings import ExchangeType
    
    executor = OrderExecutor(
        exchange_type=ExchangeType.BINANCE,
        paper_mode=True  # Force paper mode
    )
    
    balance = executor.get_balance()
    assert 'USDT' in balance
    assert balance['USDT'] > 0
    
    print(f"✅ Paper executor balance test passed (balance={balance['USDT']:.2f} USDT)")


if __name__ == '__main__':
    print("\n=== Running Executor Tests ===\n")
    
    test_order_creation()
    test_position_sizing()
    test_simple_position_sizing()
    test_risk_evaluation()
    test_phase_detection()
    test_paper_executor_balance()
    
    print("\n✅ All executor tests passed!\n")
```

### tests\test_logger.py
```py
"""
Tests for the logging system
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_json_formatter():
    """Test that logs are formatted as valid JSON"""
    import logging
    from utils.logger import JSONFormatter
    
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name='test',
        level=logging.INFO,
        pathname='test.py',
        lineno=1,
        msg='Test message',
        args=(),
        exc_info=None
    )
    
    result = formatter.format(record)
    
    # Should be valid JSON
    parsed = json.loads(result)
    
    assert parsed['level'] == 'INFO'
    assert parsed['message'] == 'Test message'
    assert 'timestamp' in parsed
    
    print("✅ JSON formatter test passed")


def test_log_file_creation():
    """Test that log files are created correctly"""
    from utils.logger import setup_logging, get_logger
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        setup_logging(log_dir=log_dir, log_file='test.log', console=False)
        logger = get_logger('test')
        
        logger.info("Test log message")
        
        log_path = log_dir / 'test.log'
        assert log_path.exists(), "Log file should be created"
        
        with open(log_path, 'r') as f:
            content = f.read()
            parsed = json.loads(content.strip())
            assert parsed['message'] == 'Test log message'
        
        print("✅ Log file creation test passed")


def test_trade_logger():
    """Test TradeLogger specific methods"""
    from utils.logger import get_logger
    
    logger = get_logger('trade_test')
    
    # Test trade_executed
    logger.trade_executed(
        pair='BTC/USDT',
        side='buy',
        amount=0.01,
        price=50000.0,
        order_id='test123',
        strategy='SMA_10_50'
    )
    
    # Test balance_update
    logger.balance_update(100.50, 5.50, 'trade')
    
    # Test strategy_result
    logger.strategy_result(
        strategy='SMA_10_50',
        pair='BTC/USDT',
        roi=0.15,
        sharpe=1.5,
        drawdown=0.05,
        win_rate=0.60
    )
    
    print("✅ Trade logger test passed")


def test_log_parsing():
    """Test log file parsing utilities"""
    from utils.logger import setup_logging, get_logger, parse_log_file
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        
        setup_logging(log_dir=log_dir, log_file='parse_test.log', console=False)
        logger = get_logger('parse_test')
        
        # Write multiple log entries
        logger.info("First message")
        logger.warning("Second message")
        logger.error("Third message")
        
        # Import after setup to get fresh loggers
        from utils.logger import parse_log_file, get_recent_errors
        
        entries = parse_log_file(log_dir / 'parse_test.log')
        
        assert len(entries) >= 3, f"Expected at least 3 entries, got {len(entries)}"
        
        errors = get_recent_errors(log_dir / 'parse_test.log')
        assert len(errors) >= 1, "Expected at least 1 error"
        
        print("✅ Log parsing test passed")


if __name__ == '__main__':
    print("\n=== Running Logger Tests ===\n")
    
    test_json_formatter()
    test_log_file_creation()
    test_trade_logger()
    test_log_parsing()
    
    print("\n✅ All logger tests passed!\n")
```

### tests\__init__.py
```py
"""
Elite Trading System - Tests Package
"""
```

### utils\database.py
```py
"""
SQLite Database Management for Elite Trading System

Stores:
- Strategy backtest results
- Trade history
- Balance snapshots
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import json


class Database:
    """SQLite database manager for trading system"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_tables()
    
    @contextmanager
    def connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure_tables(self):
        """Create tables if they don't exist"""
        with self.connection() as conn:
            cursor = conn.cursor()
            
            # Strategies table - stores backtest results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    roi REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profit_factor REAL,
                    timeframe TEXT,
                    test_start_date TEXT,
                    test_end_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, pair, params_json)
                )
            ''')
            
            # Trades table - stores executed trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    cost REAL,
                    fee REAL,
                    strategy TEXT,
                    status TEXT DEFAULT 'executed',
                    exchange TEXT,
                    is_paper INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Balance snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL,
                    change REAL,
                    source TEXT,
                    exchange TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_pair ON strategies(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_sharpe ON strategies(sharpe_ratio DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC)')
    
    # ==================== Strategy Methods ====================
    
    def save_strategy_result(
        self,
        name: str,
        pair: str,
        params: Dict[str, Any],
        roi: float,
        sharpe_ratio: float = None,
        max_drawdown: float = None,
        win_rate: float = None,
        total_trades: int = None,
        profit_factor: float = None,
        timeframe: str = None,
        test_start_date: str = None,
        test_end_date: str = None
    ) -> int:
        """Save a strategy backtest result"""
        params_json = json.dumps(params, sort_keys=True)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO strategies 
                (name, pair, params_json, roi, sharpe_ratio, max_drawdown, win_rate,
                 total_trades, profit_factor, timeframe, test_start_date, test_end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, pair, params_json, roi, sharpe_ratio, max_drawdown, win_rate,
                  total_trades, profit_factor, timeframe, test_start_date, test_end_date))
            return cursor.lastrowid
    
    def get_top_strategies(
        self, 
        limit: int = 10, 
        pair: str = None,
        min_sharpe: float = None,
        min_roi: float = None
    ) -> List[Dict]:
        """Get top performing strategies ranked by Sharpe ratio"""
        query = 'SELECT * FROM strategies WHERE 1=1'
        params = []
        
        if pair:
            query += ' AND pair = ?'
            params.append(pair)
        
        if min_sharpe is not None:
            query += ' AND sharpe_ratio >= ?'
            params.append(min_sharpe)
        
        if min_roi is not None:
            query += ' AND roi >= ?'
            params.append(min_roi)
        
        query += ' ORDER BY sharpe_ratio DESC, roi DESC LIMIT ?'
        params.append(limit)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_strategy_by_name(self, name: str, pair: str = None) -> List[Dict]:
        """Get all results for a specific strategy"""
        query = 'SELECT * FROM strategies WHERE name = ?'
        params = [name]
        
        if pair:
            query += ' AND pair = ?'
            params.append(pair)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ==================== Trade Methods ====================
    
    def save_trade(
        self,
        pair: str,
        side: str,
        amount: float,
        price: float,
        order_id: str = None,
        cost: float = None,
        fee: float = None,
        strategy: str = None,
        status: str = 'executed',
        exchange: str = None,
        is_paper: bool = False
    ) -> int:
        """Save an executed trade"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades 
                (order_id, pair, side, amount, price, cost, fee, strategy, status, exchange, is_paper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, pair, side, amount, price, cost, fee, strategy, status, exchange, 
                  1 if is_paper else 0))
            return cursor.lastrowid
    
    def get_recent_trades(self, limit: int = 50, pair: str = None) -> List[Dict]:
        """Get recent trades"""
        query = 'SELECT * FROM trades'
        params = []
        
        if pair:
            query += ' WHERE pair = ?'
            params.append(pair)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_trade_stats(self, pair: str = None) -> Dict:
        """Get trading statistics"""
        query = '''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sell_count,
                SUM(cost) as total_volume,
                SUM(fee) as total_fees
            FROM trades
        '''
        params = []
        
        if pair:
            query += ' WHERE pair = ?'
            params.append(pair)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    # ==================== Balance Methods ====================
    
    def save_balance(
        self,
        balance: float,
        change: float = None,
        source: str = None,
        exchange: str = None
    ) -> int:
        """Save a balance snapshot"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO balances (balance, change, source, exchange)
                VALUES (?, ?, ?, ?)
            ''', (balance, change, source, exchange))
            return cursor.lastrowid
    
    def get_balance_history(self, limit: int = 100) -> List[Dict]:
        """Get balance history"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM balances ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_current_balance(self) -> Optional[float]:
        """Get the most recent balance"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT balance FROM balances ORDER BY created_at DESC LIMIT 1')
            row = cursor.fetchone()
            return row['balance'] if row else None
    
    def calculate_drawdown(self) -> Dict:
        """Calculate current and max drawdown from balance history"""
        history = self.get_balance_history(limit=1000)
        
        if not history:
            return {'current_drawdown': 0, 'max_drawdown': 0, 'peak_balance': 0}
        
        balances = [h['balance'] for h in reversed(history)]
        peak = balances[0]
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        current_balance = balances[-1]
        current_drawdown = (peak - current_balance) / peak if peak > 0 else 0
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'peak_balance': peak,
            'current_balance': current_balance
        }


# Singleton instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get or create database singleton"""
    global _database
    if _database is None:
        from config.settings import get_settings
        settings = get_settings()
        _database = Database(settings.db_path)
    return _database
```

### utils\logger.py
```py
"""
Robust JSON Logging System for Elite Trading System

Features:
- JSON formatted logs for easy parsing
- Rotating file handler (10MB, 5 backups)
- Console + file output
- Exception capture with traceback
- Module-specific loggers
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON objects.
    Easy for automated parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        
        # Base log structure
        log_record: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_record['data'] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }
        
        # Add trade-specific fields if present
        trade_fields = ['pair', 'side', 'amount', 'price', 'order_id', 'strategy', 'roi', 'balance']
        for field in trade_fields:
            if hasattr(record, field):
                if 'trade' not in log_record:
                    log_record['trade'] = {}
                log_record['trade'][field] = getattr(record, field)
        
        return json.dumps(log_record, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Format message
        msg = f"{color}[{timestamp}] [{record.levelname:8}] [{record.module}]{self.RESET} {record.getMessage()}"
        
        # Add exception if present
        if record.exc_info:
            msg += f"\n{color}{traceback.format_exception(*record.exc_info)[-1].strip()}{self.RESET}"
        
        return msg


class TradeLogger(logging.LoggerAdapter):
    """
    Logger adapter with trade-specific methods.
    Provides convenient methods for logging trades, errors, and strategy results.
    """
    
    def trade_executed(self, pair: str, side: str, amount: float, price: float, 
                       order_id: str = None, strategy: str = None):
        """Log a trade execution"""
        extra = {
            'pair': pair,
            'side': side,
            'amount': amount,
            'price': price,
            'order_id': order_id,
            'strategy': strategy
        }
        self.info(f"Trade executed: {side.upper()} {amount} {pair} @ {price}", extra=extra)
    
    def trade_failed(self, pair: str, side: str, error: str, strategy: str = None):
        """Log a failed trade"""
        extra = {'pair': pair, 'side': side, 'strategy': strategy}
        self.error(f"Trade failed: {side.upper()} {pair} - {error}", extra=extra)
    
    def strategy_result(self, strategy: str, pair: str, roi: float, sharpe: float, 
                        drawdown: float, win_rate: float):
        """Log strategy backtest result"""
        self.info(
            f"Strategy result: {strategy} on {pair} | ROI: {roi:.2%} | Sharpe: {sharpe:.2f} | "
            f"Drawdown: {drawdown:.2%} | Win Rate: {win_rate:.2%}",
            extra={'extra_data': {
                'strategy': strategy, 'pair': pair, 'roi': roi, 
                'sharpe': sharpe, 'drawdown': drawdown, 'win_rate': win_rate
            }}
        )
    
    def balance_update(self, balance: float, change: float = None, source: str = None):
        """Log balance update"""
        msg = f"Balance: {balance:.2f} USDT"
        if change is not None:
            msg += f" ({'+' if change >= 0 else ''}{change:.2f})"
        if source:
            msg += f" | Source: {source}"
        self.info(msg, extra={'balance': balance})
    
    def risk_alert(self, alert_type: str, current_value: float, threshold: float, action: str):
        """Log risk management alert"""
        self.warning(
            f"RISK ALERT: {alert_type} | Current: {current_value:.2%} | Threshold: {threshold:.2%} | Action: {action}",
            extra={'extra_data': {
                'alert_type': alert_type, 'current': current_value, 
                'threshold': threshold, 'action': action
            }}
        )


# Global logger storage
_loggers: Dict[str, TradeLogger] = {}


def setup_logging(
    log_dir: Path = None,
    log_file: str = 'trades.log',
    level: str = 'DEBUG',
    console: bool = True
) -> logging.Logger:
    """
    Set up the root logger with JSON file handler and console handler.
    
    Args:
        log_dir: Directory for log files
        log_file: Name of the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to also log to console
    
    Returns:
        Configured root logger
    """
    from config.settings import get_settings
    
    settings = get_settings()
    
    if log_dir is None:
        log_dir = settings.logs_dir
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Get root logger
    root_logger = logging.getLogger('trading_beast')
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # JSON file handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str = 'trading_beast') -> TradeLogger:
    """
    Get or create a logger for the specified module.
    
    Args:
        name: Logger name (usually module name)
    
    Returns:
        TradeLogger adapter for the specified name
    """
    if name not in _loggers:
        # Ensure root logger is set up
        root = logging.getLogger('trading_beast')
        if not root.handlers:
            setup_logging()
        
        # Create child logger
        logger = logging.getLogger(f'trading_beast.{name}')
        _loggers[name] = TradeLogger(logger, {})
    
    return _loggers[name]


def parse_log_file(log_path: Path) -> list:
    """
    Parse a JSON log file and return list of log entries.
    Useful for automated analysis and error detection.
    
    Args:
        log_path: Path to the log file
    
    Returns:
        List of parsed log dictionaries
    """
    entries = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines
                    pass
    
    return entries


def get_recent_errors(log_path: Path, count: int = 10) -> list:
    """Get the most recent error entries from log file"""
    entries = parse_log_file(log_path)
    errors = [e for e in entries if e.get('level') in ('ERROR', 'CRITICAL')]
    return errors[-count:]


def get_trade_history(log_path: Path) -> list:
    """Extract all trade entries from log file"""
    entries = parse_log_file(log_path)
    return [e for e in entries if 'trade' in e]
```

### utils\__init__.py
```py
"""
Elite Trading System - Utils Module
"""
from utils.logger import get_logger, JSONFormatter
from utils.database import Database, get_database

__all__ = ['get_logger', 'JSONFormatter', 'Database', 'get_database']
```

