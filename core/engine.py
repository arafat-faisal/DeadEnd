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
from utils.telegram import TelegramNotifier

logger = get_logger('engine')


class EngineMode(Enum):
    """Engine operating modes"""
    IDLE = "idle"
    RESEARCH = "research"
    TRADING = "trade"
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
        self.telegram = TelegramNotifier()
        
        # State
        self._running = False
        self._shutdown_event = threading.Event()
        self._current_signals: List[TradingSignal] = []
        self._active_grids: Dict[str, Any] = {}
        
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
        if pairs is None:
            from research.pair_discovery import PairDiscovery
            logger.info("Auto Pair Discovery active. Scanning for top pairs...")
            pairs = PairDiscovery(exchange_id=self.exchange_type.value).get_daily_top_pairs(limit=12)
        
        logger.info(f"Starting research: {len(pairs)} pairs")
        
        # Ensure we run grid and scalping strategies as prioritised
        if strategy_types is None:
            strategy_types = list(StrategyType) # includes GRID and SCALPING
            
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
        
        # Ensure we run exactly 6 strategies
        strategy_types = [
            StrategyType.SMA_CROSSOVER,
            StrategyType.EMA_CROSSOVER,
            StrategyType.MACD_SIGNAL,
            StrategyType.BOLLINGER_BANDS,
            StrategyType.GRID,
            StrategyType.SCALPING
        ]
        strategies = list(generator.generate_all(strategy_types))
        
        if pairs is None:
            from research.pair_discovery import PairDiscovery
            logger.info("Quick Mode: Auto Pair Discovery active. Scanning for top pairs...")
            pairs = PairDiscovery(exchange_id=self.exchange_type.value).get_daily_top_pairs(limit=10)
            
        logger.info(f"Starting quick research: {len(pairs)} pairs, {len(strategies)} strategies")
        
        # Run backtests
        results = self.backtester.run_batch(pairs, strategies, '15m', 800)
        
        # Generate priority list
        entries = self.priority_manager.generate(limit=10)
        self.priority_manager.save(entries)
        self.priority_manager.print_summary(entries)
        
        return [r.to_dict() for r in results]
    
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
        self._last_daily_pnl_time = time.time()
        
        while self._running:
            if max_iterations and iteration >= max_iterations:
                break
            
            if self._shutdown_event.is_set():
                break
                
            try:
                iteration += 1
                logger.info(f"Trading iteration {iteration}")
                
                # --- News Sentiment Guard ---
                from utils.sentiment import SentimentGuard
                sentiment_guard = SentimentGuard()
                sentiment_score = sentiment_guard.get_sentiment_score()
                
                if sentiment_score < -0.3:
                    logger.warning(f"BEARISH SENTIMENT — TRADING PAUSED ({sentiment_score:.2f}). Waiting 30 minutes...")
                    time.sleep(30 * 60)
                    continue
                
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
                    # Map strategy string to StrategyType properly
                    st_val = entry.strategy.split('_')[0]
                    if st_val == "SMA":
                        st_type = StrategyType.SMA_CROSSOVER
                    elif st_val == "GRID":
                        st_type = StrategyType.GRID
                    elif st_val == "SCALP":
                        st_type = StrategyType.SCALPING
                    else:
                        st_type = StrategyType.SMA_CROSSOVER # Fallback

                    if st_type == StrategyType.GRID:
                        # Handle Grid logic
                        grid_key = f"{entry.pair}_{entry.strategy}"
                        
                        # fetch current price for grid updates
                        df = self.pipeline.fetch_ohlcv(entry.pair, '1m', 1)
                        if df.empty:
                            continue
                        current_price = df['close'].iloc[-1]
                        
                        if grid_key not in self._active_grids:
                            from core.grid_strategy import GridStrategy
                            params = entry.params
                            params['pair'] = entry.pair
                            grid = GridStrategy(self.executor, params)
                            grid.start(current_price, current_balance)
                            self._active_grids[grid_key] = grid
                        else:
                            self._active_grids[grid_key].update(current_price)
                        continue # Skip standard signal generation

                    strategy_params = StrategyParams(
                        strategy_type=st_type,
                        name=entry.strategy,
                        params=entry.params
                    )
                    
                    signal = self._generate_signal(entry.pair, strategy_params)
                    
                    if signal and signal.signal_type != SignalType.HOLD:
                        logger.info(f"Signal: {signal.signal_type.value.upper()} {signal.pair} ({signal.strategy})")
                        self._process_signal(signal, entry)
                
                # Send daily PnL report every 24 hours
                if time.time() - self._last_daily_pnl_time > 86400:
                    self.telegram.send_daily_pnl(status.daily_pnl, status.current_balance, status.open_positions)
                    self._last_daily_pnl_time = time.time()
                
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
        
        # Stop all active grids
        for grid_key, grid in self._active_grids.items():
            grid.stop()
        
        # Log final status
        balance = self.executor.get_balance()
        if hasattr(logger, 'balance_update'):
            logger.balance_update(balance.get('USDT', 0), source="shutdown")
        else:
            logger.info(f"Final Balance: {balance.get('USDT', 0)} USDT")
        
        logger.info("Engine shutdown complete")
