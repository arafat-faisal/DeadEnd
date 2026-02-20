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

        elif strategy.strategy_type == StrategyType.SCALPING:
            df['rsi'] = ta.momentum.rsi(df['close'], window=params['rsi_period'])
            bb = ta.volatility.BollingerBands(
                df['close'], 
                window=params['bb_period'], 
                window_dev=params['bb_std']
            )
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
        elif strategy.strategy_type == StrategyType.GRID:
            pass # Grid relies on price levels in signal generator
            
        elif strategy.strategy_type == StrategyType.ARB_FUNDING:
            pass # Arb is delta-neutral, simulated directly in return calculation
        
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

        elif strategy.strategy_type == StrategyType.SCALPING:
            # Scalping: Buy when price <= lower BB and RSI is low. 
            # Sell when price >= upper BB and RSI is high.
            # (ATR trailing stop can be simulated by tighter signal ranges or applied in execution)
            buy_signal = (df['close'] <= df['bb_lower']) & (df['rsi'] < 40)
            sell_signal = (df['close'] >= df['bb_upper']) & (df['rsi'] > 60)
            df['signal'] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

        elif strategy.strategy_type == StrategyType.GRID:
            # Grid approximation: Calculate grid levels from initial price
            spacing = params['spacing']
            levels = params['levels']
            
            if len(df) > 0:
                base_price = df['close'].iloc[0]
                # Inverse relationship: price down = inventory up (buy)
                raw_inv = -np.round((df['close'] - base_price) / (base_price * spacing))
                df['inv'] = raw_inv.clip(-levels, levels)
                
                # We record a signal when inventory changes
                inv_diff = df['inv'].diff()
                df['signal'] = np.where(inv_diff > 0, 1, np.where(inv_diff < 0, -1, 0))
                
        elif strategy.strategy_type == StrategyType.ARB_FUNDING:
            # We assume position is always held for simplicity
            df['signal'] = 1
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame, strategy: StrategyParams) -> pd.DataFrame:
        """Calculate trade returns with position tracking"""
        df = df.copy()
        
        if strategy.strategy_type == StrategyType.GRID:
            # Risk per level = 1% equity, Leverage = 10 -> Exposure per level = 10% (0.10)
            exposure_per_level = 0.10
            df['position'] = df.get('inv', 0) * exposure_per_level
            
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['position'].shift(1) * df['returns']
            
            pos_diff_abs = df['position'].diff().abs()
            trade_costs = pos_diff_abs * (self.commission + self.slippage)
            df['strategy_returns'] -= trade_costs.fillna(0)
            
            # Funding fee equivalent for simulated timeframe
            # approx 0.01% per 8h. If 15m -> 32 candles per 8h
            funding_rate_per_candle = 0.0001 / 32.0 
            df['strategy_returns'] -= (df['position'].abs().shift(1) * funding_rate_per_candle).fillna(0)
            
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
        elif strategy.strategy_type == StrategyType.ARB_FUNDING:
            # Funding Arb is delta-neutral. We earn the spread (threshold) every 8 hours on 1x leverage.
            df['position'] = 1.0
            
            threshold = strategy.params['threshold']
            
            # Calculate candles per 8 hours based on timestamp differences (in ms)
            if len(df) > 1:
                candle_ms = df['timestamp'].diff().median()
                if pd.isna(candle_ms) or candle_ms == 0:
                    candle_ms = 15 * 60 * 1000 # default to 15m
                candles_per_8h = (8 * 60 * 60 * 1000) / candle_ms
            else:
                candles_per_8h = 32
                
            fee_per_candle = threshold / candles_per_8h
            
            df['returns'] = 0.0
            df['strategy_returns'] = fee_per_candle
            
            # Subtract entry costs (2 legs) on first candle
            df.loc[df.index[0], 'strategy_returns'] -= (self.commission + self.slippage) * 2
            
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            
        else:
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
            # Force 15m for Grid
            if strategy.strategy_type == StrategyType.GRID:
                timeframe = '15m'
                if limit < 2000:
                    limit = min(limit * 4, 4000)

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
            df = self._calculate_returns(df, strategy)
            
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
