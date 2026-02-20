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
    GRID = "grid"
    SCALPING = "scalping"
    ARB_FUNDING = "arb_funding"


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
    
    # Grid parameters
    grid_levels: List[int] = field(default_factory=lambda: [5, 7, 10])
    grid_spacing: List[float] = field(default_factory=lambda: [0.003, 0.005, 0.008])
    
    # Scalping parameters
    scalping_rsi_periods: List[int] = field(default_factory=lambda: [7, 14])
    scalping_atr_multipliers: List[float] = field(default_factory=lambda: [1.5, 2.0])
    
    # Funding Arb parameters
    arb_thresholds: List[float] = field(default_factory=lambda: [0.0005, 0.001])


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
            if short >= long and len(self.config.ma_short_periods) > 1:
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
            if short >= long and len(self.config.ma_short_periods) > 1:
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
            if fast >= slow and len(self.config.macd_fast) > 1:
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
            if short >= long and len(self.config.ma_short_periods) > 1:
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

    def generate_grid(self) -> Generator[StrategyParams, None, None]:
        """Generate Grid strategy variants"""
        for levels, spacing in product(self.config.grid_levels, self.config.grid_spacing):
            name = f"GRID_{levels}_{spacing:.3f}"
            yield StrategyParams(
                strategy_type=StrategyType.GRID,
                name=name,
                params={
                    'levels': levels,
                    'spacing': spacing
                }
            )

    def generate_scalping(self) -> Generator[StrategyParams, None, None]:
        """Generate Scalping strategy variants"""
        for rsi_period, bb_period, bb_std, atr_mult in product(
            self.config.scalping_rsi_periods,
            self.config.bb_periods,
            self.config.bb_std,
            self.config.scalping_atr_multipliers
        ):
            name = f"SCALP_RSI{rsi_period}_BB{bb_period}_ATR{atr_mult}"
            yield StrategyParams(
                strategy_type=StrategyType.SCALPING,
                name=name,
                params={
                    'rsi_period': rsi_period,
                    'bb_period': bb_period,
                    'bb_std': bb_std,
                    'atr_multiplier': atr_mult
                }
            )

    def generate_arb_funding(self) -> Generator[StrategyParams, None, None]:
        """Generate Funding Rate Arbitrage variants"""
        for threshold in self.config.arb_thresholds:
            name = f"ARB_FUNDING_{threshold:.4f}"
            yield StrategyParams(
                strategy_type=StrategyType.ARB_FUNDING,
                name=name,
                params={
                    'threshold': threshold,
                    'exchanges': ['binance', 'bitget']
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
            StrategyType.GRID: self.generate_grid,
            StrategyType.SCALPING: self.generate_scalping,
            StrategyType.ARB_FUNDING: self.generate_arb_funding,
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
            StrategyType.GRID: len(self.config.grid_levels) * len(self.config.grid_spacing),
            StrategyType.SCALPING: len(self.config.scalping_rsi_periods) * len(self.config.bb_periods) * len(self.config.bb_std) * len(self.config.scalping_atr_multipliers),
            StrategyType.ARB_FUNDING: len(self.config.arb_thresholds),
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
            bb_std=[2.0],
            grid_levels=[5],
            grid_spacing=[0.005],
            scalping_rsi_periods=[14],
            scalping_atr_multipliers=[2.0]
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
            bb_std=[1.5, 2.0, 2.5],
            grid_levels=[5, 7, 10],
            grid_spacing=[0.003, 0.005, 0.008],
            scalping_rsi_periods=[7, 10, 14],
            scalping_atr_multipliers=[1.5, 2.0, 2.5]
        )
