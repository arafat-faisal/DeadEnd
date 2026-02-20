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
