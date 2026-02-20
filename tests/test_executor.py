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
