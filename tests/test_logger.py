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
