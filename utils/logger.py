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
