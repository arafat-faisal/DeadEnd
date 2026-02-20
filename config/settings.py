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
