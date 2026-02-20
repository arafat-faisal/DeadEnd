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
