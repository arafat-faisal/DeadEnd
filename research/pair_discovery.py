"""
Pair Discovery for Elite Trading System

Scans exchange for top active USDT perpetuals based on volume and volatility.
"""

import ccxt
import numpy as np
from typing import List, Dict, Any
from utils.logger import get_logger

logger = get_logger('pair_discovery')

class PairDiscovery:
    """
    Discovers top trading pairs based on volume and volatility.
    """
    
    def __init__(self, exchange_id: str = 'binance'):
        self.exchange_id = exchange_id.lower()
        self.exchange = None
        
        if hasattr(ccxt, self.exchange_id):
            exchange_class = getattr(ccxt, self.exchange_id)
            try:
                self.exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
            except Exception as e:
                logger.error(f"Failed to initialize {self.exchange_id}: {e}")

    def get_daily_top_pairs(self, limit: int = 12, min_volume: float = 25_000_000) -> List[str]:
        """
        Scan all active USDT perpetuals, calculate score, and return top pairs.
        Score = quoteVolume * annualized_volatility
        """
        logger.info(f"Scanning {self.exchange_id} for top {limit} pairs...")
        all_pairs_data: Dict[str, Dict[str, Any]] = {}
        
        # Hard blacklist for problematic or low-liquidity coins, commodities, fiat
        blacklist = ['USDC', 'BUSD', 'TUSD', 'UST', 'LUNA', 'FTT', 'SPACE', 'NAORIS', 'AWE', 'RIVER', 'RAVE', 'ESP', 'ENSO',
                     'XAG', 'XAU', 'EUR', 'GBP', 'PIPPIN', 'SIREN', 'MYX', 'WLFI']
        
        if not self.exchange:
            logger.error("Exchange not initialized.")
            return []
            
        try:
            markets = self.exchange.load_markets()
            # Fetch tickers for futures/swap
            tickers = self.exchange.fetch_tickers(params={'type': 'future'} if self.exchange_id == 'binance' else {'type': 'swap'})
            
            for symbol, ticker in tickers.items():
                if 'USDT' not in symbol:
                    continue
                    
                market = markets.get(symbol, {})
                
                # Ensure it's active and a linear derivative/future
                if not market.get('active', True):
                    continue
                if market.get('spot', False):
                    continue
                if not market.get('linear', True) and not market.get('swap', True):
                    continue
                    
                # Blacklist check
                base_coin = symbol.split('/')[0].split(':')[0]
                if base_coin in blacklist:
                    continue
                    
                # Filter by minimum 24h quote volume
                vol = ticker.get('quoteVolume', 0)
                if vol is None or vol < min_volume:
                    continue
                    
                last = ticker.get('last')
                high = ticker.get('high')
                low = ticker.get('low')
                
                if not all([last, high, low]) or last <= 0:
                    continue
                    
                # Calculate approximated daily volatility
                daily_vol = (high - low) / last
                # Annualize volatility
                annualized_vol = daily_vol * np.sqrt(365)
                
                # Calculate final score
                score = vol * annualized_vol
                
                # Normalize symbol to base format, e.g., BTC/USDT
                norm_symbol = symbol.split(':')[0]
                if '/' not in norm_symbol:
                    norm_symbol = norm_symbol.replace('USDT', '/USDT')
                
                # Store if better score
                if norm_symbol not in all_pairs_data or all_pairs_data[norm_symbol]['score'] < score:
                    all_pairs_data[norm_symbol] = {
                        'symbol': norm_symbol,
                        'score': score,
                        'vol': vol,
                        'volatility': annualized_vol
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching symbols from {self.exchange_id}: {e}")
            
        # Sort by score descending
        sorted_pairs = sorted(all_pairs_data.values(), key=lambda x: x['score'], reverse=True)
        top_pairs = sorted_pairs[:limit]
        
        final_list = [p['symbol'] for p in top_pairs]
        logger.info(f"Valid liquid pairs loaded: {len(final_list)}")
        logger.info(f"Top pairs discovered: {final_list}")
        
        return final_list
