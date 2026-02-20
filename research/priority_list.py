"""
Priority List Manager for Elite Trading System

Ranks strategies and outputs priority list for the trading engine.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from config.settings import get_settings
from utils.database import get_database
from utils.logger import get_logger

logger = get_logger('priority_list')


@dataclass
class PriorityEntry:
    """Single entry in the priority list"""
    priority: int
    pair: str
    strategy: str
    params: Dict[str, Any]
    expected_roi: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    confidence_score: float  # Composite score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'priority': self.priority,
            'pair': self.pair,
            'strategy': self.strategy,
            'params': self.params,
            'expected_roi': round(self.expected_roi, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 4),
            'win_rate': round(self.win_rate, 4),
            'confidence_score': round(self.confidence_score, 2)
        }


class PriorityListManager:
    """
    Manages the priority list of trading strategies.
    Ranks by composite score combining multiple metrics.
    """
    
    def __init__(self, output_path: Path = None):
        self.settings = get_settings()
        self.db = get_database()
        self.output_path = output_path or self.settings.priority_list_path
    
    def _calculate_confidence_score(
        self,
        roi: float,
        sharpe: float,
        drawdown: float,
        win_rate: float,
        total_trades: int
    ) -> float:
        """
        Calculate composite confidence score.
        
        Weights:
        - Sharpe ratio: 40% (risk-adjusted returns)
        - Win rate: 25%
        - ROI: 20%
        - Drawdown: 15% (penalize high drawdown)
        """
        # Normalize each metric
        sharpe_score = min(max(sharpe, 0), 3) / 3 * 100  # Cap at 3
        win_rate_score = win_rate * 100
        roi_score = min(max(roi, -1), 2) / 2 * 100 + 50  # Normalize -100% to +200%
        drawdown_score = (1 - min(drawdown, 1)) * 100  # Lower is better
        
        # Trade count penalty for low sample size
        trade_penalty = 1.0 if total_trades >= 30 else (total_trades / 30)
        
        # Weighted average
        score = (
            sharpe_score * 0.40 +
            win_rate_score * 0.25 +
            roi_score * 0.20 +
            drawdown_score * 0.15
        ) * trade_penalty * (1 - drawdown**1.5)
        
        return score
    
    def generate(
        self,
        limit: int = 10,
        min_sharpe: float = 0.5,
        min_trades: int = 20,
        pairs: List[str] = None
    ) -> List[PriorityEntry]:
        """
        Generate priority list from database.
        
        Args:
            limit: Maximum number of entries
            min_sharpe: Minimum Sharpe ratio filter
            min_trades: Minimum number of trades filter
            pairs: Filter by specific pairs
        
        Returns:
            List of PriorityEntry sorted by confidence score
        """
        if pairs is None:
            from research.pair_discovery import PairDiscovery
            discovery = PairDiscovery()
            logger.info("No pairs provided, running PairDiscovery...")
            pairs = discovery.get_daily_top_pairs(limit=15)

        logger.info(f"Generating priority list (limit={limit}, min_sharpe={min_sharpe})")
        
        # Get all strategies from DB
        strategies = self.db.get_top_strategies(limit=limit * 3, min_sharpe=min_sharpe)
        
        if not strategies:
            logger.warning("No strategies found in database")
            return []
        
        # Calculate confidence scores and filter
        scored_strategies = []
        
        for s in strategies:
            # Apply pair filter
            if pairs and s['pair'] not in pairs:
                continue
            
            # Apply trade count filter
            if s['total_trades'] and s['total_trades'] < min_trades:
                continue
                
            # Apply win rate filter
            if s['win_rate'] and s['win_rate'] <= 0.12:
                continue
            
            confidence = self._calculate_confidence_score(
                roi=s['roi'] or 0,
                sharpe=s['sharpe_ratio'] or 0,
                drawdown=s['max_drawdown'] or 0,
                win_rate=s['win_rate'] or 0,
                total_trades=s['total_trades'] or 0
            )
            
            scored_strategies.append({
                **s,
                'confidence_score': confidence
            })
        
        # Sort by confidence score
        scored_strategies.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Create priority entries
        entries = []
        for i, s in enumerate(scored_strategies[:limit]):
            params = json.loads(s['params_json']) if isinstance(s['params_json'], str) else s['params_json']
            
            entry = PriorityEntry(
                priority=i + 1,
                pair=s['pair'],
                strategy=s['name'],
                params=params,
                expected_roi=s['roi'] or 0,
                sharpe_ratio=s['sharpe_ratio'] or 0,
                max_drawdown=s['max_drawdown'] or 0,
                win_rate=s['win_rate'] or 0,
                confidence_score=s['confidence_score']
            )
            entries.append(entry)
        
        logger.info(f"Generated priority list with {len(entries)} entries")
        return entries
    
    def save(self, entries: List[PriorityEntry] = None) -> Path:
        """
        Save priority list to JSON file.
        
        Args:
            entries: List of entries to save (generates if not provided)
        
        Returns:
            Path to saved file
        """
        if entries is None:
            entries = self.generate()
        
        data = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'count': len(entries),
            'strategies': [e.to_dict() for e in entries]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved priority list to {self.output_path}")
        return self.output_path
    
    def load(self) -> List[PriorityEntry]:
        """Load priority list from JSON file"""
        if not self.output_path.exists():
            logger.warning(f"Priority list not found: {self.output_path}")
            return []
        
        with open(self.output_path, 'r') as f:
            data = json.load(f)
        
        entries = []
        for s in data.get('strategies', []):
            entries.append(PriorityEntry(
                priority=s['priority'],
                pair=s['pair'],
                strategy=s['strategy'],
                params=s['params'],
                expected_roi=s['expected_roi'],
                sharpe_ratio=s['sharpe_ratio'],
                max_drawdown=s['max_drawdown'],
                win_rate=s['win_rate'],
                confidence_score=s['confidence_score']
            ))
        
        logger.info(f"Loaded {len(entries)} entries from priority list")
        return entries
    
    def get_top(self, n: int = 3) -> List[PriorityEntry]:
        """Get top N strategies from current priority list"""
        entries = self.load()
        return entries[:n]
    
    def get_by_pair(self, pair: str) -> Optional[PriorityEntry]:
        """Get best strategy for a specific pair"""
        entries = self.load()
        for e in entries:
            if e.pair == pair:
                return e
        return None
    
    def print_summary(self, entries: List[PriorityEntry] = None):
        """Print priority list summary to console"""
        if entries is None:
            entries = self.load()
        
        if not entries:
            print("Priority list is empty")
            return
        
        print("\n" + "="*80)
        print("PRIORITY LIST SUMMARY")
        print("="*80)
        print(f"{'#':<3} {'Pair':<12} {'Strategy':<25} {'ROI':<10} {'Sharpe':<8} {'Win%':<8} {'Score':<8}")
        print("-"*80)
        
        for e in entries:
            print(f"{e.priority:<3} {e.pair:<12} {e.strategy:<25} "
                  f"{e.expected_roi*100:>7.2f}% {e.sharpe_ratio:>7.2f} "
                  f"{e.win_rate*100:>6.1f}% {e.confidence_score:>7.2f}")
        
        print("="*80 + "\n")
