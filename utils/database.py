"""
SQLite Database Management for Elite Trading System

Stores:
- Strategy backtest results
- Trade history
- Balance snapshots
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import json


class Database:
    """SQLite database manager for trading system"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_tables()
    
    @contextmanager
    def connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure_tables(self):
        """Create tables if they don't exist"""
        with self.connection() as conn:
            cursor = conn.cursor()
            
            # Strategies table - stores backtest results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    roi REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profit_factor REAL,
                    timeframe TEXT,
                    test_start_date TEXT,
                    test_end_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, pair, params_json)
                )
            ''')
            
            # Trades table - stores executed trades
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    price REAL NOT NULL,
                    cost REAL,
                    fee REAL,
                    strategy TEXT,
                    status TEXT DEFAULT 'executed',
                    exchange TEXT,
                    is_paper INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Balance snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL,
                    change REAL,
                    source TEXT,
                    exchange TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_pair ON strategies(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategies_sharpe ON strategies(sharpe_ratio DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_at DESC)')
    
    # ==================== Strategy Methods ====================
    
    def save_strategy_result(
        self,
        name: str,
        pair: str,
        params: Dict[str, Any],
        roi: float,
        sharpe_ratio: float = None,
        max_drawdown: float = None,
        win_rate: float = None,
        total_trades: int = None,
        profit_factor: float = None,
        timeframe: str = None,
        test_start_date: str = None,
        test_end_date: str = None
    ) -> int:
        """Save a strategy backtest result"""
        params_json = json.dumps(params, sort_keys=True)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO strategies 
                (name, pair, params_json, roi, sharpe_ratio, max_drawdown, win_rate,
                 total_trades, profit_factor, timeframe, test_start_date, test_end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, pair, params_json, roi, sharpe_ratio, max_drawdown, win_rate,
                  total_trades, profit_factor, timeframe, test_start_date, test_end_date))
            return cursor.lastrowid
    
    def get_top_strategies(
        self, 
        limit: int = 10, 
        pair: str = None,
        min_sharpe: float = None,
        min_roi: float = None
    ) -> List[Dict]:
        """Get top performing strategies ranked by Sharpe ratio"""
        query = 'SELECT * FROM strategies WHERE 1=1'
        params = []
        
        if pair:
            query += ' AND pair = ?'
            params.append(pair)
        
        if min_sharpe is not None:
            query += ' AND sharpe_ratio >= ?'
            params.append(min_sharpe)
        
        if min_roi is not None:
            query += ' AND roi >= ?'
            params.append(min_roi)
        
        query += ' ORDER BY sharpe_ratio DESC, roi DESC LIMIT ?'
        params.append(limit)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_strategy_by_name(self, name: str, pair: str = None) -> List[Dict]:
        """Get all results for a specific strategy"""
        query = 'SELECT * FROM strategies WHERE name = ?'
        params = [name]
        
        if pair:
            query += ' AND pair = ?'
            params.append(pair)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    # ==================== Trade Methods ====================
    
    def save_trade(
        self,
        pair: str,
        side: str,
        amount: float,
        price: float,
        order_id: str = None,
        cost: float = None,
        fee: float = None,
        strategy: str = None,
        status: str = 'executed',
        exchange: str = None,
        is_paper: bool = False
    ) -> int:
        """Save an executed trade"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades 
                (order_id, pair, side, amount, price, cost, fee, strategy, status, exchange, is_paper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, pair, side, amount, price, cost, fee, strategy, status, exchange, 
                  1 if is_paper else 0))
            return cursor.lastrowid
    
    def get_recent_trades(self, limit: int = 50, pair: str = None) -> List[Dict]:
        """Get recent trades"""
        query = 'SELECT * FROM trades'
        params = []
        
        if pair:
            query += ' WHERE pair = ?'
            params.append(pair)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_trade_stats(self, pair: str = None) -> Dict:
        """Get trading statistics"""
        query = '''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buy_count,
                SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sell_count,
                SUM(cost) as total_volume,
                SUM(fee) as total_fees
            FROM trades
        '''
        params = []
        
        if pair:
            query += ' WHERE pair = ?'
            params.append(pair)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    # ==================== Balance Methods ====================
    
    def save_balance(
        self,
        balance: float,
        change: float = None,
        source: str = None,
        exchange: str = None
    ) -> int:
        """Save a balance snapshot"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO balances (balance, change, source, exchange)
                VALUES (?, ?, ?, ?)
            ''', (balance, change, source, exchange))
            return cursor.lastrowid
    
    def get_balance_history(self, limit: int = 100) -> List[Dict]:
        """Get balance history"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM balances ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_current_balance(self) -> Optional[float]:
        """Get the most recent balance"""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT balance FROM balances ORDER BY created_at DESC LIMIT 1')
            row = cursor.fetchone()
            return row['balance'] if row else None
    
    def calculate_drawdown(self) -> Dict:
        """Calculate current and max drawdown from balance history"""
        history = self.get_balance_history(limit=1000)
        
        if not history:
            return {'current_drawdown': 0, 'max_drawdown': 0, 'peak_balance': 0}
        
        balances = [h['balance'] for h in reversed(history)]
        peak = balances[0]
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        current_balance = balances[-1]
        current_drawdown = (peak - current_balance) / peak if peak > 0 else 0
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'peak_balance': peak,
            'current_balance': current_balance
        }


# Singleton instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get or create database singleton"""
    global _database
    if _database is None:
        from config.settings import get_settings
        settings = get_settings()
        _database = Database(settings.db_path)
    return _database
