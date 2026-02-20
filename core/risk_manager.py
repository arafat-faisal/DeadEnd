"""
Risk Manager for Elite Trading System

Enforces position sizing, drawdown limits, and phase transitions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from config.settings import get_settings
from utils.logger import get_logger
from utils.database import get_database
from utils.telegram import TelegramNotifier

logger = get_logger('risk_manager')


class RiskLevel(Enum):
    """Risk alert levels"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class TradingPhase(Enum):
    """Trading phases"""
    FUTURES_GROWTH = "futures_growth"  # 50 -> 2000 USDT
    SPOT_ELITE = "spot_elite"          # 2000+ USDT


@dataclass
class RiskStatus:
    """Current risk status snapshot"""
    level: RiskLevel
    current_drawdown: float
    max_drawdown_limit: float
    current_balance: float
    peak_balance: float
    daily_pnl: float
    open_positions: int
    phase: TradingPhase
    can_trade: bool
    message: str


class RiskManager:
    """
    Risk management system.
    
    Features:
    - Position sizing based on risk per trade
    - Maximum drawdown enforcement
    - Phase transition (futures -> spot at target balance)
    - Daily loss limits
    - Maximum position limits
    """
    
    def __init__(
        self,
        max_drawdown: float = None,
        risk_per_trade: float = None,
        max_positions: int = 3,
        daily_loss_limit: float = 0.10  # 10% daily loss limit
    ):
        self.settings = get_settings()
        self.db = get_database()
        self.telegram = TelegramNotifier()
        
        self.max_drawdown = max_drawdown or self.settings.max_drawdown
        self.risk_per_trade = risk_per_trade or self.settings.risk_per_trade
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit
        
        self._peak_balance = self.settings.starting_balance
        self._daily_start_balance = self.settings.starting_balance
        self._open_positions: Dict[str, Dict] = {}
        self._last_daily_reset: Optional[datetime] = None
        self._trading_halted = False
        self._halt_reason = ""
        self._last_alerted_level = RiskLevel.NORMAL
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Formula: Position = (Balance * Risk%) / (Entry - StopLoss) / Entry
        
        Args:
            balance: Current account balance
            entry_price: Expected entry price
            stop_loss_price: Stop loss price
            leverage: Leverage multiplier
        
        Returns:
            Position size in base currency
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        # Calculate risk amount in USDT
        risk_amount = balance * self.risk_per_trade
        
        # Calculate price risk (distance to stop loss)
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        
        if price_risk == 0:
            return 0.0
        
        # Position size without leverage
        position_value = risk_amount / price_risk
        
        # Apply leverage
        position_value *= leverage
        
        # Convert to base currency amount
        position_size = position_value / entry_price
        
        logger.debug(
            f"Position sizing: balance={balance}, risk={risk_amount:.2f}, "
            f"price_risk={price_risk:.2%}, size={position_size:.6f}"
        )
        
        return position_size
    
    def calculate_position_size_simple(
        self,
        balance: float,
        leverage: int = 1
    ) -> float:
        """
        Simple position size based on risk per trade without stop loss.
        Uses a default 2% stop loss distance.
        
        Args:
            balance: Current account balance
            leverage: Leverage multiplier
        
        Returns:
            Maximum position value in USDT
        """
        # Risk amount
        risk_amount = balance * self.risk_per_trade
        
        # Assume 2% price movement for stop
        default_stop_distance = 0.02
        
        # Position value
        position_value = (risk_amount / default_stop_distance) * leverage
        
        # Cap at percentage of balance
        max_position = balance * 0.25 * leverage  # Max 25% per position
        
        return min(position_value, max_position)
    
    def update_balance(self, new_balance: float, source: str = None) -> RiskStatus:
        """
        Update balance and check risk levels.
        
        Args:
            new_balance: Current balance
            source: Source of balance update
        
        Returns:
            Current RiskStatus
        """
        # Update peak
        if new_balance > self._peak_balance:
            self._peak_balance = new_balance
        
        # Check for daily reset
        now = datetime.now()
        if self._last_daily_reset is None or now.date() > self._last_daily_reset.date():
            self._daily_start_balance = new_balance
            self._last_daily_reset = now
        
        # Calculate drawdown
        current_drawdown = (self._peak_balance - new_balance) / self._peak_balance if self._peak_balance > 0 else 0
        
        # Calculate daily PnL
        daily_pnl = (new_balance - self._daily_start_balance) / self._daily_start_balance if self._daily_start_balance > 0 else 0
        
        # Log balance update
        change = new_balance - (self.db.get_current_balance() or self.settings.starting_balance)
        logger.balance_update(new_balance, change, source)
        
        # Save to database
        self.db.save_balance(new_balance, change, source)
        
        # Determine phase
        phase = TradingPhase.SPOT_ELITE if new_balance >= self.settings.futures_target else TradingPhase.FUTURES_GROWTH
        
        # Determine risk level and check limits
        status = self._evaluate_risk(current_drawdown, daily_pnl, new_balance, phase)
        
        return status
    
    def _evaluate_risk(
        self,
        drawdown: float,
        daily_pnl: float,
        balance: float,
        phase: TradingPhase
    ) -> RiskStatus:
        """Evaluate current risk level and determine if trading should continue"""
        
        can_trade = True
        message = "Trading normal"
        level = RiskLevel.NORMAL
        
        # Check drawdown thresholds
        if drawdown >= self.max_drawdown:
            level = RiskLevel.CRITICAL
            can_trade = False
            message = f"Max drawdown exceeded: {drawdown:.1%} >= {self.max_drawdown:.1%}"
            self._trading_halted = True
            self._halt_reason = message
            logger.risk_alert("MAX_DRAWDOWN", drawdown, self.max_drawdown, "HALT_TRADING")
            
        elif drawdown >= self.max_drawdown * 0.8:
            level = RiskLevel.HIGH
            message = f"Approaching max drawdown: {drawdown:.1%}"
            logger.risk_alert("HIGH_DRAWDOWN", drawdown, self.max_drawdown * 0.8, "REDUCE_RISK")
            
        elif drawdown >= self.max_drawdown * 0.5:
            level = RiskLevel.ELEVATED
            message = f"Elevated drawdown: {drawdown:.1%}"
        
        # Check daily loss limit
        if daily_pnl <= -self.daily_loss_limit:
            level = RiskLevel.CRITICAL
            can_trade = False
            message = f"Daily loss limit hit: {daily_pnl:.1%}"
            logger.risk_alert("DAILY_LOSS_LIMIT", abs(daily_pnl), self.daily_loss_limit, "HALT_TRADING")
        
        # Check position limits
        if len(self._open_positions) >= self.max_positions:
            if level != RiskLevel.CRITICAL:
                level = RiskLevel.HIGH
            message = f"Max positions reached: {len(self._open_positions)}"
        
        status = RiskStatus(
            level=level,
            current_drawdown=drawdown,
            max_drawdown_limit=self.max_drawdown,
            current_balance=balance,
            peak_balance=self._peak_balance,
            daily_pnl=daily_pnl,
            open_positions=len(self._open_positions),
            phase=phase,
            can_trade=can_trade,
            message=message
        )
        
        # Only send alert if risk level changes to avoid spam
        if level != self._last_alerted_level:
            if level in [RiskLevel.ELEVATED, RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.telegram.send_risk_alert(status)
            self._last_alerted_level = level
            
        return status
    
    def can_open_position(self, pair: str, balance: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Returns:
            Tuple of (can_open, reason)
        """
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"
        
        if pair in self._open_positions:
            return False, f"Position already open for {pair}"
        
        if len(self._open_positions) >= self.max_positions:
            return False, "Maximum positions reached"
        
        # Check current risk status
        status = self.get_status(balance)
        if not status.can_trade:
            return False, status.message
        
        return True, "OK"
    
    def open_position(
        self,
        pair: str,
        size: float,
        entry_price: float,
        side: str,
        stop_loss: float = None
    ):
        """Record an opened position"""
        self._open_positions[pair] = {
            'size': size,
            'entry_price': entry_price,
            'side': side,
            'stop_loss': stop_loss,
            'opened_at': datetime.now()
        }
        logger.info(f"Position opened: {side.upper()} {size} {pair} @ {entry_price}")
    
    def close_position(self, pair: str, exit_price: float = None) -> Optional[Dict]:
        """Record a closed position and return PnL"""
        if pair not in self._open_positions:
            return None
        
        position = self._open_positions.pop(pair)
        
        if exit_price:
            if position['side'] == 'buy':
                pnl = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
            pnl_value = position['size'] * position['entry_price'] * pnl
            
            logger.info(f"Position closed: {pair} @ {exit_price}, PnL: {pnl:.2%} ({pnl_value:+.2f} USDT)")
            
            return {
                **position,
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'pnl_value': pnl_value
            }
        
        return position
    
    def get_status(self, current_balance: float = None) -> RiskStatus:
        """Get current risk status"""
        if current_balance is None:
            current_balance = self.db.get_current_balance() or self.settings.starting_balance
        
        drawdown_data = self.db.calculate_drawdown()
        daily_pnl = (current_balance - self._daily_start_balance) / self._daily_start_balance if self._daily_start_balance > 0 else 0
        
        phase = TradingPhase.SPOT_ELITE if current_balance >= self.settings.futures_target else TradingPhase.FUTURES_GROWTH
        
        return self._evaluate_risk(
            drawdown_data.get('current_drawdown', 0),
            daily_pnl,
            current_balance,
            phase
        )
    
    def should_switch_to_spot(self, balance: float) -> bool:
        """Check if should switch from futures to spot mode"""
        return balance >= self.settings.futures_target
    
    def reset_daily(self, current_balance: float):
        """Manual reset of daily tracking"""
        self._daily_start_balance = current_balance
        self._last_daily_reset = datetime.now()
        logger.info(f"Daily tracking reset at balance: {current_balance}")
    
    def resume_trading(self):
        """Resume trading after halt"""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("Trading resumed")
