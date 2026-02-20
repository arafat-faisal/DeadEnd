"""
Order Executor for Elite Trading System

Handles order execution via CCXT with retry logic,
paper trading mode, and comprehensive logging.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

import ccxt

from config.settings import get_settings, get_exchange, ExchangeType
from utils.logger import get_logger
from utils.database import get_database
from utils.telegram import TelegramNotifier

logger = get_logger('executor')


class OrderStatus(Enum):
    """Order status codes"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Order representation"""
    id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution details
    exchange_order_id: Optional[str] = None
    filled_amount: float = 0.0
    average_price: Optional[float] = None
    cost: float = 0.0
    fee: float = 0.0
    
    # Metadata
    strategy: Optional[str] = None
    is_paper: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pair': self.pair,
            'side': self.side.value,
            'type': self.order_type.value,
            'amount': self.amount,
            'price': self.price,
            'status': self.status.value,
            'exchange_order_id': self.exchange_order_id,
            'filled_amount': self.filled_amount,
            'average_price': self.average_price,
            'cost': self.cost,
            'fee': self.fee,
            'strategy': self.strategy,
            'is_paper': self.is_paper,
            'created_at': self.created_at.isoformat(),
            'executed_at': self.executed_at.isoformat() if self.executed_at else None
        }


class OrderExecutor:
    """
    Executes orders on exchanges with paper trading support.
    
    Features:
    - Paper trading mode (simulated execution)
    - Retry logic with exponential backoff
    - Logging of all order activities
    - Database persistence
    """
    
    def __init__(
        self,
        exchange_type: ExchangeType = ExchangeType.BINANCE,
        paper_mode: bool = None,
        futures: bool = True,
        leverage: int = None
    ):
        self.settings = get_settings()
        self.exchange_type = exchange_type
        self.paper_mode = paper_mode if paper_mode is not None else self.settings.is_paper_mode
        self.futures = futures
        self.leverage = leverage or self.settings.default_leverage
        
        self.db = get_database()
        self.telegram = TelegramNotifier()
        self._exchange: Optional[ccxt.Exchange] = None
        
        # Paper trading simulated balance
        self._paper_balance = self.settings.starting_balance
        self._paper_positions: Dict[str, float] = {}
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._executed_orders: List[Order] = []
    
    @property
    def exchange(self) -> ccxt.Exchange:
        """Get or create exchange connection"""
        if self._exchange is None:
            self._exchange = get_exchange(self.exchange_type, self.futures)
            logger.info(f"Connected to {self.exchange_type.value} ({'futures' if self.futures else 'spot'})")
        return self._exchange
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"order_{uuid.uuid4().hex[:12]}"
    
    def _get_current_price(self, pair: str) -> float:
        """Get current market price for a pair"""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get price for {pair}: {e}")
            raise
    
    def _execute_paper_order(self, order: Order) -> Order:
        """Execute order in paper trading mode"""
        try:
            # Get current price
            price = self._get_current_price(order.pair)
            
            # Calculate cost
            cost = order.amount * price
            fee = cost * 0.001  # 0.1% fee simulation
            
            if order.side == OrderSide.BUY:
                # Check balance
                if cost + fee > self._paper_balance:
                    order.status = OrderStatus.FAILED
                    logger.trade_failed(order.pair, order.side.value, "Insufficient balance", order.strategy)
                    return order
                
                # Execute
                self._paper_balance -= (cost + fee)
                base_asset = order.pair.split('/')[0]
                self._paper_positions[base_asset] = self._paper_positions.get(base_asset, 0) + order.amount
                
            else:  # SELL
                base_asset = order.pair.split('/')[0]
                position = self._paper_positions.get(base_asset, 0)
                
                if order.amount > position:
                    order.status = OrderStatus.FAILED
                    logger.trade_failed(order.pair, order.side.value, "Insufficient position", order.strategy)
                    return order
                
                # Execute
                self._paper_positions[base_asset] -= order.amount
                self._paper_balance += (cost - fee)
            
            # Update order
            order.status = OrderStatus.FILLED
            order.exchange_order_id = f"paper_{uuid.uuid4().hex[:8]}"
            order.filled_amount = order.amount
            order.average_price = price
            order.cost = cost
            order.fee = fee
            order.executed_at = datetime.now()
            
            logger.trade_executed(
                pair=order.pair,
                side=order.side.value,
                amount=order.amount,
                price=price,
                order_id=order.exchange_order_id,
                strategy=order.strategy
            )
            self.telegram.send_trade_alert(order)
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            logger.error(f"Paper order execution failed: {e}")
            return order
    
    def _execute_live_order(self, order: Order, max_retries: int = 3) -> Order:
        """Execute order on live exchange with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Set leverage for futures
                if self.futures:
                    try:
                        self.exchange.set_leverage(self.leverage, order.pair)
                    except Exception as e:
                        logger.warning(f"Could not set leverage: {e}")
                
                # Execute order
                if order.order_type == OrderType.MARKET:
                    if order.side == OrderSide.BUY:
                        result = self.exchange.create_market_buy_order(order.pair, order.amount)
                    else:
                        result = self.exchange.create_market_sell_order(order.pair, order.amount)
                else:
                    result = self.exchange.create_limit_order(
                        order.pair,
                        order.side.value,
                        order.amount,
                        order.price
                    )
                
                # Update order from result
                order.exchange_order_id = result.get('id')
                order.status = OrderStatus.FILLED if result.get('status') == 'closed' else OrderStatus.SUBMITTED
                order.filled_amount = result.get('filled', order.amount)
                order.average_price = result.get('average', result.get('price'))
                order.cost = result.get('cost', 0)
                order.fee = result.get('fee', {}).get('cost', 0)
                order.executed_at = datetime.now()
                
                logger.trade_executed(
                    pair=order.pair,
                    side=order.side.value,
                    amount=order.amount,
                    price=order.average_price,
                    order_id=order.exchange_order_id,
                    strategy=order.strategy
                )
                self.telegram.send_trade_alert(order)
                
                return order
                
            except ccxt.InsufficientFunds as e:
                order.status = OrderStatus.FAILED
                logger.trade_failed(order.pair, order.side.value, f"Insufficient funds: {e}", order.strategy)
                return order
                
            except ccxt.InvalidOrder as e:
                order.status = OrderStatus.FAILED
                logger.trade_failed(order.pair, order.side.value, f"Invalid order: {e}", order.strategy)
                return order
                
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
                last_error = e
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Network error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                
            except Exception as e:
                order.status = OrderStatus.FAILED
                logger.error(f"Order execution failed: {e}")
                return order
        
        # Max retries exceeded
        order.status = OrderStatus.FAILED
        logger.trade_failed(order.pair, order.side.value, f"Max retries exceeded: {last_error}", order.strategy)
        return order
    
    def execute(
        self,
        pair: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: float = None,
        strategy: str = None
    ) -> Order:
        """
        Execute a trading order.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Order amount in base currency
            order_type: 'market' or 'limit'
            price: Limit price (required for limit orders)
            strategy: Strategy name for logging
        
        Returns:
            Order object with execution details
        """
        # Create order
        order = Order(
            id=self._generate_order_id(),
            pair=pair,
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            amount=amount,
            price=price,
            strategy=strategy,
            is_paper=self.paper_mode
        )
        
        logger.info(f"Executing {'PAPER' if self.paper_mode else 'LIVE'} order: "
                   f"{side.upper()} {amount} {pair} @ {order_type}")
        
        # Execute
        if self.paper_mode:
            order = self._execute_paper_order(order)
        else:
            order = self._execute_live_order(order)
        
        # Save to database
        if order.status == OrderStatus.FILLED:
            self.db.save_trade(
                pair=order.pair,
                side=order.side.value,
                amount=order.filled_amount,
                price=order.average_price,
                order_id=order.exchange_order_id,
                cost=order.cost,
                fee=order.fee,
                strategy=order.strategy,
                exchange=self.exchange_type.value,
                is_paper=order.is_paper
            )
            self._executed_orders.append(order)
        
        return order
    
    def get_balance(self) -> Dict[str, float]:
        """Get current balance"""
        if self.paper_mode:
            return {
                'USDT': self._paper_balance,
                'positions': self._paper_positions.copy()
            }
        
        try:
            balance = self.exchange.fetch_balance()
            return {
                'USDT': balance.get('USDT', {}).get('free', 0),
                'total': balance.get('total', {})
            }
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {'USDT': 0, 'error': str(e)}
    
    def get_position(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get current position for a pair"""
        if self.paper_mode:
            base_asset = pair.split('/')[0]
            return {
                'amount': self._paper_positions.get(base_asset, 0),
                'pair': pair
            }
        
        try:
            if self.futures:
                positions = self.exchange.fetch_positions([pair])
                return positions[0] if positions else None
            else:
                balance = self.exchange.fetch_balance()
                base_asset = pair.split('/')[0]
                return {
                    'amount': balance.get(base_asset, {}).get('free', 0),
                    'pair': pair
                }
        except Exception as e:
            logger.error(f"Failed to fetch position for {pair}: {e}")
            return None
    
    def cancel_order(self, order_id: str, pair: str) -> bool:
        """Cancel a pending order"""
        if self.paper_mode:
            logger.info(f"Paper order {order_id} cancelled")
            return True
        
        try:
            self.exchange.cancel_order(order_id, pair)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
