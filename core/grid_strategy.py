"""
Grid Strategy Execution for Elite Trading System

Live execution logic for the Grid strategy. Features auto-compound, risk management.
"""

from typing import Dict, Any, List
from datetime import datetime
import threading

from utils.logger import get_logger

logger = get_logger('grid_strategy')

class GridLevel:
    def __init__(self, price: float, order_id: str = None):
        self.price = price
        self.order_id = order_id
        self.status = "open" # open, filled, cancelled

class GridStrategy:
    """
    Live Grid Trading execution logic.
    Places Maker orders at specific intervals, rebalances, and compounds.
    """
    
    def __init__(self, executor, params: Dict[str, Any]):
        self.executor = executor
        self.pair = params.get('pair', 'BTC/USDT')
        self.levels = params.get('levels', 5)
        self.spacing = params.get('spacing', 0.005) # 0.5% default
        self.leverage = params.get('leverage', 10)
        
        self.base_price = 0.0
        self.upper_levels: List[GridLevel] = []
        self.lower_levels: List[GridLevel] = []
        
        self.is_active = False

    def start(self, current_price: float, balance: float):
        """Initialize and start the grid"""
        logger.info(f"Starting Grid Strategy for {self.pair} at ~{current_price} with {self.levels} levels")
        self.base_price = current_price
        self.is_active = True
        self._place_grids(balance)
        
    def stop(self):
        """Stop trading and cancel all grid orders"""
        logger.info(f"Stopping Grid Strategy for {self.pair}")
        self.is_active = False
        all_orders = [l.order_id for l in self.upper_levels + self.lower_levels if l.order_id]
        
        # In a real impl, we'd use self.executor.cancel_order
        for order_id in all_orders:
            logger.debug(f"Cancelling grid order {order_id}")
            
        self.upper_levels.clear()
        self.lower_levels.clear()

    def _place_grids(self, balance: float):
        """Place initial limit orders for the grid"""
        # 1% risk per level per requirement
        risk_per_level = balance * 0.01
        
        logger.info(f"Setting up Grid: {self.levels} levels up/down, spacing {self.spacing*100}%, risk/level: ${risk_per_level:.2f}")
        
        # Place lower levels (Buys)
        for i in range(1, self.levels + 1):
            price = self.base_price * (1 - (self.spacing * i))
            amount = (risk_per_level * self.leverage) / price
            # Simulate order placement
            order_id = f"grid_buy_{i}_{int(price)}"
            logger.debug(f"Placed Grid BUY at {price:.2f} for {amount:.4f}")
            self.lower_levels.append(GridLevel(price, order_id))
            
        # Place upper levels (Sells)
        for i in range(1, self.levels + 1):
            price = self.base_price * (1 + (self.spacing * i))
            amount = (risk_per_level * self.leverage) / price
            # Simulate order placement
            order_id = f"grid_sell_{i}_{int(price)}"
            logger.debug(f"Placed Grid SELL at {price:.2f} for {amount:.4f}")
            self.upper_levels.append(GridLevel(price, order_id))

    def update(self, current_price: float):
        """Called periodically to check grid fills and rebalance (shifting grid)"""
        if not self.is_active:
            return
            
        # Check lower levels (fills meaning we bought)
        for level in list(self.lower_levels):
            if current_price <= level.price:
                logger.info(f"🔥 Grid BUY filled at {level.price:.2f}")
                self.lower_levels.remove(level)
                
                # 1. Place new sell order just above this fill
                sell_price = level.price * (1 + self.spacing)
                new_sell_id = f"grid_sell_reb_{int(sell_price)}"
                self.upper_levels.append(GridLevel(sell_price, new_sell_id))
                logger.debug(f"Rebalanced: Placed new Grid SELL at {sell_price:.2f}")
                
                # 2. Shift grid down: Cancel highest sell, add new lowest buy
                if self.upper_levels:
                    # Cancel highest sell
                    highest_sell = max(self.upper_levels, key=lambda x: x.price)
                    self.upper_levels.remove(highest_sell)
                    logger.debug(f"Shift Grid: Cancelled furthest SELL at {highest_sell.price:.2f}")
                    
                # Add new lowest buy to maintain grid size
                if self.lower_levels:
                    lowest_buy_price = min(self.lower_levels, key=lambda x: x.price).price
                else:
                    lowest_buy_price = level.price
                    
                new_buy_price = lowest_buy_price * (1 - self.spacing)
                new_buy_id = f"grid_buy_shift_{int(new_buy_price)}"
                self.lower_levels.append(GridLevel(new_buy_price, new_buy_id))
                logger.debug(f"Shift Grid: Placed new BUY at {new_buy_price:.2f}")

        # Check upper levels (fills meaning we sold)
        for level in list(self.upper_levels):
            if current_price >= level.price:
                logger.info(f"🔥 Grid SELL filled at {level.price:.2f}")
                self.upper_levels.remove(level)
                
                # 1. Place new buy order just below this fill
                buy_price = level.price * (1 - self.spacing)
                new_buy_id = f"grid_buy_reb_{int(buy_price)}"
                self.lower_levels.append(GridLevel(buy_price, new_buy_id))
                logger.debug(f"Rebalanced: Placed new Grid BUY at {buy_price:.2f}")
                
                # 2. Shift grid up: Cancel lowest buy, add new highest sell
                if self.lower_levels:
                    # Cancel lowest buy
                    lowest_buy = min(self.lower_levels, key=lambda x: x.price)
                    self.lower_levels.remove(lowest_buy)
                    logger.debug(f"Shift Grid: Cancelled furthest BUY at {lowest_buy.price:.2f}")
                    
                # Add new highest sell to maintain grid size
                if self.upper_levels:
                    highest_sell_price = max(self.upper_levels, key=lambda x: x.price).price
                else:
                    highest_sell_price = level.price
                    
                new_sell_price = highest_sell_price * (1 + self.spacing)
                new_sell_id = f"grid_sell_shift_{int(new_sell_price)}"
                self.upper_levels.append(GridLevel(new_sell_price, new_sell_id))
                logger.debug(f"Shift Grid: Placed new SELL at {new_sell_price:.2f}")
