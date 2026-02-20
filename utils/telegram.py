"""
Telegram Notifier for Elite Trading System

Sends trade alerts, risk warnings, and daily PnL summaries via Telegram.
"""
import os
import threading
from utils.logger import get_logger

logger = get_logger('telegram')

try:
    import telebot
except ImportError:
    telebot = None
    logger.warning("telebot not installed. Telegram alerts disabled. (pip install pyTelegramBotAPI)")

class TelegramNotifier:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelegramNotifier, cls).__new__(cls)
            cls._instance._init_bot()
        return cls._instance
        
    def _init_bot(self):
        self.enabled = False
        self.bot = None
        
        token = os.environ.get('TELEGRAM_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        enabled_flag = os.environ.get('TELEGRAM_ENABLED') == 'true'
        
        if enabled_flag and token and self.chat_id and telebot:
            try:
                self.bot = telebot.TeleBot(token)
                self.enabled = True
                logger.info("Telegram notifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                
    def _send_message(self, text: str):
        if not self.enabled:
            return
            
        def send():
            try:
                self.bot.send_message(self.chat_id, text, parse_mode='HTML')
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
                
        threading.Thread(target=send, daemon=True).start()

    def send_trade_alert(self, order):
        """Send alert when a trade is executed."""
        if not self.enabled or not order:
            return
            
        status_icon = "🟢" if order.side.value == "buy" else "🔴"
        mode_text = "[PAPER]" if order.is_paper else "[LIVE]"
        
        msg = f"<b>{status_icon} Trade Executed {mode_text}</b>\n\n"
        msg += f"<b>Pair:</b> {order.pair}\n"
        msg += f"<b>Side:</b> {order.side.value.upper()}\n"
        msg += f"<b>Amount:</b> {order.filled_amount:.6f}\n"
        
        if order.average_price is not None:
            msg += f"<b>Price:</b> {order.average_price:.5f}\n"
            
        msg += f"<b>Strategy:</b> {order.strategy or 'Manual'}\n"
        
        self._send_message(msg)
        
    def send_risk_alert(self, status):
        """Send alert for risk level changes."""
        if not self.enabled:
            return
            
        icon = "⚠️"
        if status.level.value == 'critical':
            icon = "🚨"
            
        msg = f"<b>{icon} Risk Alert: {status.level.value.upper()}</b>\n\n"
        msg += f"<b>Message:</b> {status.message}\n"
        msg += f"<b>Drawdown:</b> {status.current_drawdown:.2%}\n"
        msg += f"<b>Balance:</b> {status.current_balance:.2f} USDT\n"
        
        self._send_message(msg)

    def send_daily_pnl(self, pnl_pct: float, balance: float, active_trades: int):
        """Send daily PnL summary."""
        if not self.enabled:
            return
            
        icon = "📈" if pnl_pct >= 0 else "📉"
        
        msg = f"<b>{icon} Daily Summary</b>\n\n"
        msg += f"<b>Daily PnL:</b> {pnl_pct:.2%}\n"
        msg += f"<b>Current Balance:</b> {balance:.2f} USDT\n"
        msg += f"<b>Active Positions:</b> {active_trades}\n"
        
        self._send_message(msg)
