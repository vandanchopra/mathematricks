"""Margin and P&L management for orders"""
from typing import Dict
from vault.base_strategy import Order, Signal
from systems.utils import create_logger

class OrderMarginManager:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.logger = create_logger(log_level='DEBUG', logger_name='MarginManager', print_to_console=True)
        self.brokerage_fee = config_dict.get('brokerage_fee', 0.0035)
        self.slippage = config_dict.get('slippage', 0.001)

    def calculate_pnl(self, order: Order, current_price: float, avg_entry_price: float) -> float:
        """Calculate P&L for a trade"""
        # Basic P&L calculation
        gross_pnl = (current_price - avg_entry_price) * order.orderQuantity
        
        # Calculate transaction costs
        transaction_costs = current_price * order.orderQuantity * (self.brokerage_fee + self.slippage)
        
        # Net P&L
        net_pnl = gross_pnl - transaction_costs
        
        self.logger.info(f"""
P&L Calculation for {order.symbol}:
- Current Price: ${current_price:.2f}
- Entry Price: ${avg_entry_price:.2f}
- Quantity: {order.orderQuantity}
- Gross P&L: ${gross_pnl:.2f}
- Costs: ${transaction_costs:.2f}
- Net P&L: ${net_pnl:.2f}""")
        
        return net_pnl

    def update_margin(self, margin_dict: Dict, order: Order, fill_price: float, strategy_name: str,
                     pnl: float = 0) -> Dict:
        """Update margin for a filled order"""
        broker = 'sim' if self.config_dict['run_mode'] == 3 else 'ibkr'
        base_account_number = self.config_dict['base_account_numbers'][broker]
        trading_currency = self.config_dict['trading_currency']
        
        # Calculate margin impact
        margin_used = fill_price * order.orderQuantity
        transaction_costs = margin_used * (self.brokerage_fee + self.slippage)
        total_margin = margin_used + transaction_costs
        
        # Apply margin impact based on order direction
        margin_multiplier = 1 if order.orderDirection == 'BUY' else -1
        
        # Update strategy margin
        for account in [strategy_name, 'combined']:
            margin_dict[broker][base_account_number][account][trading_currency].update({
                'buying_power_used': margin_dict[broker][base_account_number][account][trading_currency]['buying_power_used'] + (total_margin * margin_multiplier),
                'buying_power_available': margin_dict[broker][base_account_number][account][trading_currency]['buying_power_available'] - (total_margin * margin_multiplier),
            })
            
            # Update total buying power if there was a P&L
            if pnl != 0:
                margin_dict[broker][base_account_number][account][trading_currency]['total_buying_power'] += pnl

        self.logger.info(f"""
Margin Update for {order.symbol}:
- Direction: {order.orderDirection}
- Margin Used: ${total_margin * margin_multiplier:.2f}
- Transaction Costs: ${transaction_costs:.2f}
- P&L Applied: ${pnl:.2f}
- Total Buying Power: ${margin_dict[broker][base_account_number]['combined'][trading_currency]['total_buying_power']:.2f}
- Buying Power Used: ${margin_dict[broker][base_account_number]['combined'][trading_currency]['buying_power_used']:.2f}""")

        return margin_dict