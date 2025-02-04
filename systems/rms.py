from copy import deepcopy
import json
import uuid
from systems.utils import create_logger, sleeper
from vault.base_strategy import Signal, Order
from typing import List, Dict

class RMS:
    def __init__(self, config_dict, market_data_extractor):
        self.config_dict = config_dict
        self.logger = create_logger(log_level='DEBUG', logger_name='RMS', print_to_console=True)
        self.max_risk_per_bet = self.config_dict["risk_management"]["max_risk_per_bet"]
        self.brokerage_fee = self.config_dict.get("brokerage_fee", 0.0035)  # Default 35bps
        self.slippage = self.config_dict.get("slippage", 0.001)  # Default 10bps
        self.market_data_extractor = market_data_extractor

    def get_portfolio_from_signals(self, open_signals: List[Signal]) -> Dict:
        """Calculate current portfolio positions from open signals"""
        portfolio = {}
        
        for signal in open_signals:
            strategy_name = signal.strategy_name
            
            if strategy_name not in portfolio:
                portfolio[strategy_name] = {}
                
            for order in signal.orders:
                if order.status == 'closed':
                    symbol = order.symbol
                    if symbol not in portfolio[strategy_name]:
                        portfolio[strategy_name][symbol] = 0
                    
                    direction = 1 if order.orderDirection == 'BUY' else -1
                    portfolio[strategy_name][symbol] += order.orderQuantity * direction
                    
        return portfolio

    def calculate_risk_metrics(self, signal: Signal, margin_available: Dict, live_bool: bool) -> Signal:
        """Calculate risk metrics for a signal and its orders"""
        broker = 'sim' if not live_bool else 'ibkr'
        base_account_number = self.config_dict['base_account_numbers'][broker]
        trading_currency = self.config_dict['trading_currency']
        strategy_name = signal.strategy_name

        # Get total buying power for strategy
        total_buying_power = margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power']
        max_risk_amount = total_buying_power * self.max_risk_per_bet

        # Calculate total risk across all orders and multi-asset orders
        total_risk = 0
        total_transaction_costs = 0

        # Process regular orders
        # Group orders by symbol to find entry-exit pairs
        orders_by_symbol = {}
        for order in signal.orders:
            if order.symbol not in orders_by_symbol:
                orders_by_symbol[order.symbol] = []
            orders_by_symbol[order.symbol].append(order)

        # Calculate risk for each symbol's order pair
        for symbol, orders in orders_by_symbol.items():
            entry_order = next((order for order in orders if order.order_type == "MARKET"), None)
            exit_order = next((order for order in orders if order.order_type == "STOPLOSS"), None)
            
            if not entry_order or not exit_order:
                continue

            order = entry_order  # Use entry order for transaction costs
            current_price = list(order.symbol_ltp.values())[-1]
            transaction_cost = current_price * order.orderQuantity * (self.brokerage_fee + self.slippage)
            total_transaction_costs += transaction_cost
            total_risk += abs(current_price - exit_order.price) * entry_order.orderQuantity

        if total_risk > max_risk_amount:
            signal.status = 'rejected'
            signal.rejection_reason = f'Total risk ({total_risk}) exceeds max allowed ({max_risk_amount})'
            self.logger.warning(f"Total risk ({total_risk}) exceeds max allowed ({max_risk_amount})")
            sleeper(1, 'Sleeping for 1 second for the warning...')

        return signal

    def validate_margin(self, signal: Signal, margin_available: Dict, live_bool: bool) -> Signal:
        """Validate margin requirements for a signal"""
        broker = 'sim' if not live_bool else 'ibkr'
        base_account_number = self.config_dict['base_account_numbers'][broker]
        trading_currency = self.config_dict['trading_currency']
        strategy_name = signal.strategy_name

        # Calculate total margin required
        total_margin_required = 0
        
        # Calculate margin for regular orders
        for order in signal.orders:
            if order.entryOrderBool:
                current_price = list(order.symbol_ltp.values())[-1]
                margin_required = current_price * order.orderQuantity
                total_margin_required += margin_required
            
        # Check if sufficient margin is available
        if total_margin_required > margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available']:
            signal.status = 'rejected'
            signal.rejection_reason = 'Insufficient margin available'
            self.logger.info('Insufficient margin available: Total margin required: {}, Buying power available: {}'.format(total_margin_required, margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available']))
            sleeper(2, 'Sleeping for 2 seconds for the warning...')
        else:
            # Update available margin if signal is accepted
            margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] -= total_margin_required
            margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] -= total_margin_required

        return signal

    def process_signals(self, signals: List[Signal], margin_available: Dict, open_signals: List[Signal], system_timestamp, live_bool: bool) -> List[Signal]:
        """Process and validate signals"""
        processed_signals = []
        margin_available_local = deepcopy(margin_available)

        # Track symbols with active positions
        active_symbols = set()
        for signal in open_signals:
            active_symbols.update(order.symbol for order in signal.orders if order.status != 'closed')

        for signal in signals:
            # Assign UUID if signal doesn't have an ID
            if not signal.signal_id:
                signal.signal_id = str(uuid.uuid4().hex)
                # self.logger.info(f"- Assigned new ID: {signal.signal_id}")

            # Assign UUIDs to orders that don't have IDs
            for order in signal.orders:
                if not order.order_id:
                    order.order_id = str(uuid.uuid4().hex)
                    # self.logger.debug(f"- Assigned new order ID: {order.order_id}")

            # self.logger.info(f"""
            #                     Processing Signal:
            #                     - ID: {signal.signal_id}
            #                     - Strategy: {signal.strategy_name}
            #                     - Status: {signal.status}
            #                     - Orders: {len(signal.orders)}""")
            
            # Check for duplicate positions
            signal_symbols = set()
            # for order in signal.orders:
                # self.logger.info(f"""
                #                     Order:
                #                     - Signal_ID: {signal.signal_id}
                #                     - Symbol: {order.symbol}
                #                     - Direction: {order.orderDirection}
                #                     - Quantity: {order.orderQuantity}""")
            signal_symbols.update(order.symbol for order in signal.orders)
            
            # if signal_symbols.intersection(active_symbols):
            #     self.logger.warning(f"Note: Active positions exist for: {signal_symbols.intersection(active_symbols)}")

            # Validate risk and margin requirements
            # self.logger.info("Validating risk metrics...")
            signal = self.calculate_risk_metrics(signal, margin_available_local, live_bool)
            if signal.status != 'rejected':
                # self.logger.info("Risk metrics valid, checking margin...")
                signal = self.validate_margin(signal, margin_available_local, live_bool)
                # self.logger.info("Margin requirements met")
            processed_signals.append(signal)
            # self.logger.info(f"Signal RMS Check complete - Final: Signal_id: {signal.signal_id} | status: {signal.status}")
            
        return processed_signals

    def run_rms(self, new_signals, margin_available, open_signals, system_timestamp, live_bool):
        """Run risk management system on signals"""
        # Early return if no signals
        if not new_signals or "signals" not in new_signals:
            return []
        
        # Process signals through risk management
        processed_signals = self.process_signals(new_signals["signals"], margin_available, open_signals, system_timestamp, live_bool)
        
        # Return only non-rejected signals
        return [signal for signal in processed_signals if signal.status != 'rejected']