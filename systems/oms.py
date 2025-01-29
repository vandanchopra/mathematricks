from copy import deepcopy
import enum
import pandas as pd
from datetime import datetime
from systems.utils import create_logger, MarketDataExtractor
from brokers.brokers import Brokers
from systems.performance_reporter import PerformanceReporter
from systems.telegram import TelegramBot
from vault.base_strategy import Signal, Order
from typing import List, Dict

class OMS:
    def __init__(self, config):
        self.config_dict = config
        self.logger = create_logger(log_level='DEBUG', logger_name='OMS', print_to_console=True)
        self.market_data_extractor = MarketDataExtractor()
        self.brokers = Brokers()
        self.profit = 0
        self.margin_available = self.update_all_margin_available()
        self.open_signals = []
        self.closed_signals = []
        self.portfolio = {}
        self.reporter = PerformanceReporter(self.market_data_extractor)
        self.granularity_lookup_dict = {"1m":60,"2m":120,"5m":300,"1d":86400}
        self.telegram_bot = TelegramBot()
        self.update_telegram = self.config_dict['update_telegram']
        self.brokerage_fee = self.config_dict.get('brokerage_fee', 0.0035)
        self.slippage = self.config_dict.get('slippage', 0.001)
    
    def get_open_signals(self):
        """Get currently open signals"""
        return self.open_signals

    def get_unfilled_orders(self) -> List[Order]:
        """Get all unfilled orders from open signals
        Returns:
            List[Order]: List of orders that are not closed or cancelled
        """
        unfilled_orders = []
        for signal in self.open_signals:
            for order in signal.orders:
                if order.status in ['pending', 'open']:
                    unfilled_orders.append(order)
        return unfilled_orders

    def get_strategy_margin_available(self, strategy_name: str) -> float:
        """Calculate available margin for a strategy"""
        num_of_strategy_count = len(self.config_dict["strategies"])
        strategy_margin = self.margin_available['all']['total_margin_available'] / num_of_strategy_count
        return strategy_margin
    
    def update_all_margin_available(self) -> Dict:
        """Update available margin across all accounts and strategies"""
        trading_currency = self.config_dict['trading_currency']
        base_currency = self.config_dict['base_currency']
        base_currency_to_trading_currency_exchange_rate = self.config_dict['base_currency_to_trading_currency_exchange_rate']
        margin_dict = {}
        # self.logger.info({'self.brokers':self.brokers})

        # Update margin for each broker and account
        for broker in self.config_dict['account_info']:
            if broker not in margin_dict:
                margin_dict[broker] = {}

            for account_number in self.config_dict['account_info'][broker]:
                if account_number not in margin_dict[broker]:
                    margin_dict[broker][account_number] = {'combined':{}}
                
                if broker == 'sim':
                    margin_dict[broker][account_number]['combined'] = (
                        self.brokers.sim.execute.create_account_summary(
                        trading_currency, base_currency, base_currency_to_trading_currency_exchange_rate, 
                        self.config_dict['account_info'][broker][account_number]
                        )
                    )
                elif broker == 'ibkr' and self.config_dict['run_mode'] in [1,2]:
                    margin_dict[broker][account_number]['combined'] = (
                        self.brokers.ib.execute.get_account_summary(
                            trading_currency, base_currency, account_number
                        )
                    )
                elif broker == 'ibkr' and self.config_dict['run_mode'] == 3:
                            margin_dict[broker][account_number]['combined'] = self.brokers.sim.execute.create_account_summary(
                        trading_currency, base_currency, base_currency_to_trading_currency_exchange_rate,
                        account_number
                    )
                else:
                    raise AssertionError(f"Broker '{broker}' not supported in run_mode {self.config_dict['run_mode']}")
        # Distribute margin among strategies
        for broker in margin_dict:
            for account_number in margin_dict[broker]:
                for strategy_name in self.config_dict["strategies"]:
                    strategy_name = strategy_name.split('.')[-1] if '.' in strategy_name else strategy_name
                    if strategy_name not in margin_dict[broker][account_number]:
                        margin_dict[broker][account_number][strategy_name] = {}
                    
                    for currency in margin_dict[broker][account_number]['combined']:
                        if currency not in margin_dict[broker][account_number][strategy_name]:
                            margin_dict[broker][account_number][strategy_name][currency] = {}
                        
                        for key in margin_dict[broker][account_number]['combined'][currency]:
                            if key not in ['cushion', 'margin_multiplier', 'pct_of_margin_used']:
                                margin_dict[broker][account_number][strategy_name][currency][key] = (
                                    margin_dict[broker][account_number]['combined'][currency][key] / 
                                    len(self.config_dict["strategies"])
                                )
                            else:
                                margin_dict[broker][account_number][strategy_name][currency][key] = (
                                    margin_dict[broker][account_number]['combined'][currency][key]
                                )

        return margin_dict

    def update_portfolio(self, order: Order, signal: Signal):
        """Update portfolio based on a single executed order"""
        if order.status != 'closed':
            return
        
        symbol = order.symbol
        
        # Initialize portfolio structures if needed
        if signal.strategy_name not in self.portfolio:
            self.portfolio[signal.strategy_name] = {}
        if 'all' not in self.portfolio:
            self.portfolio['all'] = {}
        
        # Initialize symbol in strategy portfolio if needed
        if symbol not in self.portfolio[signal.strategy_name]:
            self.portfolio[signal.strategy_name][symbol] = {
                'position': 0,
                'average_price': 0,
                'total_value': 0
            }
            
            
        # Initialize symbol in overall portfolio if needed
        if symbol not in self.portfolio['all']:
            self.portfolio['all'][symbol] = {
                'position': 0,
                'average_price': 0,
                'total_value': 0
            }
        
        # Update positions and values
        position_change = order.orderQuantity if order.orderDirection == 'BUY' else -order.orderQuantity
        
        for portfolio_type in [signal.strategy_name, 'all']:
            current_pos = self.portfolio[portfolio_type][symbol]['position']
            new_pos = current_pos + position_change
            # Update position and recalculate value
            self.portfolio[portfolio_type][symbol]['position'] = new_pos
            if new_pos == 0:
                self.portfolio[portfolio_type][symbol]['average_price'] = 0
                self.portfolio[portfolio_type][symbol]['total_value'] = 0
            else:
                self.portfolio[portfolio_type][symbol]['average_price'] = abs((order.filled_price or 0))
                self.portfolio[portfolio_type][symbol]['total_value'] = new_pos * self.portfolio[portfolio_type][symbol]['average_price']
            
    def calculate_position_pnl(self, order: Order, signal: Signal) -> float:
        """Calculate PnL for a position"""
        if order.status != 'closed' or order.entryOrderBool:
            return 0, 0
        
        symbol = order.symbol
        order_direction = order.orderDirection
        exit_order_multiplier = 1 if order_direction == 'SELL' else -1
        exit_position_size = order.orderQuantity
        exit_position_value = exit_position_size * order.filled_price
        exit_brokerage_fee = order.brokerage_fee_abs
        exit_slippage = order.slippage_abs
        
        # Find the entry order for the symbol and calculate the average entry price
        entry_position_size = 0
        entry_position_value = 0
        entry_brokerage_fee_abs = 0
        entry_slippage_abs = 0
        for entry_order in signal.orders:
            if entry_order.symbol == symbol and entry_order.entryOrderBool and entry_order.status == 'closed':
                if entry_position_size + entry_order.orderQuantity <= exit_position_size:
                    entry_position_size += entry_order.orderQuantity
                    entry_position_value += entry_order.orderQuantity * entry_order.filled_price
                    entry_brokerage_fee_abs += entry_order.brokerage_fee_abs
                    entry_slippage_abs += entry_order.slippage_abs
                    break
                else:
                    entry_position_size = exit_position_size
                    entry_position_value += (exit_position_size - entry_position_size) * entry_order.filled_price
                    entry_brokerage_fee_abs += entry_order.brokerage_fee_abs
                    entry_slippage_abs += entry_order.slippage_abs
            
        # Calculate PnL
        pnl = (exit_position_value - entry_position_value) * exit_order_multiplier
        self.logger.info(f"PnL for {symbol} - Entry Position: {entry_position_size}, Entry Value: {entry_position_value}, Exit Value: {exit_position_value}, PnL: {pnl}")
        pnl_with_fee_and_slippage = pnl - (exit_brokerage_fee + exit_slippage + entry_brokerage_fee_abs + entry_slippage_abs)
        
        return pnl, pnl_with_fee_and_slippage
    
    def update_margin_on_fill(self, order: Order, signal: Signal):
        if order.status != 'closed' or not signal.strategy_name:
            return
            
        broker = 'sim' if self.config_dict['run_mode'] == 3 else 'ibkr'
        account = list(self.config_dict['account_info'][broker].keys())[0]  # Get first account from config
        strategy_name = signal.strategy_name
        trading_currency = self.config_dict['trading_currency']
        
        # Calculate transaction cost
        transaction_cost = order.brokerage_fee_abs + order.slippage_abs
        # self.logger.info(f"Transaction Cost: {transaction_cost}, Brokerage Fee: {order.brokerage_fee_abs}, Slippage: {order.slippage_abs}")
        
        # Reduce total_buying_power by transaction cost
        self.margin_available[broker][account]['combined'][trading_currency]['buying_power_available'] -= transaction_cost
        self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_available'] -= transaction_cost
        
        # Calculate position value
        position_value = order.orderQuantity * order.filled_price
        
        # If entering position
        if order.entryOrderBool:
            # Increase margin used
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_used'] += abs(position_value)
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_used'] += abs(position_value)
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_available'] -= abs(position_value)
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_available'] -= abs(position_value)
            
        else:
            # Decrease margin used and add PnL * -1
            exit_order_direction = order.orderDirection
            exit_order_multiplier = -1 if exit_order_direction == 'SELL' else 1
            
            entry_order_margin_used = position_value + (order.pnl * exit_order_multiplier)
            exit_order_value_returned = entry_order_margin_used + order.pnl
            
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_used'] -= entry_order_margin_used
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_used'] -= entry_order_margin_used
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_available'] += exit_order_value_returned
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_available'] += exit_order_value_returned

        self.margin_available[broker][account]['combined'][trading_currency]['total_buying_power'] = (
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_available'] + 
            self.margin_available[broker][account]['combined'][trading_currency]['buying_power_used']
        )
        self.margin_available[broker][account][strategy_name][trading_currency]['total_buying_power'] = (
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_available'] + 
            self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_used']
        )
        # self.logger.info(f"Updated margin for {strategy_name} - Buying Power Used: {self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_used']}, Buying Power Available: {self.margin_available[broker][account][strategy_name][trading_currency]['buying_power_available']}, Total Buying Power: {self.margin_available[broker][account][strategy_name][trading_currency]['total_buying_power']}")
        
    def calculate_signal_pnl(self, signal: Signal) -> float:
        """Calculate PnL for a signal"""
        pnl = 0
        for order in signal.orders:
            pnl += order.pnl or 0
            
        pnl_with_fee_and_slippage = 0
        for order in signal.orders:
            pnl_with_fee_and_slippage += order.pnl_with_fee_and_slippage or 0
            
        self.logger.info(f"Signal PnL: {pnl}, Signal PnL with Fee and Slippage: {pnl_with_fee_and_slippage}")
        
        return pnl, pnl_with_fee_and_slippage
    
    def check_if_signal_closed(self, signal: Signal):
        # Close the signal and move it to closed_signals and pop it from open_signals if all orders in the signal are closed
        updated_signal = deepcopy(signal)
        if all([order.status == 'closed' for order in signal.orders]):
            signal_pnl, signal_pnl_with_fee_and_slippage = self.calculate_signal_pnl(updated_signal)
            updated_signal.pnl = signal_pnl
            updated_signal.pnl_with_fee_and_slippage = signal_pnl_with_fee_and_slippage
            self.closed_signals.append(updated_signal)
            self.open_signals.remove(signal)
            self.logger.info(f"Signal {updated_signal.signal_id} closed. Moving to closed_signals...")
    
    def execute_signals(self, new_signals: List[Signal], system_timestamp: datetime, market_data_df: pd.DataFrame, live_bool: bool):
        """Process and execute trading signals"""
        # Add new signals to open signals
        if new_signals:
            # self.logger.info("\n=== OMS Executing Signals ===")
            self.open_signals.extend(new_signals)
        
        # Process all open signals
        closed_signals = []
            
        self.logger.info(f"Processing {len(self.open_signals)} open signals... Signal IDs: {[signal.signal_id for signal in self.open_signals]}")
        for sig_i, signal in enumerate(self.open_signals):
            active_orders = []
            positions_by_symbol = {}  # Track positions from closed orders

            for order in signal.orders:
                # self.logger.info({'order':order})
                symbol = order.symbol
                status = order.status
                
                if status not in ['closed', 'cancelled']:
                    # Update market data
                    min_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
                    close_prices = self.market_data_extractor.get_market_data_df_symbol_prices(
                        market_data_df, min_granularity, symbol, 'close'
                    )
                    if close_prices is not None and not close_prices.empty:
                        close_prices.dropna(inplace=True)
                        if len(close_prices) > 0:
                            last_price = close_prices.iloc[-1]
                            if hasattr(order, 'symbol_ltp'):
                                order.symbol_ltp[system_timestamp] = last_price
                            else:
                                setattr(order, 'symbol_ltp', {str(system_timestamp): last_price})
                            # self.logger.info(f"Updated price: ${last_price:.2f}")

                    # Execute through broker
                    prev_status = getattr(order, 'status', None)
                    broker = 'sim' if self.config_dict['run_mode'] == 3 or not live_bool else 'ibkr'
                    
                    # 1. Execute order through broker
                    if broker == 'ibkr':
                        self.logger.info(f"Sending to IBKR broker...")
                        response = self.brokers.ib.execute.execute_order(
                            order,
                            market_data_df,
                            system_timestamp
                        )
                    else:
                        # self.logger.info(f"Sending to SIM broker...")
                        response = self.brokers.sim.execute.execute_order(
                            order,
                            market_data_df,
                            system_timestamp)
                    
                    # Process broker response
                    # self.logger.info({'type':type(response), 'response':response})
                    if response:
                        # Get new status after update
                        new_status = response.status
                        # 2. Update history - Log status change
                        if prev_status != new_status:
                            # Update the order with response data
                            old_order = deepcopy(order)
                            order = response
                            # Update order with response data
                            old_order_history = getattr(old_order, 'history', [])
                            old_order.history = []
                            order.history = old_order_history + [old_order]
                            
                            # Calculate transaction cost
                            if order.status == 'closed':
                                transaction_value = order.orderQuantity * order.filled_price
                                order.brokerage_fee_abs = self.brokerage_fee * transaction_value
                                order.slippage_abs = self.slippage * transaction_value
                                # Exiting position - calculate PnL
                                pnl, pnl_with_fee_and_slippage = self.calculate_position_pnl(order, signal)
                                order.pnl = pnl
                                order.pnl_with_fee_and_slippage = pnl_with_fee_and_slippage
                        
                            # Update the order in both the current signal and open_signals list
                            # This ensures the updated status persists across iterations
                            for i, existing_order in enumerate(signal.orders):
                                if existing_order.order_id == order.order_id:
                                    signal.orders[i] = order
                                    break
                            
                            self.logger.info(f"""
                                                ORDER STATUS CHANGE:
                                                - Signal ID: {signal.signal_id}
                                                - Strategy: {signal.strategy_name}
                                                - Order Type: {order.order_type}
                                                - Symbol: {order.symbol}
                                                - Direction: {order.orderDirection}
                                                - Quantity: {order.orderQuantity}
                                                - Status Change: {prev_status} -> {new_status}
                                                - Filled Price: ${getattr(order, 'filled_price', 'N/A')},
                                                - PnL: ${order.pnl if getattr(order, 'pnl', None) else 'N/A'}
                                                - Brokerage Fee: ${order.brokerage_fee_abs if getattr(order, 'brokerage_fee_abs', None) else 'N/A'}
                                                - Slippage: ${order.slippage_abs if getattr(order, 'slippage_abs', None) else 'N/A'}
                                                - PnL with Fee and Slippage: ${order.pnl_with_fee_and_slippage if getattr(order, 'pnl_with_fee_and_slippage', None) else 'N/A'}
                                                - Order Value: ${order.orderQuantity * order.filled_price if getattr(order, 'filled_price', None) else 'N/A'}
                                                """)
                            
                            # input("Press Enter to continue...")
                            if new_status == 'closed':
                                # 3. Update portfolio based on executed orders
                                self.update_portfolio(order, signal)
                                # self.logger.info({"portfolio":self.portfolio})
                            
                                # 4. Update margin based on new portfolio state
                                self.update_margin_on_fill(order, signal)
                                # input("Press Enter to continue...")
                            
                                # 5. Move closed signals to closed_signals list if all orders are closed
                                self.check_if_signal_closed(signal)
