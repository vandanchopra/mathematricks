from copy import deepcopy
import pandas as pd
from datetime import datetime
from systems.utils import create_logger, MarketDataExtractor
from brokers.brokers import Brokers
from systems.performance_reporter import PerformanceReporter
from systems.order_margin_manager import OrderMarginManager
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
        self.margin_manager = OrderMarginManager(self.config_dict)
    
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
                        self.config_dict['account_info']['sim']['sim_1']
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

    def update_portfolio(self, signal: Signal):
        """Update portfolio based on executed signal orders"""
        # Initialize portfolio structures if needed
        if signal.strategy_name not in self.portfolio:
            self.portfolio[signal.strategy_name] = {}
        if 'all' not in self.portfolio:
            self.portfolio['all'] = {}
            
        # Process all orders (both regular and multi-asset)
        all_orders = []
        for order in signal.orders:
            if order.status != 'closed':
                continue
            all_orders.append({
                'symbol': order.symbol,
                'direction': 1 if order.orderDirection == 'BUY' else -1,
                'quantity': order.orderQuantity,
                'price': order.filled_price or 0
            })

        # Update portfolio for each order
        for order in all_orders:
            symbol = order['symbol']
            
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
            position_change = order['quantity'] * order['direction']
            value_change = order['price'] * order['quantity'] * order['direction']
            
            # Update strategy portfolio
            self.portfolio[signal.strategy_name][symbol]['position'] += position_change
            self.portfolio[signal.strategy_name][symbol]['total_value'] += value_change
            
            # Update overall portfolio
            self.portfolio['all'][symbol]['position'] += position_change
            self.portfolio['all'][symbol]['total_value'] += value_change
            
            # Update average prices
            for portfolio_type in [signal.strategy_name, 'all']:
                if self.portfolio[portfolio_type][symbol]['position'] != 0:
                    self.portfolio[portfolio_type][symbol]['average_price'] = (
                        self.portfolio[portfolio_type][symbol]['total_value'] /
                        self.portfolio[portfolio_type][symbol]['position']
                    )
        
        # Clean up zero positions
        for strategy in list(self.portfolio.keys()):
            for symbol in list(self.portfolio[strategy].keys()):
                if self.portfolio[strategy][symbol]['position'] == 0:
                    del self.portfolio[strategy][symbol]

    def update_margin_on_fill(self, order: Order, signal: Signal):
        """Update margin immediately when an order is filled"""
        if order.filled_price is None:
            self.logger.error(f"Cannot update margin: Order {order.symbol} marked as closed but has no fill price")
            return
        
        broker = 'sim' if self.config_dict['run_mode'] == 3 else 'ibkr'
        base_account_number = self.config_dict['base_account_numbers'][broker]
        trading_currency = self.config_dict['trading_currency']
        strategy_name = signal.strategy_name

        # Get current position
        current_position = 0
        if strategy_name in self.portfolio and order.symbol in self.portfolio[strategy_name]:
            current_position = self.portfolio[strategy_name][order.symbol]['position']

        # Calculate base margin amount
        margin_used = order.filled_price * order.orderQuantity
        transaction_costs = margin_used * (self.brokerage_fee + self.slippage)
        total_margin = margin_used + transaction_costs

        # Calculate position after this order
        order_size = order.orderQuantity if order.orderDirection == 'BUY' else -order.orderQuantity
        new_position = current_position + order_size

        # Determine if this is a position opening/increasing or closing/reducing order
        is_opening_order = False
        if order.orderDirection == 'BUY':
            # BUY orders:
            # - If current position >= 0: Opening/increasing long (use margin)
            # - If current position < 0: Closing/reducing short (free margin)
            is_opening_order = current_position >= 0
        else:  # SELL order
            # SELL orders:
            # - If current position <= 0: Opening/increasing short (use margin)
            # - If current position > 0: Closing/reducing long (free margin)
            is_opening_order = current_position <= 0

        # For opening orders we use margin (positive), for closing orders we free margin (negative)
        margin_multiplier = 1 if is_opening_order else -1
        
        # Update strategy margin
        self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used'] += (total_margin * margin_multiplier)
        self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] -= (total_margin * margin_multiplier)

        # Update combined margin
        self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used'] += (total_margin * margin_multiplier)
        self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] -= (total_margin * margin_multiplier)
        
        # # Enhanced logging with position and margin details
        # self.logger.info(f"""
        #                     Margin Update for {order.symbol}:
        #                     Order Details:
        #                     - Direction: {order.orderDirection}
        #                     - Quantity: {order.orderQuantity}
        #                     - Fill Price: ${order.filled_price:.2f}

        #                     Position Status:
        #                     - Current Position: {current_position} ({'Long' if current_position > 0 else 'Short' if current_position < 0 else 'Flat'})
        #                     - New Position: {new_position} ({'Long' if new_position > 0 else 'Short' if new_position < 0 else 'Flat'})
        #                     - Order Type: {'Opening/Increasing' if is_opening_order else 'Closing/Reducing'}

        #                     Margin Calculation:
        #                     - Base Margin: ${margin_used:.2f}
        #                     - Transaction Costs: ${transaction_costs:.2f}
        #                     - Margin Impact: ${total_margin * margin_multiplier:.2f} ({'Using' if margin_multiplier > 0 else 'Freeing'} margin)

        #                     Current Margin Status:
        #                     - Strategy {strategy_name} Margin Used: ${self.margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_used']:.2f}
        #                     - Overall Account Margin Used: ${self.margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_used']:.2f}
        #                     """)

    def execute_signals(self, new_signals: List[Signal], system_timestamp: datetime, market_data_df: pd.DataFrame, live_bool: bool):
        """Process and execute trading signals"""
        # Add new signals to open signals
        if new_signals:
            # self.logger.info("\n=== OMS Executing Signals ===")
            self.open_signals.extend(new_signals)
        
        # Process all open signals
        updated_signals = []
        closed_signals = []
            
        for signal in self.open_signals:
            active_orders = []
            positions_by_symbol = {}  # Track positions from closed orders
            # self.logger.info(f"\nProcessing Signal {signal.signal_id} - {signal.strategy_name}")

            all_orders = signal.orders
            
            for order in all_orders:
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
                            all_orders,
                            market_data_df,
                            system_timestamp
                        )
                    else:
                        # self.logger.info(f"Sending to SIM broker...")
                        response = self.brokers.sim.execute.execute_order(
                            order,
                            market_data_df,
                            system_timestamp
                        )
                    
                    # Process broker response
                    # self.logger.info({'type':type(response), 'response':response})
                    if response:
                        # Update order with response data
                        old_order = deepcopy(order)
                        order = deepcopy(response)
                        order.history.append(old_order)
                        
                        # Get new status after update
                        new_status = order.status
                        
                        # 2. Update history - Log status change
                        if prev_status != new_status:
                            self.logger.info(f"""
                                                ORDER STATUS CHANGE:
                                                - Signal ID: {signal.signal_id}
                                                - Strategy: {signal.strategy_name}
                                                - Symbol: {order.symbol}
                                                - Direction: {order.orderDirection}
                                                - Quantity: {order.orderQuantity}
                                                - Status Change: {prev_status} -> {new_status}
                                                - Fill Price: ${getattr(response, 'fill_price', 'N/A')}
                                                """)
                        
                        # 3. Handle closed status updates
                        if new_status == 'closed' and prev_status != 'closed':
                            # Calculate PnL when closing any position (long or short)
                            current_position = 0
                            if order.symbol in self.portfolio.get(signal.strategy_name, {}):
                                current_position = self.portfolio[signal.strategy_name][order.symbol]['position']
                                # Check if this order is closing a position
                                is_closing_position = (current_position > 0 and order.orderDirection == 'SELL') or \
                                                    (current_position < 0 and order.orderDirection == 'BUY')
                                if is_closing_position:
                                    avg_price = self.portfolio[signal.strategy_name][order.symbol]['average_price']
                                    # For shorts, reverse the PnL calculation
                                    position_multiplier = 1 if current_position > 0 else -1
                                    pnl = position_multiplier * (response.fill_price - avg_price)
                                    # Calculate PnL based on actual closing quantity
                                    closing_quantity = min(order.orderQuantity, abs(current_position))
                                    pnl = position_multiplier * (response.fill_price - avg_price) * closing_quantity
                                    self.logger.info(f"Final PnL: ${pnl:.2f} ({'PROFIT' if pnl > 0 else 'LOSS' if pnl < 0 else 'BREAKEVEN'})")
                                    self.margin_available = self.margin_manager.update_margin(self.margin_available, order, response.fill_price, signal.strategy_name, pnl)
                            
                            # Update margin and position tracking
                            self.update_margin_on_fill(order, signal)
                            
                            # Calculate position change
                            direction = 1 if order.orderDirection == 'BUY' else -1
                            quantity = order.orderQuantity * direction
                            
                            # Remove from unfilled orders if it was there

                            positions_by_symbol[order.symbol] = positions_by_symbol.get(order.symbol, 0) + quantity
                            
                            position_type = "LONG" if positions_by_symbol[order.symbol] > 0 else \
                                          "SHORT" if positions_by_symbol[order.symbol] < 0 else "FLAT"
                        
                        # 4. Track active orders
                        if new_status != 'closed':
                            active_orders.append(order)

            # Update signal status
            all_positions_closed = all(position == 0 for position in positions_by_symbol.values())

            if not active_orders and all_positions_closed:
                if all(order.status == 'closed' for order in signal.orders):
                    # Update signal status and portfolio
                    signal.status = 'closed'
                    self.update_portfolio(signal)

                    # Prepare margin info for final report
                    broker = 'sim' if self.config_dict['run_mode'] == 3 else 'ibkr'
                    base_account_number = self.config_dict['base_account_numbers'][broker]
                    trading_currency = self.config_dict['trading_currency']
                    margin_info = self.margin_available[broker][base_account_number]
                    
                    # Generate position summary
                    positions_str = "\n".join([f"  - {symbol}: {pos} (FLAT)" for symbol, pos in positions_by_symbol.items()])
                    
                    # Log final signal status
                    self.logger.info(f"""
                                    Signal {signal.signal_id} closed:
                                    - All orders complete
                                    - All positions flat
                                    - Strategy: {signal.strategy_name}
                                    - Order count: {len(signal.orders)}

                                    Strategy Margin ({signal.strategy_name}):
                                    - Total Buying Power: ${margin_info[signal.strategy_name][trading_currency]['total_buying_power']:.2f}
                                    - Buying Power Used: ${margin_info[signal.strategy_name][trading_currency]['buying_power_used']:.2f}
                                    - Buying Power Available: ${margin_info[signal.strategy_name][trading_currency]['buying_power_available']:.2f}

                                    Overall Account Margin:
                                    - Total Buying Power: ${margin_info['combined'][trading_currency]['total_buying_power']:.2f}
                                    - Buying Power Used: ${margin_info['combined'][trading_currency]['buying_power_used']:.2f}
                                    - Buying Power Available: ${margin_info['combined'][trading_currency]['buying_power_available']:.2f} 
                                    - Margin Used: {(margin_info['combined'][trading_currency]['buying_power_used'] / 
                                            margin_info['combined'][trading_currency]['total_buying_power'] * 100):.1f}%

                                    Position Summary:
                                    {positions_str}""")
                    # Move signal to closed signals list
                    closed_signals.append(signal)
                    self.open_signals = [s for s in self.open_signals if s.signal_id != signal.signal_id]
                else:
                    self.logger.warning(f"Signal {signal.signal_id} has flat positions but unclosed orders - keeping open")
            else:
                # Update active orders and keep signal open
                signal.orders = active_orders
                updated_signals.append(signal)
                # self.logger.info(f"Signal {signal.signal_id} remains open with {len(active_orders)} active orders")

        # Update closed signals list
        self.closed_signals.extend(closed_signals)

        return updated_signals, closed_signals
