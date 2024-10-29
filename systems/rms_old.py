from audioop import mul
from copy import deepcopy
import json, os, logging

from matplotlib.cbook import delete_masked_points
from matplotlib.pylab import f, norm
from systems.utils import create_logger, sleeper

class RMS:
    def __init__(self, config_dict):
        #initializing constants from config dict
        self.config_dict = config_dict
        self.logger = create_logger(log_level='DEBUG', logger_name='RMS', print_to_console=True)
        self.max_risk_per_bet = self.config_dict["risk_management"]["max_risk_per_bet"]
        # self.orders = self.load_orders_from_db()
        # self.get_strategy_portfolio()

    def load_orders_from_db(self):
        if os.path.exists('db/vault/orders.json'):
            with open('db/vault/orders.json') as file:
                self.orders = json.load(file)
            self.logger.debug(f"Loaded {len(self.orders)} orders from the database.")
        else:
            self.orders = []
        
        return self.orders
    
    def get_strategy_portfolio(self) -> None:
        self.logger.warning("NOTE NOTE NOTE: WARNING: RED: CHECK THIS FUNCTION PROPERLY")
        for order in self.orders:
            order = order[0]
            order_qty = order["orderQuantity"]
            order_sym = order["symbol"]
            orderSide = order["orderSide"]
            if orderSide == "BUY":
                sign = 1
            else:
                sign = -1
            order_stra = order["strategy_name"]
            if order_stra == "strategy_1":
                self.strategy_1_portfolio[order_sym] = self.strategy_1_portfolio.get(order_sym, 0) + (sign*order_qty)
            elif order_stra == "strategy_2":
                self.strategy_2_portfolio[order_sym] = self.strategy_2_portfolio.get(order_sym, 0) + (sign*order_qty)

    def RMS_check_available_margin(self, signal_list, margin_available):
        signal = signal_list[-1]
        status = signal["status"]
        
        if status == 'pending':
            strategy_name = signal["strategy_name"]
            current_price = signal["symbol_ltp"]
            order_qty = signal["orderQuantity"]      
            current_margin_available = margin_available[strategy_name]['current_margin_available']
            margin_required = current_price * abs(order_qty)
            # self.logger.debug(f'Symbol: {signal_list[-1]["symbol"]}, Margin Required: {margin_required}, Margin Available: {current_margin_available}')
            
            if(margin_required > current_margin_available) and current_margin_available > 0:
                msg = "RMS Test Failed: Symbol: {}, Not enough margin. Available margin: ${}, required margin: ${}".format(signal['symbol'], current_margin_available, margin_required)
                signal_new = signal.copy()
                signal_new['status'] = 'rejected'
                signal_new['signal_update_by'] = 'RMS'
                signal_new['signal_update_reason'] = msg
                signal_list.append(signal_new)
                self.logger.debug(f"RMS Test Failed: Symbol: {signal['symbol']}, Not Enough margin. Available margin: ${margin_available[strategy_name]['current_margin_available']}, required margin: ${margin_required}")

            else:
                # self.logger.debug({'margin_available':margin_available})
                # self.logger.debug(f"RMS Test Passed: BEFORE: Symbol: {signal['symbol']}, Enough margin. Available margin: ${margin_available[strategy_name]['current_margin_available']}, required margin: ${margin_required}")
                margin_available[strategy_name]['current_margin_available'] -= margin_required
                margin_available['all']['current_margin_available'] -= margin_required
                # self.logger.debug(f"RMS Test Passed - AFTER: Symbol: {signal['symbol']}, Enough margin. Available margin: ${margin_available[strategy_name]['current_margin_available']}, required margin: ${margin_required}")
                
        return signal_list, margin_available
            
    def RMS_check_max_risk(self, signal_list, margin_available):
        signal = signal_list[-1]
        status = signal["status"]
        if status != 'rejected':
            strategy_name = signal["strategy_name"]
            current_price = signal["symbol_ltp"]
            sl_price = signal["stoploss_abs"]
            order_qty = signal["orderQuantity"]
            
            if strategy_name in margin_available:
                total_strategy_margin = margin_available[strategy_name]['total_margin_available']
            else:
                total_strategy_margin = 0

            if 'max_risk_per_bet' in signal:
                max_risk_per_bet_for_this_order = signal['max_risk_per_bet']
            else:
                max_risk_per_bet_for_this_order = self.max_risk_per_bet
                self.logger.warning(f"Max risk per bet not found in signal from Strategy: {signal['strategy_name']}, using default value of {self.max_risk_per_bet}")
            
            max_risk = max_risk_per_bet_for_this_order * total_strategy_margin
            order_risk = (current_price - sl_price) * abs(order_qty)

            if(order_risk > max_risk):
                msg = "RMS Test Failed: exceeded max risk of ${}, order_risk was calulated at: ${}".format(max_risk, round(order_risk, 2))
                signal_new = signal.copy()
                signal_new['status'] = 'rejected'
                signal_new['signal_update_by'] = 'RMS'
                signal_new['signal_update_reason'] = msg
                signal_list.append(signal_new)
            
        return signal_list
    
    def get_order(self, signal_list, open_orders):
        # self.logger.debug({'signal_list':signal_list[-1]})
        order_list = []
        signal = signal_list[-1]
        signal_type = signal['signal_type']
        status = signal["status"]
        
        if status != 'rejected':
            if signal_type == 'BUY_SELL':
                if 'entry_order_type' in signal:
                    order_leg = {'symbol': signal["symbol"], 
                        'timestamp': signal["timestamp"], 
                        "orderDirection":signal["orderDirection"], 
                        "entryPrice":round(signal["symbol_ltp"], 2), 
                        "orderType":signal["entry_order_type"], 
                        "timeInForce":signal["timeInForce"],
                        "orderQuantity":abs(int(signal['orderQuantity'])),
                        "strategy_name":signal["strategy_name"],
                        "broker": 'IBKR' if self.config_dict['run_mode'] in [1,2] else 'SIM', 
                        'granularity': signal['granularity'],
                        "status": 'pending'
                        }
                    order_list.append(order_leg)
                if 'exit_order_type' in signal:
                    # raise AssertionError('exit_order_type in signal')
                    order_leg = {'symbol': signal["symbol"], 
                        'timestamp': signal["timestamp"], 
                        "orderDirection": 'BUY' if signal["orderDirection"] == 'SELL' else 'SELL', 
                        "exitPrice":round(signal["stoploss_abs"], 2),
                        'stoploss_pct': signal["stoploss_pct"],
                        'stoploss_abs': round(signal["stoploss_abs"], 2),
                        "orderType":signal["exit_order_type"], 
                        "timeInForce": 'GTC' if signal["exit_order_type"] in ['stoploss_abs', 'stoploss_pct'] else signal["timeInForce"],
                        "orderQuantity":abs(int(signal['orderQuantity'])),
                        "strategy_name":signal["strategy_name"],
                        "broker": 'IBKR' if self.config_dict['run_mode'] in [1,2] else 'SIM',
                        'granularity': signal['granularity'],
                        "status": 'pending'
                        }
                    order_list.append(order_leg)
            if signal_type == 'ORDER_CANCELLATION':
                orders_to_cancel = signal['orders_to_cancel']
                orders_to_add = signal['orders_to_add']
                
                updated_open_orders = []
                for count_1, multi_leg_order in enumerate(open_orders):
                    updated_multi_leg_order_added = False 
                    for count_2, order in enumerate(multi_leg_order):
                        if order['status'] != 'pending':
                            order_id = order['order_id']
                            for cancel_order in orders_to_cancel:
                                if order_id == cancel_order[-1]['order_id']:
                                    updated_multi_leg_order = deepcopy(multi_leg_order)
                                    updated_multi_leg_order[count_2]['status'] = 'cancelled'
                                    for order_to_add in orders_to_add:
                                        updated_multi_leg_order.append(order_to_add)
                                    updated_open_orders.append(updated_multi_leg_order)
                                    if not updated_multi_leg_order_added:
                                        updated_multi_leg_order_added = True
                    if not updated_multi_leg_order_added:
                        updated_multi_leg_order_added = True
                open_orders = updated_open_orders

                    # Now find the order to update, and update that order.
        # self.logger.debug({'order_list':order_list})
        return order_list, open_orders
    
    def calculate_sl_price(self, signal_list):
        signal = signal_list[-1]
        signal_new = signal.copy()
        current_price = signal["symbol_ltp"]
        entry_orderDirection = signal["orderDirection"]
        signal_new["stoploss_abs"] = current_price * (1-signal["stoploss_pct"]) if entry_orderDirection == "BUY" else current_price * (1+signal["stoploss_pct"])
        # signal_new['exit_order_type'] = 'stoploss_abs'
        signal_new['signal_update_by'] = 'RMS'
        signal_new['signal_update_reason'] = 'SL pct converted to SL abs'
        # Remove sl_pct from the signal
        # signal_new.pop("sl_pct")
        signal_list.append(signal_new)
        return signal_list
    
    def RMS_check_signal(self, signal_list, margin_available):
        # finding stop loss points
        if signal_list[-1]['signal_type'] == 'BUY_SELL':
            if (signal_list[-1]["exit_order_type"] == "stoploss_pct"):
                signal_list = self.calculate_sl_price(signal_list)
            # self.logger.debug({'signal_list':signal_list})
            #checking whether the order risk is below the max risk
            signal_list = self.RMS_check_max_risk(signal_list, margin_available)
            # self.logger.debug({'signal_list':signal_list})
            
            #checking if there is enough available funds for the order
            signal_list, margin_available = self.RMS_check_available_margin(signal_list, margin_available)
            # self.logger.debug({'signal_list-RMS_check_signal':signal_list})
        
        return signal_list, margin_available

    def normalize_signal_strength(self, ideal_portfolio, market_neutral_bool):
        if market_neutral_bool:
            '''the total signal strength should be 0, if not, make it zero'''
            if sum([abs(ideal_portfolio[i]['signal_strength']) for i in ideal_portfolio.keys()]) != 0:
                # get all the long orders and short orders
                long_symbols = {}
                short_symbols = {}
                
                for i in ideal_portfolio.keys():
                    if ideal_portfolio[i]['orderDirection'] == 'BUY':
                        long_symbols[i] = ideal_portfolio[i]
                    else:
                        short_symbols[i] = ideal_portfolio[i]
                # make the sum of the signal strength for all long orders 1 and for all short orders -1
                total_long_signal_strength = sum([abs(long_symbols[i]['signal_strength']) for i in long_symbols.keys()])
                total_short_signal_strength = sum([abs(short_symbols[i]['signal_strength']) for i in short_symbols.keys()])
                for symbol in ideal_portfolio.keys():
                    ideal_portfolio[symbol]['signal_strength'] = ideal_portfolio[symbol]['signal_strength'] / total_long_signal_strength
        else:
            '''the total signal strength should be 1, if not, make it 1'''
            if sum([abs(ideal_portfolio[i]['signal_strength']) for i in ideal_portfolio.keys()]) != 1:
                total_signal_strength = sum([abs(ideal_portfolio[i]['signal_strength']) for i in ideal_portfolio.keys()])
                for i in ideal_portfolio.keys():
                    ideal_portfolio[i]['signal_strength'] = ideal_portfolio[i]['signal_strength'] / total_signal_strength
            
        return ideal_portfolio
        
    def ideal_portfolio_to_signals(self, ideal_portfolio_entry, margin_available, portfolio, open_orders, system_timestamp):
        
        # Get the ideal portfolio
        ideal_portfolio = ideal_portfolio_entry["ideal_portfolio"]
        strategy_name = ideal_portfolio_entry["strategy_name"]
        
        # Normalize the weights
        normalized_ideal_portfolio = self.normalize_signal_strength(ideal_portfolio, ideal_portfolio_entry['market_neutral'])
        self.logger.debug({'normalized_ideal_portfolio':normalized_ideal_portfolio})
        # Get Current portfolio
        current_portfolio = portfolio[strategy_name] if strategy_name in portfolio else {}
        self.logger.debug({'current_portfolio':current_portfolio})
        
        # Create a delta portfolio
        delta_portfolio = {'additions':{}, 'deletions':{}}
        
        # Get the current margins available, make a copy of it as 'temp_margin_for_calculation'
        if strategy_name in margin_available:
            current_margin_available_temp = margin_available[strategy_name]['current_margin_available']
            total_margin_available_temp = margin_available[strategy_name]['total_margin_available']
        else:
            current_margin_available_temp = 0
            total_margin_available_temp = 0
        
        # Deletions
        for symbol in current_portfolio:
            if symbol not in normalized_ideal_portfolio:
                delta_portfolio['deletions'][symbol] = current_portfolio[symbol]['position']
                current_margin_available_temp += (current_portfolio[symbol]['position'] * current_portfolio[symbol]['current_price'])

        # Additions
        for symbol in normalized_ideal_portfolio:
            orderDirection_multiplier = 1 if normalized_ideal_portfolio[symbol]['orderDirection'] == 'BUY' else -1
            if symbol not in current_portfolio:
                margin_multipler = 0.5 if ideal_portfolio_entry['market_neutral'] else 1
                fund_available_to_symbol = (total_margin_available_temp * margin_multipler) * ideal_portfolio[symbol]['signal_strength']
                current_price = ideal_portfolio[symbol]['current_price']
                order_qty = int(fund_available_to_symbol / current_price)
                delta_portfolio['additions'][symbol] = order_qty * orderDirection_multiplier
                current_margin_available_temp -= (order_qty * current_price)
                self.logger.debug({'ADDING symbol':symbol, 'fund_available_to_symbol':fund_available_to_symbol, 'current_price':current_price, 'current_quantity':0, 'order_qty':order_qty, 'order_qty_delta':order_qty})
                
            else:
                margin_multipler = 0.5 if ideal_portfolio_entry['market_neutral'] else 1
                fund_available_to_symbol = (total_margin_available_temp * margin_multipler) * ideal_portfolio[symbol]['signal_strength']
                current_price = ideal_portfolio[symbol]['current_price']
                order_qty = int(fund_available_to_symbol / current_price) * orderDirection_multiplier
                current_quantity = current_portfolio[symbol]['position']
                order_qty_delta = order_qty - current_quantity
                self.logger.debug({'ADJUSTING symbol':symbol, 'fund_available_to_symbol':fund_available_to_symbol, 'current_price':current_price, 'current_quantity':current_quantity, 'order_qty':order_qty, 'order_qty_delta':order_qty_delta})
                if order_qty_delta != 0:
                    delta_portfolio['additions'][symbol] = order_qty_delta
                    current_margin_available_temp -= (order_qty_delta * current_price)
                
        self.logger.debug({'delta_portfolio':delta_portfolio})
        self.logger.debug({'current_margin_available_temp':current_margin_available_temp, 'current_margin_available':margin_available[strategy_name]['current_margin_available']})
        
        signals = []
        # Now, for signals that need to be removed, create cancel signals
        for symbol in delta_portfolio['deletions']:
            raise AssertionError('Check this part of the code')
        # For signals that need to be added, create new signals.
        for symbol in delta_portfolio['additions']:
            current_price = ideal_portfolio[symbol]['current_price']
            order_qty = int(delta_portfolio['additions'][symbol])
            orderDirection = normalized_ideal_portfolio[symbol]['orderDirection']
            # self.logger.info({'symbol':symbol, 'ideal_portfolio_abs_position':ideal_portfolio_abs_positions[symbol]})
            # if ideal_portfolio_abs_positions[symbol]['orderDirection'] == 'BUY':
            # raise AssertionError('Check this part of the code')            
            signal = {"symbol": symbol,
                    "strategy_name" : ideal_portfolio_entry["strategy_name"], 
                    "timestamp" : ideal_portfolio_entry["timestamp"], 
                    "entry_order_type" : ideal_portfolio_entry["entry_order_type"], 
                    "exit_order_type" : ideal_portfolio_entry["exit_order_type"], 
                    "stoploss_pct" : ideal_portfolio_entry["stoploss_pct"], 
                    "symbol_ltp" : current_price,
                    "timeInForce" : ideal_portfolio_entry["timeInForce"], 
                    "orderQuantity" : abs(order_qty),
                    "orderDirection": orderDirection,
                    "granularity": ideal_portfolio_entry["granularity"],
                    'status': 'pending',
                    'market_neutral':ideal_portfolio_entry['market_neutral'],
                    'signal_type':'BUY_SELL'
                    }
            if 'max_risk_per_bet' in ideal_portfolio_entry:
                signal['max_risk_per_bet'] = ideal_portfolio_entry['max_risk_per_bet']
            # self.logger.debug(f'Adding addition signal for {symbol}, Quantity: {order_qty}, orderDirection: {orderDirection}')
            # self.logger.debug({'ideal_porfolio_new_signal':signal})
            self.logger.debug({'signal':signal})
            signals.append([signal])
        
        return signals
    
    def ideal_portfolio_to_signals_old(self, ideal_portfolio_entry, margin_available, portfolio, open_orders, system_timestamp):
        
        ideal_portfolio = ideal_portfolio_entry["ideal_portfolio"]

        # Normalize the weights
        ideal_portfolio = self.normalize_signal_strength(ideal_portfolio, ideal_portfolio_entry['market_neutral'])
        # self.logger.debug({'ideal_portfolio':ideal_portfolio})
        
        #checking to see if the margin req for ideal portfolio is under the available margin
        strategy_name = ideal_portfolio_entry["strategy_name"]
    
        if strategy_name in margin_available:
            current_margin_available = margin_available[strategy_name]['current_margin_available']
        else:
            current_margin_available = 0
        
        signals = []
        
        # Now from ideal portfolio, create ideal_portfolio with absolute positions
        ideal_portfolio_abs_positions = {}
        for symbol in ideal_portfolio:
            margin_multipler = 0.5 if ideal_portfolio_entry['market_neutral'] else 1
            fund_available_to_symbol = (current_margin_available * margin_multipler) * ideal_portfolio[symbol]['signal_strength']
            current_price = ideal_portfolio[symbol]['current_price']
            order_qty = int(fund_available_to_symbol / current_price)
            if order_qty >= 1:
                orderDirection = ideal_portfolio[symbol]['orderDirection']
                ideal_portfolio_abs_positions[symbol] = {'orderDirection': orderDirection, 'orderQuantity': order_qty, 'current_price':current_price, 'bet_size': order_qty * current_price}
            
        # self.logger.debug({'ideal_portfolio_abs_positions':ideal_portfolio_abs_positions})
        current_portfolio = portfolio[strategy_name] if strategy_name in portfolio else {}
        # Now compare it to the current ideal portfolio for this strategy and find the delta & Create the signals for additions and deletions
        self.logger.debug({'current_portfolio':current_portfolio})
        self.logger.debug({'ideal_portfolio_abs_positions':ideal_portfolio_abs_positions})
        
        ''' First work on the deletions '''
        for symbol in current_portfolio:
            if symbol not in ideal_portfolio_abs_positions:
                orders_to_cancel = []
                # Get all open orders for this symbol
                for multi_leg_order in open_orders:
                    for order in multi_leg_order:
                        if order['symbol'] == symbol and order['status'] in ['pending', 'open']:
                            orders_to_cancel.append(multi_leg_order)

                if len(orders_to_cancel) > 0:
                    # Create the exit order, based on the SL order
                    orders_to_add = []
                    for cancel_order in orders_to_cancel:
                        order = deepcopy(cancel_order[-1])
                        # Change order type to market
                        order['orderType'] = 'MARKET'
                        # Update status
                        order['status'] = 'pending'
                        if 'order_id' in order:
                            del order['order_id']
                        if 'broker_id' in order:
                            del order['broker_id']
                        orders_to_add.append(order)
                    # Create tthe order cancellation signal
                    signal = {"symbol": symbol, 
                            "strategy_name": ideal_portfolio_entry["strategy_name"], 
                            "timestamp": system_timestamp,
                            'orders_to_cancel': orders_to_cancel,
                            'orders_to_add': orders_to_add,
                            'signal_type':'ORDER_CANCELLATION',
                            'status':'pending'
                            }
                    self.logger.debug({'signal':signal})
                    signals.append([signal])
                
        ''' Now work on the additions '''
        for symbol in ideal_portfolio_abs_positions:
            if symbol in current_portfolio:
                orderQuantity_delta = ideal_portfolio_abs_positions[symbol]['orderQuantity'] - current_portfolio[symbol]['position']
            else:
                orderQuantity_delta = ideal_portfolio_abs_positions[symbol]['orderQuantity']
            
            current_price = ideal_portfolio_abs_positions[symbol]['current_price']
            order_qty = int(orderQuantity_delta)
            orderDirection = ideal_portfolio_abs_positions[symbol]['orderDirection'] if order_qty > 0 else 'SELL' if ideal_portfolio_abs_positions[symbol]['orderDirection'] == 'BUY' else 'BUY'
            # self.logger.info({'symbol':symbol, 'ideal_portfolio_abs_position':ideal_portfolio_abs_positions[symbol]})
            # if ideal_portfolio_abs_positions[symbol]['orderDirection'] == 'BUY':
            # raise AssertionError('Check this part of the code')            
            signal = {"symbol": symbol,
                    "strategy_name" : ideal_portfolio_entry["strategy_name"], 
                    "timestamp" : ideal_portfolio_entry["timestamp"], 
                    "entry_order_type" : ideal_portfolio_entry["entry_order_type"], 
                    "exit_order_type" : ideal_portfolio_entry["exit_order_type"], 
                    "stoploss_pct" : ideal_portfolio_entry["stoploss_pct"], 
                    "symbol_ltp" : current_price,
                    "timeInForce" : ideal_portfolio_entry["timeInForce"], 
                    "orderQuantity" : abs(order_qty),
                    "orderDirection": orderDirection,
                    "granularity": ideal_portfolio_entry["granularity"],
                    'status': 'pending',
                    'market_neutral':ideal_portfolio_entry['market_neutral'],
                    'signal_type':'BUY_SELL'
                    }
            if 'max_risk_per_bet' in ideal_portfolio_entry:
                signal['max_risk_per_bet'] = ideal_portfolio_entry['max_risk_per_bet']
            # self.logger.debug(f'Adding addition signal for {symbol}, Quantity: {order_qty}, orderDirection: {orderDirection}')
            # self.logger.debug({'ideal_porfolio_new_signal':signal})
            self.logger.debug({'signal':signal})
            signals.append([signal])
        
        return signals

    def create_order(self, signal_list, margin_available, open_orders):
        order_list = []
        # self.logger.debug({'signal_listBEFORE':signal_list})
        signal_list, margin_available = self.RMS_check_signal(signal_list, margin_available)
        # self.logger.debug({'signal_list-AFTER':signal_list})
        status = signal_list[-1]["status"]
        if status != 'rejected':
            order_list, open_orders = self.get_order(signal_list, open_orders)
            #     symbol = signal_list[-1]["symbol"]
            #     order_qty = signal_list[-1]["orderQuantity"]
            # if symbol in self.current_portfolio:
            #     self.current_portfolio[symbol] += order_qty
            # else:
            #     self.current_portfolio[symbol] = order_qty
        else:
            self.logger.critical(f"REJECTED: Symbol: {signal_list[-1]['symbol']}, Quantity: {signal_list[-1]['orderQuantity']}, Reason: signal_list':{signal_list[-1]['signal_update_reason']}")
        
        order_list_new = []
        for order in order_list:
            order_list_new.append(order)
        signal_list[-1]["status"] = 'completed'
        
        return signal_list, order_list_new, margin_available, open_orders
        
    def convert_signals_to_orders(self, new_signals, margin_available, portfolio, open_orders, system_timestamp):
        margin_available_local = deepcopy(margin_available)
        # Get New Signals Signals
        all_new_signals = []
        new_orders = []
        # make each signal a list
        for signal in new_signals["signals"]:
            signal['status'] = 'pending'
            all_new_signals.append([signal])
        
        # Convert ideal portfolio to generate new signals
        # self.logger.debug({'i_p':new_signals["ideal_portfolios"]})
        for ideal_portfolio_entry in new_signals["ideal_portfolios"]:
            all_new_signals += self.ideal_portfolio_to_signals(ideal_portfolio_entry, margin_available_local, portfolio, open_orders, system_timestamp)
            # for signal_list in all_new_signals:
            #     if signal_list[-1]['signal_type'] == 'BUY_SELL':
            #         self.logger.debug(f"Signal:: Symbol: {signal_list[-1]['symbol']}, Signal Type: {signal_list[-1]['signal_type']}, Quantity: {signal_list[-1]['orderQuantity']}, Direction: {signal_list[-1]['orderDirection']}, Status: {signal_list[-1]['status']}")
            #     else:
            #         self.logger.debug(f"Signal:: Symbol: {signal_list[-1]['symbol']}, Signal Type: {signal_list[-1]['signal_type']}")
        # self.logger.debug({'all_new_signals':len(all_new_signals)})
        # Convert all signals to orders
        for count, signal_list in enumerate(all_new_signals):
            signal_list, new_order_list, margin_available_local, open_orders = self.create_order(signal_list, margin_available_local, open_orders)
            # Update the signal_list received.
            all_new_signals[count] = signal_list
            if len(new_order_list) > 0:
                new_orders.append(new_order_list)
        
        return new_orders, open_orders