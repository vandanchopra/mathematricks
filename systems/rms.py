from copy import deepcopy
import json, os, logging
from xml.dom import NotFoundErr
from systems.utils import create_logger, sleeper, generate_hash_id

class RMS:
    def __init__(self, config_dict, market_data_extractor):
        #initializing constants from config dict
        self.config_dict = config_dict
        self.logger = create_logger(log_level='DEBUG', logger_name='RMS', print_to_console=True)
        self.max_risk_per_bet = self.config_dict["risk_management"]["max_risk_per_bet"]
        self.market_data_extractor = market_data_extractor
        # self.orders = self.load_orders_from_db()
        # self.get_strategy_portfolio()
        
    def get_portfolio(self, open_orders):
        portfolio = {}
        for multi_leg_order in open_orders:
            for order in multi_leg_order:
                symbol = order['symbol']
                quantity = order['orderQuantity']
                strategy_name = order['strategy_name']
                order_status = order['status']
                order_direction_multiplier = 1 if order['orderDirection'] == 'BUY' else -1
                if order_status in ['closed']:
                    if strategy_name not in portfolio:
                        portfolio[strategy_name] = {}
                    if symbol not in portfolio[strategy_name]:
                        portfolio[strategy_name][symbol] = 0
                    portfolio[strategy_name][symbol] += quantity * order_direction_multiplier
                
        return portfolio
    
    def ideal_portfolio_to_signals(self, ideal_portfolio_entry, margin_available, open_orders, system_timestamp, live_bool):
        def normalize_signal_strength(ideal_portfolio, market_neutral_bool):
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
                        if symbol in long_symbols:
                            ideal_portfolio[symbol]['signal_strength'] = ideal_portfolio[symbol]['signal_strength'] / total_long_signal_strength
                        else:
                            ideal_portfolio[symbol]['signal_strength'] = ideal_portfolio[symbol]['signal_strength'] / total_short_signal_strength
            else:
                '''the total signal strength should be 1, if not, make it 1'''
                if sum([abs(ideal_portfolio[i]['signal_strength']) for i in ideal_portfolio.keys()]) != 1:
                    total_signal_strength = sum([abs(ideal_portfolio[i]['signal_strength']) for i in ideal_portfolio.keys()])
                    for i in ideal_portfolio.keys():
                        ideal_portfolio[i]['signal_strength'] = ideal_portfolio[i]['signal_strength'] / total_signal_strength
                
            return ideal_portfolio
        def orderDirection_reverse(orderDirection):
            if orderDirection == 'BUY':
                return 'SELL'
            else:
                return 'BUY'
        def rebalance_portfolio_strategy(current_portfolio, normalized_ideal_portfolio, strategy_name, total_buying_power, open_orders, system_timestamp):
            # Create a delta portfolio
            delta_portfolio = {'additions':{}, 'deletions':{}}
            # Create Deletions
            # self.logger.debug({'current_portfolio':current_portfolio})
            if strategy_name in current_portfolio:
                for symbol in current_portfolio[strategy_name]:
                    if symbol not in normalized_ideal_portfolio:
                        delta_portfolio['deletions'][symbol] = current_portfolio[strategy_name][symbol]
                    
            # Create Additions
            for symbol in normalized_ideal_portfolio:
                maximum_margin_used_pct = self.config_dict['risk_management']['maximum_margin_used_pct']
                ideal_orderValue = abs(normalized_ideal_portfolio[symbol]['signal_strength'] * ((total_buying_power * maximum_margin_used_pct)/2))
                # self.logger.debug({'symbol':symbol, 'ideal_orderValue':ideal_orderValue, 'Signal Strength':normalized_ideal_portfolio[symbol]['signal_strength'], 'total_buying_power':total_buying_power, 'maximum_margin_used_pct':maximum_margin_used_pct})
                
                ideal_orderQuantity = abs(ideal_orderValue / normalized_ideal_portfolio[symbol]['current_price'])
                # self.logger.debug({'symbol':symbol, 'ideal_orderQuantity':ideal_orderQuantity, 'ideal_orderValue':ideal_orderValue, 'current_price':normalized_ideal_portfolio[symbol]['current_price']})
                order_direction_multiplier = 1 if normalized_ideal_portfolio[symbol]['orderDirection'] == 'BUY' else -1
                if strategy_name in current_portfolio and symbol in current_portfolio[strategy_name]:
                    ''' If a symbol is already present, then don't touch it. Don't add or delete it. Let it die its natural death.'''
                    # delta_portfolio['additions'][symbol] = (ideal_orderQuantity * order_direction_multiplier) - current_portfolio[strategy_name][symbol]
                    pass
                else:
                    delta_portfolio['additions'][symbol] = ideal_orderQuantity * order_direction_multiplier
            
            # self.logger.debug({'delta_portfolio':delta_portfolio})            
            rebalanced_portfolio = {}
            return delta_portfolio, rebalanced_portfolio
        
        # Get the ideal portfolio
        ideal_portfolio = ideal_portfolio_entry["ideal_portfolio"]
        strategy_name = ideal_portfolio_entry["strategy_name"]
        
        # Normalize the weights
        normalized_ideal_portfolio = normalize_signal_strength(ideal_portfolio, ideal_portfolio_entry['market_neutral'])
        # self.logger.debug({'normalized_ideal_portfolio':normalized_ideal_portfolio})
        
        # Create current portfolio from open orders
        current_portfolio = self.get_portfolio(open_orders)        
        # self.logger.debug({'open_orders':open_orders})
        # self.logger.debug({'current_portfolio':current_portfolio})
        
        # Total buying power for the strategy
        broker = 'sim' if not live_bool else 'ibkr'
        base_account_number = self.config_dict['base_account_numbers'][broker]
        trading_currency = self.config_dict['trading_currency']
        # self.logger.debug({'margin_available':margin_available})
        total_buying_power = margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power']
        
        # Create a delta portfolio
        delta_portfolio, rebalanced_portfolio = rebalance_portfolio_strategy(current_portfolio, normalized_ideal_portfolio, strategy_name, total_buying_power, open_orders, system_timestamp)
        
        self.logger.debug({'delta_portfolio':delta_portfolio})
        
        # Start Creating Signals
        signal_template = {"symbol": 'symbol',
                    "strategy_name" : strategy_name, 
                    "timestamp" : system_timestamp, 
                    "entry_order_type" : ideal_portfolio_entry["entry_order_type"], 
                    "exit_order_type" : ideal_portfolio_entry["exit_order_type"], 
                    "stoploss_pct" : ideal_portfolio_entry["stoploss_pct"], 
                    "timeInForce" : ideal_portfolio_entry["timeInForce"], 
                    "orderQuantity" : 'abs(order_qty)',
                    "orderDirection": 'orderDirection',
                    "granularity": ideal_portfolio_entry["granularity"],
                    'status': 'pending',
                    'market_neutral':ideal_portfolio_entry['market_neutral'],
                    'signal_type':'BUY_SELL'
                    }
        
        ideal_portfolio_signals = []
        # Start with creating the Deletion Signals
        for symbol in delta_portfolio['deletions']:
            # Create the list of open_orders that need to be cancelled
            orders_to_cancel = []
            symbol_ltp = None
            for multi_leg_order in open_orders:
                for order in multi_leg_order:
                    if order['symbol'] == symbol and order['status'] == 'open':
                        orders_to_cancel.append(multi_leg_order)
                        # self.logger.debug({'order_to_cancel':order})
                        symbol_ltp = list(order['symbol_ltp'].values())[-1]

            signal = deepcopy(signal_template)
            signal['symbol'] = symbol
            # self.logger.debug({'symbol':symbol, 'len-orders_to_cancel':len(orders_to_cancel)})
            signal['orderQuantity'] = abs(delta_portfolio['deletions'][symbol])
            signal['orderDirection'] = 'BUY' if delta_portfolio['deletions'][symbol] < 0 else 'SELL'
            signal['orders_to_cancel'] = orders_to_cancel
            if 'symbol_ltp' not in signal:
                signal['symbol_ltp'] = {}
            try:
                signal['symbol_ltp'][str(system_timestamp)] = symbol_ltp
            except Exception as e:
                self.logger.debug({'multi_leg_order':multi_leg_order})
                self.logger.debug({'symbol':symbol, 'symbol_ltp':symbol_ltp, 'system_timestamp':system_timestamp})
                raise Exception('Symbol LTP not found')
            signal['signal_type'] = 'ORDER_CANCELLATION'
            # self.logger.debug({'orderQuantity':signal['orderQuantity']})
            ideal_portfolio_signals.append(signal)
        
        # Create the Addition Signals
        for symbol in delta_portfolio['additions']:
            signal = deepcopy(signal_template)
            signal['symbol'] = symbol
            if 'symbol_ltp' not in signal:
                signal['symbol_ltp'] = {}
            signal['symbol_ltp'][str(system_timestamp)] = normalized_ideal_portfolio[symbol]['current_price']
            signal['orderQuantity'] = abs(delta_portfolio['additions'][symbol])
            signal['orderDirection'] = 'BUY' if delta_portfolio['additions'][symbol] > 0 else 'SELL'
            signal['signal_type'] = 'BUY_SELL'
            ideal_portfolio_signals.append(signal)
            
        return ideal_portfolio_signals

    def update_signal_history(self, signal, new_signal):
        old_signal_history = signal['history'] if 'history' in signal else []
        old_signal_history.append(signal)
        new_signal['history'] = old_signal_history
        return new_signal
    
    def calculate_stoploss(self, signal):
        if signal['signal_type'] == 'BUY_SELL':
            if (signal["exit_order_type"] == "stoploss_pct"):
                signal_new = signal.copy()
                # self.logger.debug({'signal_new':signal_new})
                current_price = list(signal["symbol_ltp"].values())[-1]
                entry_orderDirection = signal["orderDirection"]
                signal_new["stoploss_abs"] = current_price * (1-signal["stoploss_pct"]) if entry_orderDirection == "BUY" else current_price * (1+signal["stoploss_pct"])
                # signal_new['exit_order_type'] = 'stoploss_abs'
                signal_new['signal_update_by'] = 'RMS'
                signal_new['signal_update_reason'] = 'SL pct converted to SL abs'
                signal = self.update_signal_history(signal, signal_new)
                
        return signal
    
    def calculate_total_risk(self, signal, margin_available, live_bool):
        if signal['signal_type'] == 'BUY_SELL':
            if signal["exit_order_type"] in ["stoploss_abs", "stoploss_pct"]:
                current_price = list(signal["symbol_ltp"].values())[-1]
                entry_orderDirection = signal["orderDirection"]
                stoploss_abs = signal["stoploss_abs"]
                # stoploss_pct = signal["stoploss_pct"]
                orderQuantity = signal["orderQuantity"]
                # self.logger.debug({'current_price':current_price, 'stoploss_abs':stoploss_abs, 'entry_orderDirection':entry_orderDirection, 'orderQuantity':orderQuantity})
                total_risk = abs(current_price - stoploss_abs) * orderQuantity if entry_orderDirection == 'BUY' else abs(stoploss_abs - current_price) * orderQuantity
                broker = 'sim' if not live_bool else 'ibkr'
                base_account_number = self.config_dict['base_account_numbers'][broker]
                trading_currency = self.config_dict['trading_currency']
                strategy_name = signal['strategy_name']
                total_buying_power_strategy = margin_available[broker][base_account_number][strategy_name][trading_currency]['total_buying_power']
                max_risk_per_bet_abs = self.max_risk_per_bet * total_buying_power_strategy
                # self.logger.debug({'total_risk':total_risk, 'max_risk_per_bet_abs':max_risk_per_bet_abs})
                if total_risk > max_risk_per_bet_abs:
                    signal['status'] = 'rejected'
                    signal['rejection_reason'] = 'Risk exceeds max risk per bet'
                    signal['signal_update_by'] = 'RMS'
                    signal['signal_update_reason'] = 'Risk exceeds max risk per bet'
                    signal = self.update_signal_history(signal, signal)
                    self.logger.warning(f"Signal Rejected: {signal['rejection_reason']} | {signal['symbol']} | Total Risk: {total_risk} | Max Risk Per Bet: {max_risk_per_bet_abs}")
        
        return signal
    
    def confirm_margin_available(self, signal, margin_available, live_bool):
        orderType = signal['entry_order_type']
        if orderType.lower() == 'market':
            # self.logger.debug({'signal':signal})
            strategy_name = signal['strategy_name']
            symbol = signal['symbol']
            orderDirection = signal['orderDirection']
            orderDirection_multiplier = 1 if orderDirection == 'BUY' else -1
            orderQuantity = float(signal['orderQuantity'])
            fill_price = float(list(signal["symbol_ltp"].values())[-1])
            broker = 'sim' if not live_bool else 'ibkr'
            margin_used_by_order = (fill_price * orderQuantity)
            base_account_number = list(self.config_dict['account_info'][broker].keys())[0]
            trading_currency = self.config_dict['trading_currency']
            
            try:
                if margin_used_by_order > margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] or margin_used_by_order > margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']:
                    signal['status'] = 'rejected'
                    signal['rejection_reason'] = 'Insufficient Margin Available'
                    signal['signal_update_by'] = 'RMS'
                    signal['signal_update_reason'] = 'Insufficient Margin Available'
                    signal = self.update_signal_history(signal, signal)
                    # self.logger.warning(f"Signal Rejected: {signal['rejection_reason']} | Margin Used: {margin_used_by_order} | Buying Power Available: {margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available']} | Combined Buying Power Available: {margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']}")
                    # sleeper(5, 'Sleeping for 5 seconds to highlight Signal Rejection')
                else:
                    margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
                    margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available'] -= abs(margin_used_by_order)
            except Exception as e:
                self.logger.debug({'margin_used_by_order':margin_used_by_order, 'buying_power_available':margin_available[broker][base_account_number][strategy_name][trading_currency]['buying_power_available'], 'combined_buying_power_available':margin_available[broker][base_account_number]['combined'][trading_currency]['buying_power_available']})
                self.logger.debug({'signal':signal})
                raise Exception(f"Error in confirm_margin_available: {e}")
                    
        return signal, margin_available
    
    def risk_management_checklist(self, signal, margin_available, live_bool):
        ''' 
        1) Now calculate the total risk being taken. If it is more than the max_risk_per_bet, then reject the signal.
        2) Confirm the margin available. If the margin available is less than the total risk, then reject the signal.
        '''
        
        ''' Calculate the total risk being taken '''
        signal = self.calculate_total_risk(signal, margin_available, live_bool)
        
        ''' Confirm the margin available '''
        signal, margin_available = self.confirm_margin_available(signal, margin_available, live_bool)
        
        return signal, margin_available
    
    def create_order(self, signal):
        signal_orders = []
        signal_type = signal['signal_type']
        status = signal["status"]
        symbol = signal["symbol"]
        
        if status != 'rejected':
            if signal_type == 'BUY_SELL':
                if 'entry_order_type' in signal:
                    try:
                        order_leg = {'symbol': signal["symbol"], 
                            'timestamp': signal["timestamp"], 
                            "orderDirection":signal["orderDirection"],
                            "entryPrice":list(signal["symbol_ltp"].values())[-1], 
                            "orderType":signal["entry_order_type"], 
                            "timeInForce":signal["timeInForce"],
                            "orderQuantity":float(abs(int(signal['orderQuantity']))),
                            "strategy_name":signal["strategy_name"],
                            "broker": 'IBKR' if self.config_dict['run_mode'] in [1,2] else 'SIM', 
                            'granularity': signal['granularity'],
                            "status": 'pending',
                            "signal_id": signal["signal_id"],
                            "symbol_ltp": {str(signal["timestamp"]):list(signal["symbol_ltp"].values())[-1]}
                            }
                        signal_orders.append(order_leg)
                    except Exception as e:
                        self.logger.debug({'signal':signal})
                        raise Exception(f'Error in create_order: {e}, Signal: {signal}')
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
                        "status": 'pending',
                        "signal_id": signal["signal_id"],
                        "symbol_ltp": {str(signal["timestamp"]):list(signal["symbol_ltp"].values())[-1]}
                        }
                    signal_orders.append(order_leg)
            if signal_type == 'ORDER_CANCELLATION':
                # validate the total order quantity
                signal_orderQuantity = signal['orderQuantity']
                orders_to_cancel_orderQuantity = 0
                for multi_leg_order in signal['orders_to_cancel']:
                    for order in multi_leg_order:
                        if order['status'] == 'closed':
                            orders_to_cancel_orderQuantity += order['orderQuantity']
                if signal_orderQuantity != orders_to_cancel_orderQuantity:
                    self.logger.debug({'signal_orderQuantity':signal_orderQuantity, 'orders_to_cancel_orderQuantity':orders_to_cancel_orderQuantity, 'symbol':symbol})
                    raise AssertionError('Order Quantity mismatch in Order Cancellation')
                
                for multi_leg_order in signal['orders_to_cancel']:
                    # Cancel the pending order and place them as market orders and status == pending
                    for order in multi_leg_order:
                        if order['status'] in ['pending', 'open']:
                            # Send Cancel Rquest for the pending order
                            order_copy = deepcopy(order)
                            order_copy['status'] = 'cancel_pending'
                            signal_orders.append(order_copy)
                            
                            # Create a market order for the cancelled order
                            order_leg = {'symbol': order["symbol"], 
                                'timestamp': signal["timestamp"], 
                                "orderDirection": order["orderDirection"], 
                                "exitPrice":list(order["symbol_ltp"].values())[-1], 
                                "orderType":'MARKET_EXIT', 
                                "timeInForce":'GTC',
                                "orderQuantity":abs(int(order['orderQuantity'])),
                                "strategy_name":order["strategy_name"],
                                "broker": order['broker'],
                                'granularity': order['granularity'],
                                "status": 'pending',
                                "signal_id": signal["signal_id"]
                                }
                            
                            signal_orders.append(order_leg)
                # self.logger.debug({'cancel_orders':signal_orders})
        
        return signal_orders
        
    def convert_signals_to_orders(self, new_signals, margin_available, open_orders, system_timestamp, live_bool):
        all_new_signals = []
        '''Step 1: Add Signals to all_new_signals'''
        for signal in new_signals["signals"]:
            signal['status'] = 'pending'
            all_new_signals.append(signal)
        '''Step 2: Convert Ideal Portfolios to signals and add to all_new_signals'''
        for ideal_portfolio_entry in new_signals["ideal_portfolios"]:
            new_signals = self.ideal_portfolio_to_signals(ideal_portfolio_entry, margin_available, open_orders, system_timestamp, live_bool)
            for signal in new_signals:
                signal['status'] = 'pending'
                all_new_signals.append(signal)
                # self.logger.debug({'ideal_portfolio_signal':signal})
        '''Step 3: Convert all_new_signals to orders'''
        new_orders = []
        margin_available_local = deepcopy(margin_available)
        for signal in all_new_signals:
            # add signal ids if not present
            if 'signal_id' not in signal:
                signal_ = deepcopy(signal)
                del signal_['symbol_ltp']
                signal['signal_id'] = generate_hash_id(input_dict=signal_, system_timestamp=system_timestamp)
            signal = self.calculate_stoploss(signal)
            signal, margin_available_local = self.risk_management_checklist(signal, margin_available_local, live_bool)
            signal_orders = self.create_order(signal)
            if len(signal_orders) > 0:
                new_orders.append(signal_orders)
        
        return new_orders
        
            