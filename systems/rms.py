import json, os, logging
from systems.utils import create_logger, sleeper

class RMS:
    def __init__(self, config_dict):
        #initializing constants from config dict
        self.config_dict = config_dict
        self.logger = create_logger(log_level='INFO', logger_name='RMS', print_to_console=True)
        self.max_risk_per_bet = self.config_dict["risk_management"]["max_risk_per_bet"]
        self.orders = self.load_orders_from_db()
        self.margin_available = self.update_all_margin_available()
        # self.get_strategy_portfolio()

    def load_orders_from_db(self):
        if os.path.exists('db/vault/orders.json'):
            with open('db/vault/orders.json') as file:
                self.orders = json.load(file)
            self.logger.debug(f"Loaded {len(self.orders)} orders from the database.")
        else:
            self.orders = []
        
        return self.orders
    
    def update_all_margin_available(self):
        self.margin_available = {}
        if 'total' not in self.margin_available:
            self.margin_available['total'] = {}
        self.margin_available['total']['total_margin_available'] = self.config_dict["oms"]["funds_available"] * (1 - self.config_dict["risk_management"]["margin_reserve_pct"])
        self.margin_available['total']['current_margin_available'] = self.config_dict["oms"]["funds_available"] * (1 - self.config_dict["risk_management"]["margin_reserve_pct"])
        
        for strategy_name in self.config_dict["strategies"]:
            if strategy_name not in self.margin_available:
                self.margin_available[strategy_name] = {}
            self.margin_available[strategy_name]['total_margin_available'] = self.get_strategy_margin_available(strategy_name)
            self.margin_available[strategy_name]['current_margin_available'] = self.get_strategy_margin_available(strategy_name)
        return self.margin_available
    
    def get_current_portfolio(self, strategy_name):
        if strategy_name in self.config_dict["oms"]["portfolio"]:
            self.current_portfolio = self.config_dict["oms"]["portfolio"][strategy_name]
        else:
            self.current_portfolio = {}
        
        return self.current_portfolio
    
    def get_strategy_margin_available(self, strategy_name):
        num_of_strategy_count = len(self.config_dict["strategies"])
        strategy_margin = self.margin_available['total']['total_margin_available'] / num_of_strategy_count
        
        return strategy_margin
        
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

    def RMS_check_available_margin(self, signal_list):
        status = signal_list[-1]["status"]
        if status != 'rejected':
            strategy_name = signal_list[-1]["strategy_name"]
            current_price = signal_list[-1]["symbol_ltp"]
            order_qty = signal_list[-1]["orderQuantity"]      
            
            margin_available = self.margin_available[strategy_name]['current_margin_available']
            margin_required = current_price * abs(order_qty)
            
            if(margin_required > margin_available):
                msg = "RMS Test Failed: not enough margin. Available margin: ${}, required margin: ${}".format(margin_available, margin_required)
                signal = signal_list[-1]
                signal_new = signal.copy()
                signal_new['status'] = 'rejected'
                signal_new['signal_update_by'] = 'RMS'
                signal_new['signal_update_reason'] = msg
                signal_list.append(signal_new)
                
        return signal_list
            
    def RMS_check_max_risk(self, signal_list):
        status = signal_list[-1]["status"]
        if status != 'rejected':
            strategy_name = signal_list[-1]["strategy_name"]
            current_price = signal_list[-1]["symbol_ltp"]
            sl_price = signal_list[-1]["stoploss_abs"]
            order_qty = signal_list[-1]["orderQuantity"]
            
            if strategy_name in self.margin_available:
                total_strategy_margin = self.margin_available[strategy_name]['total_margin_available']
            else:
                total_strategy_margin = 0

            max_risk = self.max_risk_per_bet * total_strategy_margin
            order_risk = (current_price - sl_price) * abs(order_qty)

            skip_order = False
            if(order_risk > max_risk):
                msg = "RMS Test Failed: exceeded max risk of ${}, order_risk was calulated at: ${}".format(max_risk, round(order_risk, 2))
                signal = signal_list[-1]
                signal_new = signal.copy()
                signal_new['status'] = 'rejected'
                signal_new['signal_update_by'] = 'RMS'
                signal_new['signal_update_reason'] = msg
                signal_list.append(signal_new)
            
        return signal_list
    
    def get_order(self, signal_list):
        # self.logger.debug({'signal_list':signal_list[-1]})
        order_list = []
        signal = signal_list[-1]
        status = signal["status"]
        if status != 'rejected':
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
        self.logger.debug({'order_list':order_list})
        return order_list
    
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
    
    def RMS_check_signal(self, signal_list):
        # finding stop loss points
        if(signal_list[-1]["exit_order_type"] == "stoploss_pct"):
            signal_list = self.calculate_sl_price(signal_list)
        # self.logger.debug({'signal_list':signal_list})
        #checking whether the order risk is below the max risk
        signal_list = self.RMS_check_max_risk(signal_list)
        # self.logger.debug({'signal_list':signal_list})
        
        #checking if there is enough available funds for the order
        signal_list = self.RMS_check_available_margin(signal_list)
        # self.logger.debug({'signal_list-RMS_check_signal':signal_list})
        
        return signal_list

    def ideal_portfolio_to_signals(self, ideal_portfolio_entry):
        ideal_portfolio = ideal_portfolio_entry["ideal_portfolio"]
        # self.logger.debug({'ideal_portfolio':ideal_portfolio})
        # weighted_ltp = sum([abs(ideal_portfolio[i][['signal_strength']])*ideal_portfolio[i]['current_price'] for i in ideal_portfolio.keys()])

        total_weighted_ltp = sum(abs(ideal_portfolio[symbol]['signal_strength']) for symbol in ideal_portfolio)
        
        # Normalize the weights
        for symbol in ideal_portfolio:
            ideal_portfolio[symbol]['signal_strength'] = ideal_portfolio[symbol]['signal_strength'] / total_weighted_ltp
        
        #checking to see if the margin req for ideal portfolio is under the available margin
        strategy_name = ideal_portfolio_entry["strategy_name"]
        if strategy_name in self.margin_available:
            total_strategy_margin = self.margin_available[strategy_name]['total_margin_available']
        else:
            total_strategy_margin = 0
        signals = []
        
        # Now from ideal portfolio, create ideal_portfolio with absolute positions
        ideal_portfolio_abs_positions = {}
        for symbol in ideal_portfolio:
            fund_available_to_symbol = total_strategy_margin * ideal_portfolio[symbol]['signal_strength']
            current_price = ideal_portfolio[symbol]['current_price']
            order_qty = int(fund_available_to_symbol / current_price)
            orderDirection = ideal_portfolio[symbol]['orderDirection']
            ideal_portfolio_abs_positions[symbol] = {'orderDirection': orderDirection, 'orderQuantity': order_qty, 'current_price':current_price}
            
        # Now compare it to the current ideal portfolio for this strategy and find the delta & Create the signals
        current_portfolio = self.get_current_portfolio(strategy_name)
        for symbol in ideal_portfolio_abs_positions:
            if symbol in current_portfolio:
                orderQuantity_delta = ideal_portfolio_abs_positions[symbol]['orderQuantity'] - current_portfolio[symbol]['orderQuantity']
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
                    "timeInForce" : ideal_portfolio_entry["timeInForce"] , 
                    "orderQuantity" : abs(order_qty),
                    "orderDirection": orderDirection,
                    "granularity": ideal_portfolio_entry["granularity"],
                    'status': 'pending'
                    }
            signals.append([signal])
        return signals

    def create_order(self, signal_list):
        order_list = []
        # self.logger.debug({'signal_listBEFORE':signal_list})
        signal_list = self.RMS_check_signal(signal_list)
        # self.logger.debug({'signal_list-AFTER':signal_list})
        status = signal_list[-1]["status"]
        if status != 'rejected':
            order_list = self.get_order(signal_list)
            #     symbol = signal_list[-1]["symbol"]
            #     order_qty = signal_list[-1]["orderQuantity"]
            # if symbol in self.current_portfolio:
            #     self.current_portfolio[symbol] += order_qty
            # else:
            #     self.current_portfolio[symbol] = order_qty
        else:
            self.logger.debug({'REJECTED: signal_list':signal_list[-1]})
        
        order_list_new = []
        for order in order_list:
            order_list_new.append(order)
        signal_list[-1]["status"] = 'completed'
        
        return signal_list, order_list_new
        
    def convert_signals_to_orders(self, new_signals):
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
            all_new_signals +=  self.ideal_portfolio_to_signals(ideal_portfolio_entry)

        # Convert all signals to orders
        for count, signal_list in enumerate(all_new_signals):
            signal_list, new_order_list = self.create_order(signal_list)
            # Update the signal_list received.
            all_new_signals[count] = signal_list
            new_orders.append(new_order_list)
            
        if len(new_orders) > 0:
            self.orders.extend(new_order_list)
            self.logger.debug(f'NOTE: Saving new recieved signals and portfolios to a json has not yet been implemented. plz implement soon at this point.')
            self.logger.debug(f'NOTE: Saving new recieved orders to a json has not yet been implemented. plz implement soon at this point.')
        
        return new_orders