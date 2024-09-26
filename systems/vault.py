import json, os
from systems.utils import create_logger

class Vault:
    def __init__(self, config_dict):
        self.strategies = self.load_strategies(config_dict['strategies'])
        self.config_dict = self.create_datafeeder_config(config_dict, self.strategies)
        self.logger = create_logger(log_level=self.config_dict['log_level'], logger_name='Vault', print_to_console=True)

    def load_strategies(self, strategy_names):
        strategies_dict = {}
        for strategy in strategy_names:
            # import strategy module and get the class
            strategies_dict[strategy] = getattr(__import__('vault.{}'.format(strategy), fromlist=[strategy]), 'Strategy')()
        return strategies_dict
    
    def create_datafeeder_config(self, config_dict, strategies):
        def to_lowercase(d):
            if isinstance(d, dict):
                return {k.lower(): to_lowercase(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [to_lowercase(i) for i in d]
            elif isinstance(d, str):
                return d.lower()
            else:
                return d

        data_inputs = {}
        list_of_symbols = []
        for strategy in strategies.values():
            data_input_temp , list_of_symbols_temp = strategy.datafeeder_inputs()
            data_inputs = data_inputs | data_input_temp
            list_of_symbols += list_of_symbols_temp
        list_of_symbols = list(set(list_of_symbols))
        
        # Convert all the columns to lowercase before returning the datafeeder_config
        datafeeder_config = {'data_inputs':to_lowercase(data_inputs), 'list_of_symbols':list_of_symbols}
        config_dict["datafeeder_config"] = datafeeder_config
        
        return config_dict
        
    def generate_signals(self, market_data_df, system_timestamp):
        signals_output = {'signals':[], 'ideal_portfolios':[]}
        ''' 
        for each strategy in self.strategies, get the signals and ideal portfolio.
        combine the signals and ideal portfolio from all strategies and return the combined signals.
        '''
        for strategy in self.strategies.values():
            return_type, return_item = strategy.generate_signals(market_data_df, system_timestamp)
            if return_type == 'signals':
                for signal in return_item:
                    signals_output["signals"].append(signal)
            if return_type == 'ideal_portfolios':
                for ideal_portfolio in return_item:
                    signals_output["ideal_portfolios"].append(ideal_portfolio)
                
        # with open('db/vault/signals.json', 'w') as file:
        #     json.dump(signals_output, file)
        # run each strategy and get either the signals or ideal portforlio from each strategy, based on current data.
        # combine the signals from all strategies and return the combined signals.
        # self.logger.debug({'signals_output':signals_output})
        return signals_output

class RMS:
    def __init__(self, config_dict):
        #initializing constants from config dict
        self.config_dict = config_dict
        self.logger = create_logger(log_level=self.config_dict['log_level'], logger_name='RMS', print_to_console=True)
        self.max_risk_per_bet = self.config_dict["risk_management"]["max_risk_per_bet"]
        self.orders = self.load_orders_from_db()
        self.margin_available = self.update_all_margin_available()
        # self.get_strategy_portfolio()

    def load_orders_from_db(self):
        if os.path.exists('db/vault/orders.json'):
            with open('db/vault/orders.json') as file:
                self.orders = json.load(file)
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
            sl_price = signal_list[-1]["sl_abs"]
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
                order_leg = {'symbol': signal["symbol"], 
                    'timestamp': signal["timestamp"], 
                    "orderDirection": 'BUY' if signal["orderDirection"] == 'SELL' else 'SELL', 
                    "exitPrice":round(signal["sl_abs"], 2), 
                    "orderType":signal["exit_order_type"], 
                    "timeInForce": 'GTC' if signal["exit_order_type"] == 'stoploss_abs' else signal["timeInForce"],
                    "orderQuantity":abs(int(signal['orderQuantity'])),
                    "strategy_name":signal["strategy_name"],
                    "broker": 'IBKR' if self.config_dict['run_mode'] in [1,2] else 'SIM',
                    'granularity': signal['granularity'],
                    "status": 'pending'
                    }
                order_list.append(order_leg)
    
        return order_list
    
    def calculate_sl_price(self, signal_list):
        signal = signal_list[-1]
        signal_new = signal.copy()
        current_price = signal["symbol_ltp"]
        entry_orderDirection = signal["orderDirection"]
        signal_new["sl_abs"] = current_price * (1-signal["sl_pct"]) if entry_orderDirection == "BUY" else current_price * (1+signal["sl_pct"])
        signal_new['exit_order_type'] = 'stoploss_abs'
        signal_new['signal_update_by'] = 'RMS'
        signal_new['signal_update_reason'] = 'SL pct converted to SL abs'
        # Remove sl_pct from the signal
        signal_new.pop("sl_pct")
        
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
                
            signal = {"symbol": symbol, 
                    "strategy_name" : ideal_portfolio_entry["strategy_name"], 
                    "timestamp" : ideal_portfolio_entry["timestamp"], 
                    "entry_order_type" : ideal_portfolio_entry["entry_order_type"], 
                    "exit_order_type" : ideal_portfolio_entry["exit_order_type"], 
                    "sl_pct" : ideal_portfolio_entry["sl_pct"], 
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
            self.logger.warning(f'NOTE: Saving new recieved signals and portfolios to a json has not yet been implemented. plz implement soon at this point.')
            self.logger.warning(f'NOTE: Saving new recieved orders to a json has not yet been implemented. plz implement soon at this point.')
        
        return new_orders

if __name__ == '__main__':
    from config import config_dict
    import pandas as pd
    import logging
    import numpy as np
    from utils import create_logger
    logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
    
    # Delete the /Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/vault/vault.json file if it exists
    vault_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/vault/orders.json'
    if os.path.exists(vault_path):
        os.remove(vault_path)
    
    # vault = Vault(config_dict)
    # logger.debug({'datafeeder_config':vault.datafeeder_config})
    # Load dummy data and check if the signals are generated.
    # market_data_df = pd.DataFrame()
    # signals_output = vault.generate_signals(market_data_df)
    # signals_output = {'signals': [{'symbol': 'MSFT', 'signal_strength': 1, 'strategy_name': 'strategy_1', 'timestamp': '2020-08-11 00:00:00', 'entry_order_type': 'MARKET', 'exit_order_type': 'stoploss_pct', 'sl_pct': 0.2, 'symbol_ltp': np.float64(203.3800048828125), 'timeInForce': 'DAY', 'orderQuantity': 10, 'orderDirection': 'SELL'}], 'ideal_portfolios': []}
    signals_output = {'signals': [], 'ideal_portfolios': [{'strategy_name': 'strategy_2', 'timestamp': '2020-07-09 00:00:00', 'entry_order_type': 'MARKET', 'exit_order_type': 'stoploss_pct', 'sl_pct': 0.2, 'timeInForce': 'DAY', 'orderQuantity': 100, 'ideal_portfolio': {'NFLX': {'orderDirection': 'BUY', 'signal_strength': np.float64(0.3), 'current_price': np.float64(507.760009765625)}, 'NVDA': {'orderDirection': 'BUY', 'signal_strength': np.float64(0.33), 'current_price': np.float64(10.508999824523926)}, 'TSLA': {'orderDirection': 'BUY', 'signal_strength': np.float64(0.37), 'current_price': np.float64(92.9520034790039)}, 'HBNC': {'orderDirection': 'SELL', 'signal_strength': np.float64(0.28), 'current_price': np.float64(9.050000190734863)}, 'JPM': {'orderDirection': 'SELL', 'signal_strength': np.float64(0.35), 'current_price': np.float64(91.27999877929688)}, 'XOM': {'orderDirection': 'SELL', 'signal_strength': np.float64(0.36), 'current_price': np.float64(41.36000061035156)}}}]}
    
    logger.debug({'signals_output':signals_output['signals']})
    # logger.debug({'signals_output':signals_output})
    rms = RMS(config_dict)
    orders = rms.convert_signals_to_orders(signals_output)
    logger.debug({'orders_count':len(orders)})
    logger.debug({'orders':orders})
    
    '''
    1) Make ideal portfolio also work. Need to check current porfolio, and only place the orders for the delta.
    2) Check when and where jsons need to be saved. And when and where they need to be loaded.
    3) Also, need to check the current portfolio and orders jsons (Do we need to do this here on let OMS handle this?) :: We need to think about the scenario where funds are distributed among strategies.
    '''