import json, os
from systems.utils import create_logger

class Vault:
    def __init__(self, config_dict):
        self.strategies = self.load_strategies(config_dict['strategies'])
        self.config_dict = self.create_datafeeder_config(config_dict, self.strategies)
        self.logger = create_logger(log_level='INFO', logger_name='Vault', print_to_console=True)

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