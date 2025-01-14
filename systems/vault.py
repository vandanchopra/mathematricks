import json, os
from systems.utils import create_logger, sleeper

HARDCODED_DATA_DIR = "/mnt/VANDAN_DISK/code_stuff/projects/mathematricks_gagan/db/data/ibkr/1d"

class Vault:
    def __init__(self, config_dict, market_data_extractor):
        self.tickers_dict = {}
        self.logger = create_logger(log_level='DEBUG', logger_name='Vault', print_to_console=True)
        self.strategies = self.load_strategies(config_dict)
        self.config_dict = self.create_datafeeder_config(config_dict, self.strategies)
        self.market_data_extractor = market_data_extractor

    def load_strategies(self, config_dict):
        from vault.pairs_trading import ConfigManager
        config_manager = ConfigManager(config_dict)
        
        strategy_names = config_dict['strategies']
        strategies_dict = {}
        for strategy in strategy_names:
            # import strategy module and get the class
            module = __import__('vault.{}'.format(strategy), fromlist=[strategy])
            # Try to get the strategy class with both naming conventions
            strategy_name = strategy.split('.')[-1]
            # Convert snake_case to CamelCase for strategy class name
            strategy_class_name = ''.join(word.title() for word in strategy_name.split('_'))
            strategy_class = getattr(module, strategy_class_name, None) or getattr(module, strategy_class_name + 'Strategy')
            # Create market data extractor if not already created
            if not hasattr(self, 'market_data_extractor'):
                from systems.utils import MarketDataExtractor
                self.market_data_extractor = MarketDataExtractor()
            
            # Create data handler instance with proper configuration
            from vault.pairs_trading import DataHandler
            data_dir = config_dict.get('data_update_inputs', {}).get('data_paths', {}).get('ibkr', HARDCODED_DATA_DIR)
            
            # Validate data directory exists
            if not os.path.exists(data_dir):
                raise ValueError(f"Data directory does not exist: {data_dir}")
                
            # Get additional configuration from config_dict
            data_handler_config = config_dict.get('data_handler_config', {})
            
            data_handler = DataHandler(
                data_dir=data_dir,
                tickers=[],
                data_frequency=data_handler_config.get('data_frequency', 'D'),
                timezone=data_handler_config.get('timezone', 'UTC'),
                max_data_points=data_handler_config.get('max_data_points', 1000),
                cache_size=data_handler_config.get('cache_size', 100),
                logger=self.logger
            )
            
            self.logger.info(f"Initialized DataHandler for strategy {strategy_name} with config: {data_handler_config}")
            
            strategies_dict[strategy] = strategy_class(config_manager, data_handler)
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
        for strategy_name, strategy in strategies.items():
            inputs = strategy.datafeeder_inputs()
            if isinstance(inputs, tuple):
                data_input_temp, list_of_symbols_temp = inputs
                self.tickers_dict[strategy_name] = list_of_symbols_temp
                data_inputs = data_inputs | data_input_temp
            else:
                # Handle dictionary return type from pairs_trading
                data_inputs = data_inputs | inputs
                self.tickers_dict[strategy_name] = inputs.get('symbols', [])

        # self.logger.debug({'self.tickers_dict':self.tickers_dict})
        list_of_symbols = list(set([ticker for ticker_list in self.tickers_dict.values() for ticker in ticker_list]))
        
        # Convert all the columns to lowercase before returning the datafeeder_config
        datafeeder_config = {'data_inputs':to_lowercase(data_inputs), 'list_of_symbols':list_of_symbols}
        config_dict["datafeeder_config"] = datafeeder_config
        
        return config_dict
        
    def generate_signals(self, next_rows, market_data_df, system_timestamp, oms_open_orders):
        signals_output = {'signals':[], 'ideal_portfolios':[]}
        ''' 
        for each strategy in self.strategies, get the signals and ideal portfolio.
        combine the signals and ideal portfolio from all strategies and return the combined signals.
        '''
        for strategy in self.strategies.values():
            return_type, return_item, list_of_symbols_temp = strategy.generate_signals(next_rows, market_data_df, system_timestamp, oms_open_orders)
            self.tickers_dict[strategy.strategy_name] = list_of_symbols_temp
            if return_type == 'signals':
                for signal in return_item:
                    signals_output["signals"].append(signal)
            if return_type == 'ideal_portfolios':
                for ideal_portfolio in return_item:
                    signals_output["ideal_portfolios"].append(ideal_portfolio)
                
        # Update the config_dict with the latest list of symbols from all the strategies
        list_of_symbols = list(set([ticker for ticker_list in self.tickers_dict.values() for ticker in ticker_list]))
        if list_of_symbols != self.config_dict['datafeeder_config']['list_of_symbols']:
            self.config_dict['datafeeder_config']['list_of_symbols'] = list_of_symbols
            # self.logger.info(f'Updated list of symbols: {list_of_symbols}')
            # sleeper(5, 'Sleeping for 5 seconds to update the list of symbols')
        
        return signals_output, self.config_dict

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
    from systems.utils import MarketDataExtractor
    market_data_extractor = MarketDataExtractor()
    try:
        from vault.pairs_trading import ConfigManager
        config_manager = ConfigManager(config_dict)
        # Initialize RMS with proper error handling
        rms_instance = RMS(config_manager, market_data_extractor)
        
        # Initialize required parameters for convert_signals_to_orders
        margin_available = {
            'sim': {
                'sim_1': {
                    'CAD': {
                        'total_buying_power': 100000,
                        'buying_power_available': 100000
                    }
                }
            }
        }
        open_orders = []
        system_timestamp = pd.Timestamp(datetime.now())
        live_bool = False
        
        # Validate signals structure
        if not isinstance(signals_output, dict) or 'signals' not in signals_output:
            raise ValueError("Invalid signals structure - missing 'signals' key")
            
        # Add required fields if missing
        for signal in signals_output['signals']:
            if 'symbol_ltp' not in signal:
                signal['symbol_ltp'] = {str(system_timestamp): 0.0}
            if 'timestamp' not in signal:
                signal['timestamp'] = system_timestamp
            if 'strategy_name' not in signal:
                signal['strategy_name'] = 'pairs_trading'
                
        orders = rms_instance.convert_signals_to_orders(
            signals_output,
            margin_available,
            open_orders,
            system_timestamp,
            live_bool
        )
    except Exception as e:
        logger.error(f"Error in RMS processing: {str(e)}")
        orders = []
    logger.debug({'orders_count':len(orders)})
    logger.debug({'orders':orders})
    
    '''
    1) Make ideal portfolio also work. Need to check current porfolio, and only place the orders for the delta.
    2) Check when and where jsons need to be saved. And when and where they need to be loaded.
    3) Also, need to check the current portfolio and orders jsons (Do we need to do this here on let OMS handle this?) :: We need to think about the scenario where funds are distributed among strategies.
    '''