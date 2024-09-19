import sys, os, json, time, logging
import pandas as pd
from calendar import c
from pdb import run
from turtle import update
from config import config_dict
from systems.datafetcher import DataFetcher
from systems.datafeeder import DataFeeder
from systems.vault import Vault, RMS
from systems.utils import create_logger
import datetime, pytz
import warnings
warnings.filterwarnings("ignore")

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.sleep_time = self.config_dict['sleep_time']
        self.market_data_df = pd.DataFrame()
        self.logger = create_logger(log_level=self.config_dict['log_level'], logger_name='datafetcher', print_to_console=True)
        self.vault = Vault(self.config_dict)
        self.config_dict = self.vault.config_dict
        self.rms = RMS(self.config_dict)
        self.datafeeder = DataFeeder(self.config_dict)
        self.datafetcher = DataFetcher(self.config_dict)
        self.logger.debug({'config_dict': self.config_dict})
    
    def run(self):
        run_mode = config_dict['run_mode']
        if run_mode in [1,2,3]:
            while True:
                try:
                    if run_mode in [1,2]: # live trading - real money
                        start_date = datetime.datetime.now(datetime.timezone.utc).astimezone(pytz.timezone('US/Eastern'))  # Convert to Eastern Time
                        end_date = None
                        run_mode = 'LIVE'
                    elif run_mode == 3: # backtesting
                        assert 'start_time' in self.config_dict['backtest_inputs'] and 'end_time' in self.config_dict['backtest_inputs'], 'start_time and end_time must be provided in backtest_inputs if run_mode is 3'
                        start_date = self.config_dict['backtest_inputs']['start_time']
                        end_date = self.config_dict['backtest_inputs']['end_time']
                        run_mode = 'BT'
                        
                    next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode=run_mode, sleep_time=self.sleep_time, start_date=start_date, end_date=end_date)
                    if next_rows is not None:
                        self.system_timestamp = next_rows.index.get_level_values(1)[-1]
                        self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                        self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                        for interval, next_datetime in next_rows.index:
                            # self.logger.debug(f"Interval: {interval}, Datetime: {next_datetime}, system_timestamp: {self.system_timestamp}")
                            # self.logger.debug(next_rows)
                            self.logger.debug({'self.market_data_df':self.market_data_df.shape})
                            time.sleep(0.25)
                        # execute_signals(signals)
                        signals_output = self.vault.generate_signals(self.market_data_df)
                        self.logger.debug({'system_timestamp':self.system_timestamp, 'signals_output':signals_output})
                        # # Convert signals to orders
                        # orders = self.rms.convert_signals_to_orders(signals_output)
                    else:
                        print('Backtest completed.')
                        break
                    # time.sleep(0.25)
                
                except KeyboardInterrupt:
                    # self.logger.debug({'self.market_data_df':self.market_data_df})
                    self.logger.debug('Exiting...')
                    break
        elif run_mode == 4:
            self.logger.debug({'BEFORE: market_data_df': self.market_data_df.shape})
            stock_symbols_path = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/stock_symbols.json'
            with open(stock_symbols_path) as file:
                list_of_symbols = json.load(file)
            self.config_dict['datafeeder_config']['list_of_symbols'] = ['AAPL', 'MSFT'] # list_of_symbols
            self.market_data_df = self.datafetcher.fetch_updated_price_data(self.market_data_df)
            self.logger.debug({'AFTER: market_data_df': self.market_data_df.shape})
            # self.logger.debug({'market_data_df': self.market_data_df})
            
        else:
            raise AssertionError('Invalid run_mode value: {}'.format(run_mode))
        
if __name__ == '__main__':
    Mathematricks(config_dict).run()
    
    
    
    