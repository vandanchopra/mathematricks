from calendar import c
from pdb import run
from turtle import update
from config import config_dict
# from brokers import Brokers
from systems.datafetcher import DataFetcher
from systems.datafeeder import DataFeeder
from systems.vault import Vault, RMS
import pandas as pd
from systems.utils import create_logger
import logging
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self):
        self.sleep_time = config_dict['sleep_time']
        self.market_data_df = pd.DataFrame()
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
        self.vault = Vault(config_dict)
        self.rms = RMS()
        self.datafeeder = DataFeeder(self.vault.datafeeder_config)
        self.datafetcher = DataFetcher(self.vault.datafeeder_config)
    
    def run_live_real_money(self):
        x = 0
        while True:
            try:
                # Start with getting the next timestamp of data
                next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='LIVE', sleep_time=self.sleep_time)     
                self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                for interval, datetime in next_rows.index:
                    self.logger.debug(f"Interval: {interval}, Datetime: {datetime}")
                # execute_signals(signals)
                signals_output = self.vault.generate_signals(self.market_data_df)
                # Convert signals to orders
                orders = self.rms.convert_signals_to_orders(signals_output)
                    
            except KeyboardInterrupt:
                self.logger.debug({'self.market_data_df':self.market_data_df})
                self.logger.debug('Exiting...')
                break
    
    def run_live_paper_money(self):
        while True:
            try:
                # Start with getting the next timestamp of data
                next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='LIVE', sleep_time=self.sleep_time)     
                self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                for interval, datetime in next_rows.index:
                    self.logger.debug(f"Interval: {interval}, Datetime: {datetime}")
                # execute_signals(signals)
                signals_output = self.vault.generate_signals(self.market_data_df)
                # Convert signals to orders
                orders = self.rms.convert_signals_to_orders(signals_output)
                
            except KeyboardInterrupt:
                print ('Exiting...')
                break
    
    def run_backtest(self,start_date,end_date):
        while True:
            try:
                next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='LIVE', sleep_time=self.sleep_time,start_date=start_date,end_date=end_date)     
                self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                self.logger.debug({f'Back Test: next rows': next_rows})
                # execute_signals(signals)
                signals_output = self.vault.generate_signals(self.market_data_df)
                # Convert signals to orders
                orders = self.rms.convert_signals_to_orders(signals_output)

                if self.datafeeder.market_data_df is None:
                    break
                
            except KeyboardInterrupt:
                print ('Exiting...')
    
    def run_data_update(self):
        self.market_data_df = self.datafetcher.fetch_updated_price_data(self.market_data_df)
    
    def run(self):
        run_mode = config_dict['run_mode']
        if run_mode == 1: # live trading - real money
            print ('live trading - real money')
            self.run_live_real_money()
        elif run_mode == 2: # live trading - paper money
            print ('live trading - paper money')
            self.run_live_paper_money()
        elif run_mode == 3: # backtesting
            start_time = pd.Timestamp(datetime(2024,9,12)).tz_localize('UTC').tz_convert('EST')
            end_time = pd.Timestamp(datetime(2024,9,15)).tz_localize('UTC').tz_convert('EST')
            self.run_backtest(start_time,end_time)

        elif run_mode == 4: # data update only
            self.logger.debug('len-stock_symbols: {}'.format(len(config_dict['list_of_symbols'])))
            self.logger.debug({'BEFORE: market_data_df': self.market_data_df.shape})
            self.run_data_update()
            self.logger.debug({'AFTER: market_data_df': self.market_data_df.shape})
            self.logger.debug({'market_data_df': self.market_data_df})
        else:
            raise ValueError('Invalid run_mode value: {}'.format(run_mode))

if __name__ == '__main__':
    Mathematricks().run()
    
    
    
    