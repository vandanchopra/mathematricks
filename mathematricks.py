from calendar import c
from pdb import run
from turtle import update
from config import config_dict
# from brokers import Brokers
from systems.datafetcher import DataFetcher
from systems.datafeeder import DataFeeder
import pandas as pd
from utils import create_logger
import logging
import sys

'''
write the software for AAPL, MSFT only.
'''

class Mathematricks:
    def __init__(self):
        self.datafeeder = DataFeeder(config_dict)
        self.datafetcher = DataFetcher(config_dict)
        self.sleep_time = config_dict['sleep_time']
        self.market_data_df = pd.DataFrame()
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
    
    def run_live_real_money(self):
        while True:
            try:
                next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='LIVE', sleep_time=self.sleep_time)     
                self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                self.logger.debug({f'LIVE: data frame': self.market_data_df})
                # signals = generate_signals(data)
                # execute_signals(signals)
            except KeyboardInterrupt:
                print ('Exiting...')
                break
    
    def run_live_paper_money(self):
        while True:
            try:
                next_rows = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='LIVE', sleep_time=self.sleep_time)     
                self.market_data_df = pd.concat([self.market_data_df, next_rows], axis=0)
                self.market_data_df = self.market_data_df[~self.market_data_df.index.duplicated(keep='last')]
                self.logger.debug({f'LIVE: data frame': self.market_data_df})
                # signals = generate_signals(data)
                # execute_signals(signals)
            except KeyboardInterrupt:
                print ('Exiting...')
                break
    
    def run_backtest(self):
        try:
            market_data_df = self.datafeeder.next(market_data_df=self.market_data_df, run_mode='BT', sleep_time=self.sleep_time)
            # signals = generate_signals(data)
            # execute_signals(signals)
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
            self.run_backtest()
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
    
    
    
    