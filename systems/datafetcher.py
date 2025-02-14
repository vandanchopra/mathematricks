import time
from systems.utils import create_logger, sleeper
import logging
from brokers.brokers import Brokers
import sys
import pandas as pd

class DataFetcher:
    def __init__(self, config_dict, market_data_extractor):
        self.config_dict = config_dict
        self.market_data_extractor = market_data_extractor
        self.logger = create_logger(log_level='DEBUG', logger_name='datafetcher', print_to_console=True)
        self.broker = Brokers()
    
    def fetch_updated_price_data(self, start_date, end_date, lookback, throttle_secs=1, update_data=True, run_mode=4, live_bool=False):
        list_of_symbols = self.config_dict['datafeeder_config']['list_of_symbols']
        interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
        # self.logger.debug({'interval_inputs': interval_inputs})
        # self.logger.debug({'list_of_symbols': list_of_symbols})
        data_sources = self.config_dict['data_update_inputs']['data_sources']
        data_source =  data_sources['live'] if live_bool else data_sources['sim']
        # self.logger.debug({'data_source':data_source, 'data_sources':data_sources})
        # sleeper(10, 'Giving you time to read the above Message 2')

        if data_source == 'yahoo':
            market_data_df = self.broker.sim.data.update_price_data(list_of_symbols, interval_inputs=interval_inputs, throttle_secs=throttle_secs, start_date=start_date, end_date=end_date, lookback=lookback, update_data=update_data, run_mode=run_mode)
            
        elif data_source == 'ibkr':
            market_data_df = self.broker.ib.data.update_price_data(list_of_symbols, interval_inputs=interval_inputs, throttle_secs=throttle_secs, start_date=start_date, end_date=end_date, lookback=lookback, update_data=update_data, run_mode=run_mode) 
        else:
            raise ValueError(f'Invalid data source: {data_source}')
        
        # Forward fill the data
        market_data_df = market_data_df.fillna(method='ffill')
        market_data_df = market_data_df.fillna(method='bfill')
        self.logger.warning('We are not maintaining the matrix for ffill and bfill. This needs to be implemented')
        
        return market_data_df
        
if __name__ == '__main__':
    from config import config_dict
    datafetcher = DataFetcher(config_dict=config_dict)
    asset_data = datafetcher.fetch_updated_price_data_single_asset('AAPL')