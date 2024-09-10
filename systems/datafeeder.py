'''
datafeeder will run in 3 modes:
Live: Update data, and return updated data
Backtest: return historical data until the current test system date

'''
from systems.datafetcher import DataFetcher
import time
from systems.utils import create_logger
import logging

class DataFeeder:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.datafetcher = DataFetcher(self.config_dict)
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafeeder')
    
    def next(self, market_data_df, run_mode, sleep_time=0):
        # update data and return the updated data
        if run_mode == 'LIVE':
            # use datafetcher to update the data
            market_data_df = self.datafetcher.fetch_updated_price_data(market_data_df)
            
        while not market_data_df.empty:
            # extract the last row of the DataFrame and print it
            last_row = market_data_df.iloc[-1]
            self.logger.debug({'last_row':last_row.to_dict(), 'shape':market_data_df.shape})
            # remove the last row from the DataFrame
            market_data_df = market_data_df.drop(market_data_df.index[-1])
            time.sleep(sleep_time)
                
        return market_data_df