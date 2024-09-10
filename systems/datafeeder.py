'''
datafeeder will run in 3 modes:
Live: Update data, and return updated data
Backtest: return historical data until the current test system date

'''
from systems.datafetcher import DataFetcher
import time
from utils import create_logger
import logging
import pandas as pd

class DataFeeder:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.datafetcher = DataFetcher(self.config_dict)
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafeeder')
    
    def next(self, market_data_df, run_mode, sleep_time=0):
        # update data and return the updated data
        if run_mode == "BT":
            return pd.DataFrame()

        if run_mode == 'LIVE':
            # use datafetcher to update the data
            market_data_df = self.datafetcher.fetch_updated_price_data(market_data_df)
            
        # while not market_data_df.empty:
        #     # extract the last row of the DataFrame and print it
        interval_inputs = self.config_dict['data_inputs']
        last_rows = []
        for interval in interval_inputs:
            last_row = market_data_df.loc[interval,:].iloc[-1]
            last_df = pd.DataFrame(last_row).T
            last_df.index.names = ['Datetime']
            last_df['interval'] = interval
            last_df.reset_index(drop=False,inplace=True)
            last_df.set_index(['interval','Datetime'],inplace=True)
            last_rows.append(last_df)

            # remove the last row from the DataFrame
            market_data_df = market_data_df.drop((interval,last_row.name))
            
        last_rows = pd.concat(last_rows)
        return last_rows