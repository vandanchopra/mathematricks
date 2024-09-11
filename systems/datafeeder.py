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
        self.system_timestamp = None
        self.sleep_lookup = {"1m":60,"2m":120,"5m":300,"1d":86400}
        self.market_data_df = None

    def next(self, market_data_df, run_mode, sleep_time=0):
        # update data and return the updated data
        interval_inputs = self.config_dict['data_inputs']

        if run_mode == "BT":
            return pd.DataFrame()
        
        if run_mode == 'LIVE':
            # use datafetcher to update the data
            if self.market_data_df is None:
                self.market_data_df = self.datafetcher.fetch_updated_price_data(market_data_df)

            if self.system_timestamp is None:
                self.system_timestamp = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()])
            else:
                self.market_data_df = self.market_data_df.loc[self.market_data_df.index.get_level_values(1) >= self.system_timestamp,:]

        first_rows = []
        for interval in self.market_data_df.index.get_level_values(0).unique():
            first_row = self.market_data_df.loc[interval,:].iloc[0]
            first_df = pd.DataFrame(first_row).T
            first_df.index.names = ['Datetime']
            if first_df.index[0] == self.system_timestamp:
                self.system_timestamp = first_df.index[0]
                first_df.reset_index(drop=False,inplace=True)
                first_df['interval'] = interval
                first_df.set_index(['interval','Datetime'],inplace=True)
                first_rows.append(first_df)

                # remove the first row from the DataFrame
                self.market_data_df = self.market_data_df.drop((interval,first_row.name))

        first_rows = pd.concat(first_rows)

        if len(self.market_data_df) < 1:
            sleep_time = min([self.sleep_lookup[interval] for interval in interval_inputs])
            self.market_data_df = None
            time.sleep(sleep_time-10)
        else:
            self.system_timestamp = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()])

        return first_rows