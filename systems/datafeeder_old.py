'''
datafeeder will run in 3 modes:
Live: Update data, and return updated data
Backtest: return historical data until the current test system date

'''
import datetime
from pdb import run
import pytz
from systems.datafetcher import DataFetcher
import time
from systems.utils import create_logger, sleeper
import logging
import pandas as pd
from systems.indicators import Indicators
import pandas_market_calendars as mcal

class DataFeeder:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.datafetcher = DataFetcher(self.config_dict)
        self.logger = create_logger(log_level='INFO', logger_name='datafeeder')
        self.next_system_timestamp = None
        self.sleep_lookup = {"1m":60,"2m":120,"5m":300,"1d":86400}
        self.market_data_df = None
        self.indicators = Indicators()
        self.lookback_dict = self.create_lookback_dict()
    
    def create_lookback_dict(self):
        self.lookback_dict = {}
        for interval in self.config_dict['datafeeder_config']['data_inputs']:
            lookback = self.config_dict['datafeeder_config']['data_inputs'][interval]['lookback']
            self.lookback_dict[interval] = lookback

    def is_market_open(self, current_datetime):
        # Define market open and close times
        MARKET_OPEN_TIME = datetime.time(9, 30)  # 9:30 AM
        MARKET_CLOSE_TIME = datetime.time(16, 0)  # 4:00 PM
        """
        Check if the market is open at the given datetime.
        
        Parameters:
        current_datetime (datetime): The datetime to check.
        
        Returns:
        bool: True if the market is open, False otherwise.
        """
        # convert current_datetime to 'eastern' timezone
        current_datetime = current_datetime.astimezone(pytz.timezone('US/Eastern'))
        current_time = current_datetime.time()
        return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

    def next_market_open(self, current_datetime):
        
        # convert current_datetime to 'eastern' timezone
        current_datetime = current_datetime.astimezone(pytz.timezone('US/Eastern'))
        # self.logger.debug({'current_datetime':current_datetime})
        
        # Define market open and close times
        MARKET_OPEN_TIME = datetime.time(9, 30)  # 9:30 AM
        MARKET_CLOSE_TIME = datetime.time(16, 0)  # 4:00 PM
        """
        Get the next market open datetime from the given datetime.
        
        Parameters:
        current_datetime (datetime): The datetime to check from.
        
        Returns:
        datetime: The next market open datetime.
        """
        current_time = current_datetime.time()
        if current_time < MARKET_OPEN_TIME:
            # If current time is before market opens today
            next_open = current_datetime.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
        elif current_time > MARKET_CLOSE_TIME:
            # If current time is after market closes today
            next_open = (current_datetime + datetime.timedelta(days=1)).replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
        else:
            # If current time is during market hours
            next_open = current_datetime.replace(hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0)
        
        return next_open

    def previous_market_close(self, current_datetime):
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=current_datetime - datetime.timedelta(days=30), end_date=current_datetime)
        previous_close = schedule.iloc[-1]['market_close']
        return previous_close    
    
    def calculate_sleep(self):
        interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
        sleep_time_old = min([self.sleep_lookup[interval] for interval in interval_inputs])
        now = datetime.datetime.now()
        now_tz = now.astimezone(pytz.timezone('US/Eastern'))
        self.market_data_loaded_df = None
        passed_time = now.astimezone(pytz.utc) - self.next_system_timestamp
        # self.logger.debug({'passed_time':passed_time, 'system_timestamp':self.system_timestamp, 'sleep_time_old':sleep_time_old})
        
        if self.is_market_open(now_tz):
            sleep_time = sleep_time_old-passed_time.seconds
            if sleep_time < 0:
                sleep_time = sleep_time_old
        else:
            # calculate the number of seconds until the next market open
            next_open = self.next_market_open(now_tz)
            # self.logger.debug({'next_open':next_open, 'now':now_tz})
            sleep_time = int((next_open - now_tz).total_seconds())
            # self.logger.debug({'next_open':next_open, 'sleep_time':sleep_time})
        return sleep_time
    
    def next(self, market_data_df_root, run_mode, start_date,end_date, sleep_time=0,):
        # update data and return the updated data
        # self.logger.debug({'market_data_df':market_data_df})

        # if run_mode == 'LIVE':
        #     # start date is the current date
        #     now = datetime.datetime.now()
        #     now_tz = now.astimezone(pytz.timezone('US/Eastern'))
        #     start_date = now_tz
        #     end_date = None
        
        # Fix the start_date and end_date to account for the lookback period.            
            
        if self.market_data_df is None or self.next_system_timestamp is None:
            self.market_data_df = self.datafetcher.fetch_updated_price_data(market_data_df_root, start_date=start_date, end_date=end_date, lookback=self.lookback_dict)
            # self.logger.info({'combined_df':self.market_data_df[-3:]})
        
        if self.next_system_timestamp is None:
            self.next_system_timestamp = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()])
        else:
            self.market_data_df = self.market_data_df.loc[self.market_data_df.index.get_level_values(1) >= self.next_system_timestamp,:]

        first_rows = []
        for interval in self.market_data_df.index.get_level_values(0).unique():
            first_row = self.market_data_df.loc[interval,:].iloc[0]
            first_df = pd.DataFrame(first_row).T
            first_df.index.names = ['datetime']
            if first_df.index[0] == self.next_system_timestamp:
                self.next_system_timestamp = first_df.index[0]
                first_df.reset_index(drop=False,inplace=True)
                first_df['interval'] = interval
                first_df.set_index(['interval','datetime'],inplace=True)
                first_rows.append(first_df)

                # remove the first row from the DataFrame
                self.market_data_df = self.market_data_df.drop((interval, first_row.name))

        if len(first_rows) > 0:
            first_rows = pd.concat(first_rows)
            self.next_system_timestamp = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()])
        else:
            first_rows = None
            self.next_system_timestamp = None
        
        if len(self.market_data_df) < 1 and run_mode in [1,2]:
            sleep_time = self.calculate_sleep()
            sleeper(sleep_time, 'System Sleeping: Time to next timestamp')
        
        return first_rows