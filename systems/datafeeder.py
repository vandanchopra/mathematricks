'''
datafeeder will run in 3 modes:
Live: Update data, and return updated data
Backtest: return historical data until the current test system date

'''
import datetime
import pytz, os, time, logging

from traitlets import Int
from systems.datafetcher import DataFetcher
from systems.utils import create_logger, sleeper, load_symbols_universe_df, MarketDataExtractor
import pandas as pd
from systems.indicators import Indicators
import pandas_market_calendars as mcal
from copy import deepcopy
from systems.performance_reporter import PerformanceReporter


class DataFeeder:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.market_data_extractor = MarketDataExtractor()
        self.datafetcher = DataFetcher(self.config_dict, self.market_data_extractor)
        self.logger = create_logger(log_level='DEBUG', logger_name='datafeeder')
        self.sleep_lookup = {"1m":60,"2m":120,"5m":300,"1d":86400}
        self.market_data_df = None
        self.datafeeder_system_timestamp = None
        self.indicators = Indicators()
        self.lookback_dict_original = self.create_lookback_dict()
        self.lookback_dict = deepcopy(self.lookback_dict_original)
        self.first_run = True
        self.previous_config_dict = None
    
    def load_symbols_universe_df():
        def remove_old_files(stocksymbolslists_folder, days_old=30):    
            # Get file names from the folder
            file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
            # Get the number portion of the file names
            file_numbers = [float(f.split('_')[-1].split('.')[0]) for f in file_names]
            # Now assume that these numbers are EPOCH timestamps in milliseconds and calculate the age of the timestamp in days
            from datetime import datetime, timezone, timedelta
            now = datetime.now(timezone.utc)
            file_numbers = [datetime.fromtimestamp(x/1000, timezone.utc) for x in file_numbers]
            file_numbers = [now - x for x in file_numbers]
            file_numbers = [x.days for x in file_numbers]
            # if any of those are more than 30 days old, then delete them from file_name
            file_names = [f for f, age in zip(file_names, file_numbers) if age < days_old]
            return file_names
        
        # get file names from the folder and load all the csv files and concatenate them and return a pandas dataframe
        stocksymbolslists_folder = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/stocksymbolslists'
        
        # Get file names from the folder
        file_names = [f for f in os.listdir(stocksymbolslists_folder) if f.endswith('.csv')]
        # Remove all files that are more than 30 days old
        file_names = remove_old_files(stocksymbolslists_folder, days_old=90)
        # Load all the CSV files and concatenate them into a single DataFrame
        dfs = [pd.read_csv(os.path.join(stocksymbolslists_folder, file)) for file in file_names]
        symbols_universe_df = pd.concat(dfs, ignore_index=True)
        # Now sort the combined_df by the market cap column in descending order
        symbols_universe_df = symbols_universe_df.sort_values(by='Market Cap', ascending=False)
        # Drop all rows where Market Cap is NaN
        symbols_universe_df = symbols_universe_df.dropna(subset=['Market Cap'])
        # Drop all rows where Market Cap is 0
        symbols_universe_df = symbols_universe_df[symbols_universe_df['Market Cap'] != 0]
        
        return symbols_universe_df
    
    def create_lookback_dict(self):
        lookback_dict = {}
        if isinstance(self.config_dict, dict) and 'datafeeder_config' in self.config_dict:
            data_inputs = self.config_dict['datafeeder_config'].get('data_inputs', {})
            for interval, config in data_inputs.items():
                if isinstance(config, dict) and 'lookback' in config:
                    lookback_dict[interval] = config['lookback']
        return lookback_dict
    
    def reset_lookback_dict(self):
        lookback_dict = {}
        for interval in self.config_dict['datafeeder_config']['data_inputs']:
            lookback_dict[interval] = 0
        return lookback_dict

    def is_market_open(self, current_datetime):
        """
        Check if the market is open at the given datetime.

        Parameters:
        current_datetime (datetime): The datetime to check.

        Returns:
        bool: True if the market is open, False otherwise.
        """
        # convert current_datetime to 'UTC' timezone
        current_datetime = current_datetime.astimezone(pytz.timezone('UTC'))
        current_date = current_datetime.date()
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=current_datetime - datetime.timedelta(days=30), end_date=current_datetime + datetime.timedelta(days=30))
        current_date_str = current_date.strftime('%Y-%m-%d')
        if current_date_str in schedule.index:
            if current_datetime >= schedule.loc[current_date_str]['market_open'] and current_datetime <= schedule.loc[current_date_str]['market_close']:
                market_open_bool = True
            else:
                market_open_bool = False
        else:
            market_open_bool = False
        
        return market_open_bool

    def get_next_market_open(self, current_datetime):
        
        # convert current_datetime to 'eastern' timezone
        # self.logger.debug({'current_datetime':current_datetime})
        
        """
        Get the next market open datetime from the given datetime.
        
        Parameters:
        current_datetime (datetime): The datetime to check from.
        
        Returns:
        datetime: The next market open datetime.
        """
        current_datetime_EST = current_datetime.astimezone(pytz.timezone('US/Eastern'))
        current_date = current_datetime_EST.date()
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=current_datetime, end_date=current_datetime+ datetime.timedelta(days=5))
        # self.logger.debug({'schedule':schedule})
        current_date_str = current_date.strftime('%Y-%m-%d')
        next_market_open = schedule[schedule.index > current_date_str].iloc[0]['market_open']
        
        return next_market_open
    
    def get_prev_market_open(self, current_datetime):
        
        # convert current_datetime to 'eastern' timezone
        # self.logger.debug({'current_datetime':current_datetime})
        
        """
        Get the next market open datetime from the given datetime.
        
        Parameters:
        current_datetime (datetime): The datetime to check from.
        
        Returns:
        datetime: The next market open datetime.
        """
        current_datetime_EST = current_datetime.astimezone(pytz.timezone('US/Eastern'))
        current_date = current_datetime_EST.date()
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=current_datetime - datetime.timedelta(days=5), end_date=current_datetime)
        current_date_str = current_date.strftime('%Y-%m-%d')
        prev_market_open = schedule[schedule.index <= current_date_str].iloc[-1]['market_open']
        
        return prev_market_open

    def get_previous_market_close(self, current_datetime):
        # Get the NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        # Get the schedule for the past 30 days
        schedule = nyse.schedule(start_date=current_datetime - datetime.timedelta(days=30), end_date=current_datetime)
        # Ensure the 'market_close' column is timezone-aware
        schedule['market_close'] = schedule['market_close'].dt.tz_convert(pytz.UTC)
        # Filter out any future market close times
        schedule = schedule[schedule['market_close'] <= current_datetime]
        previous_close = schedule.iloc[-1]['market_close']
        return previous_close    
    
    def get_next_expected_timestamp(self, system_timestamp):
        interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
        if isinstance(interval_inputs, dict):
            # Handle case where data_inputs contains 'get_inputs' method
            if 'get_inputs' in interval_inputs:
                inputs = interval_inputs['get_inputs']()
                intervals = inputs.keys()
            else:
                intervals = interval_inputs.keys()
        else:
            intervals = []
        next_expected_timestamp_temp = min([self.sleep_lookup[interval] for interval in intervals]) if intervals else 60
        now = datetime.datetime.now().astimezone(pytz.timezone('UTC'))
        now_tz = now.astimezone(pytz.timezone('US/Eastern'))
        passed_time = now - system_timestamp
        
        # self.logger.debug({'passed_time':passed_time, 'system_timestamp':system_timestamp, 'next_expected_timestamp_temp':next_expected_timestamp_temp})
        # self.logger.debug({'is_market_open':self.is_market_open(now_tz)})
        
        if self.is_market_open(now_tz):
            next_expected_timestamp = next_expected_timestamp_temp-passed_time.seconds
            if next_expected_timestamp < 0:
                next_expected_timestamp = next_expected_timestamp_temp
                next_expected_timestamp = 0
        else:
            # calculate the number of seconds until the next market open
            next_open = self.get_next_market_open(now_tz)
            # self.logger.debug({'next_open':next_open, 'now':now_tz})
            next_expected_timestamp = int((next_open - now).total_seconds())
            # self.logger.debug({'next_open':next_open, 'sleep_time':sleep_time})
        return next_expected_timestamp
        
    def calculate_sleep(self):
        interval_inputs = self.config_dict['datafeeder_config']['data_inputs']
        # self.logger.debug({'interval_inputs':interval_inputs})
        sleep_time_temp = min([self.sleep_lookup[interval] for interval in interval_inputs])
        # self.logger.debug({'sleep_time_temp':sleep_time_temp})
        now = datetime.datetime.now()
        now_tz = now.astimezone(pytz.timezone('US/Eastern'))
        passed_time = now.astimezone(pytz.utc) - self.datafeeder_system_timestamp
        # self.logger.debug({'passed_time':passed_time, 'system_timestamp':self.datafeeder_system_timestamp, 'sleep_time_old':sleep_time_temp})
        
        if self.is_market_open(now_tz):
            sleep_time = sleep_time_temp-passed_time.seconds
            if sleep_time < 0:
                sleep_time = sleep_time_temp
        else:
            # calculate the number of seconds until the next market open
            next_open = self.get_next_market_open(now_tz)
            # self.logger.debug({'next_open':next_open, 'now':now_tz})
            sleep_time = int((next_open - now_tz).total_seconds())
            # self.logger.debug({'next_open':next_open, 'sleep_time':sleep_time})
        return sleep_time
    
    def update_all_historical_price_data(self, live_bool=False):
        symbols_universe_df = load_symbols_universe_df()
        list_of_symbols = symbols_universe_df['Symbol'].tolist()
        list_of_symbols = list(set(list_of_symbols))
        config_dict_orginal = deepcopy(self.config_dict)
        price_update_config_dict = deepcopy(self.config_dict)
        price_update_config_dict['datafeeder_config']['list_of_symbols'] = list_of_symbols
        price_update_config_dict['datafeeder_config']['data_inputs']['1m']= {'columns': ['open', 'high', 'low', 'close', 'volume'], 'lookback': 365}
        run_mode=self.config_dict['run_mode'] = 4
        self.datafetcher.config_dict = price_update_config_dict
        check_for_splits = False
        self.logger.warning(f"System is not checking for splits. Need to add this functionality.")
        if 'ibkr' in self.config_dict['data_update_inputs']['data_sources'].values():
            self.datafetcher.fetch_updated_price_data(start_date=None, end_date=None, lookback={}, throttle_secs=60, update_data=True, run_mode=self.config_dict['run_mode'], live_bool=True)
        if 'yahoo' in self.config_dict['data_update_inputs']['data_sources'].values():
            self.datafetcher.fetch_updated_price_data(start_date=None, end_date=None, lookback={}, throttle_secs=60, update_data=True, run_mode=self.config_dict['run_mode'], live_bool=False)
        self.logger.info(f'Price data updated for {len(list_of_symbols)} symbols. Granualarities Updated: {list(price_update_config_dict["datafeeder_config"]["data_inputs"].keys())}')
        self.datafetcher.config_dict = config_dict_orginal
    
    def is_time_between_0001_and_0900(self):
        # now = pd.Timestamp.now().tz_localize('UTC')
        now = pd.Timestamp.now().tz_localize('US/Eastern')
        if now.hour >= 0 and now.hour < 9:
            return True
        return False
    
    def trim_current_market_data_df(self, current_market_data_df):
        for interval in current_market_data_df.index.get_level_values(0).unique():
            lookback = self.lookback_dict_original[interval] if interval in self.lookback_dict_original.keys() else 0
            # self.logger.debug(f'Interval: {interval} | Lookback: {lookback} | Market Data DF Shape: {current_market_data_df.loc[interval].shape[0]}')
            if current_market_data_df.loc[interval].shape[0] > int(lookback * 2) and lookback > 0:
                # self.logger.debug(f'Market Data DF BEFORE TRIM: Shape: {current_market_data_df.loc[interval].shape}')
                # current_market_data_df.loc[pd.IndexSlice[interval, :], :] = current_market_data_df.loc[pd.IndexSlice[interval, :], :].iloc[int(-lookback * 1.25):].dropna()
                # Extract the slice for '1m'
                slice_to_process = current_market_data_df.loc[pd.IndexSlice[interval, :], :]
                # Take the last 120 rows and drop NaNs
                processed_slice = slice_to_process.iloc[int(-lookback * 1.25):].dropna()
                # Remove the old slice and reinsert only the cleaned data
                current_market_data_df = current_market_data_df.drop(index=slice_to_process.index)
                current_market_data_df = pd.concat([current_market_data_df, processed_slice])
                # self.logger.debug(f'Market Data DF Trimmed: Shape: {current_market_data_df.loc[interval].shape}')
                # time.sleep(2)
            elif current_market_data_df.loc[interval].shape[0] > 120 * 2 and lookback == 0:
                # self.logger.debug(f'Market Data DF BEFORE TRIM:: Shape: {current_market_data_df.loc[interval].shape}')
                # current_market_data_df.loc[pd.IndexSlice[interval, :], :] = current_market_data_df.loc[pd.IndexSlice[interval, :], :].iloc[-120:].dropna()
                # current_market_data_df.loc[pd.IndexSlice[interval, :], :] = current_market_data_df.loc[pd.IndexSlice[interval, :], :].iloc[int(-lookback * 1.25):].dropna()
                # Extract the slice for '1m'
                slice_to_process = current_market_data_df.loc[pd.IndexSlice[interval, :], :]
                # Take the last 120 rows and drop NaNs
                processed_slice = slice_to_process.iloc[-120:].dropna()
                # Remove the old slice and reinsert only the cleaned data
                current_market_data_df = current_market_data_df.drop(index=slice_to_process.index)
                current_market_data_df = pd.concat([current_market_data_df, processed_slice])
                # self.logger.debug(f'Market Data DF Trimmed: Shape: {current_market_data_df.loc[interval].shape}')
                # time.sleep(2)
        return current_market_data_df            
    
    def next(self, system_timestamp, run_mode, start_date, end_date, sleep_time=0, live_bool=False):
        throttle_secs = 1 if run_mode in [1,2] else 60
        # update data and return the updated data
        if self.previous_config_dict != self.config_dict:
            # self.logger.debug({'system_timestamp':system_timestamp, 'start_date':start_date, 'end_date':end_date, 'self.lookback_dict':self.lookback_dict})
            start_date = system_timestamp if self.previous_config_dict else start_date
            # end_date = start_date + pd.Timedelta(days=10) if not end_date else end_date
            # self.logger.debug(f'Config Dict Updated: Fetching data for {start_date} to {end_date} | System Timestamp: {system_timestamp}, Lookback Dict: {self.lookback_dict}')
            self.market_data_df = self.datafetcher.fetch_updated_price_data(start_date=start_date, end_date=end_date, throttle_secs=throttle_secs, lookback=self.lookback_dict, run_mode=self.config_dict['run_mode'], live_bool=live_bool)
            self.previous_config_dict = deepcopy(self.config_dict)
            # self.logger.info({'start_date':start_date, 'end_date':end_date})
            # msg = 'market_data_df Shape: '
            # for interval in self.market_data_df.index.get_level_values(0).unique():
            #     if interval in self.market_data_df.index.get_level_values(0).unique():
            #         msg +=  f"{interval} : {self.market_data_df.loc[interval].shape} | "
            # self.logger.info(msg)
            
            # first_timestamp_ = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()]) if len(self.market_data_df) > 0 else None
            # last_timestamp_ = max([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[-1]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()]) if len(self.market_data_df) > 0 else None
            # self.logger.info({'first_timestamp_':first_timestamp_, 'last_timestamp_':last_timestamp_})
            # sleeper(20, 'Just taking a small break 1')
        # elif
        while len(self.market_data_df) < 1 and run_mode in [1,2]:
            # self.logger.info({'Perpetual loop is running to fetch data...system_timestamp':system_timestamp})
            start_date = system_timestamp if system_timestamp else start_date
            historical_data_update_bool = False
            self.lookback_dict = self.reset_lookback_dict()
            # self.logger.info({'start_date':start_date.astimezone(pytz.timezone('US/Eastern')), 'end_date':end_date, 'self.lookback_dict':self.lookback_dict})
            # self.logger.debug(f'Market Data DF Empty: Fetching data for {start_date} to {end_date} | System Timestamp: {system_timestamp}, Lookback Dict: {self.lookback_dict}, Live Bool: {live_bool}')
            self.market_data_df = self.datafetcher.fetch_updated_price_data(start_date=start_date, end_date=end_date, throttle_secs=throttle_secs, lookback=self.lookback_dict, run_mode=self.config_dict['run_mode'], live_bool=live_bool)
            # timestamps = self.market_data_extractor.get_market_data_df_timestamps(self.market_data_df)
            # self.logger.info({'timestamp_0':timestamps[0].astimezone(pytz.timezone('US/Eastern')), 'timestamp_-1':timestamps[-1].astimezone(pytz.timezone('US/Eastern'))})
            # self.logger.info({'self.market_data_df':self.market_data_df.head(5)})
            
            # self.logger.info({'self.market_data_df':self.market_data_df})
            if len(self.market_data_df) < 1:
                now_utc = pd.Timestamp.now().tz_localize('UTC')
                sleep_time = self.get_next_expected_timestamp(now_utc)
                min_hours_for_data_update = 60*60*4
                if self.is_time_between_0001_and_0900() and sleep_time > min_hours_for_data_update and not historical_data_update_bool:
                    self.logger.info(f"Live Bool: {live_bool} | Sleep time is more than 4 hours. Updating all historical data. Sleep Time: {int(sleep_time / 60)} minutes")
                    self.update_all_historical_price_data()
                    historical_data_update_bool = True
                    pass
                else:# self.logger.info({'system_timestamp':system_timestamp, 'start_date':start_date})
                    sleeper(min(sleep_time, 60*60*2), f'Live Bool: {live_bool} | System Sleeping: Time to next timestamp')
                
            # self.logger.info({'start_date':start_date, 'end_date':end_date})
            # msg = 'market_data_df Shape: '
            # for interval in self.market_data_df.index.get_level_values(0).unique():
            #     if interval in self.market_data_df.index.get_level_values(0).unique():
            #         msg +=  f"{interval} : {self.market_data_df.loc[interval].shape} | "
            # self.logger.info(msg)
            
            # first_timestamp_ = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()]) if len(self.market_data_df) > 0 else None
            # self.logger.info({'first_timestamp_':first_timestamp_})
            # sleeper(20, 'Just taking a small break 2')

        # if self.datafeeder_system_timestamp is None:
        if len(self.market_data_df) >= 1:
            self.datafeeder_system_timestamp = min([pd.DataFrame(self.market_data_df.loc[interval,:].iloc[0]).T.index[0] for interval in self.market_data_df.index.get_level_values(0).unique()]) if len(self.market_data_df) > 0 else None
        # self.logger.debug({'Next System Timestamp':self.datafeeder_system_timestamp, 'Current System Timestamp':system_timestamp})
        
        # else:
            # self.market_data_df = self.market_data_df.loc[self.market_data_df.index.get_level_values(1) > self.datafeeder_system_timestamp,:]
            first_rows = []
            if self.datafeeder_system_timestamp:
                for interval in self.market_data_df.index.get_level_values(0).unique():
                    first_row = self.market_data_df.loc[interval,:].iloc[0]
                    # Create DataFrame with proper MultiIndex
                    first_df = pd.DataFrame(first_row).T
                    # Create DataFrame with proper index
                    first_df = pd.DataFrame(first_row).T
                    
                    # Ensure we have a valid timestamp index
                    if not isinstance(first_df.index, pd.DatetimeIndex):
                        # If index is not datetime, try to parse it
                        try:
                            first_df.index = pd.to_datetime(first_df.index)
                        except:
                            # If parsing fails, create a new datetime index
                            first_df.index = pd.date_range(start='2000-01-01', periods=len(first_df), freq='D')
                    
                    # Get the first timestamp
                    timestamp = first_df.index[0]
                    first_df.index = pd.MultiIndex.from_tuples(
                        [(interval, timestamp)],
                        names=['interval', 'datetime']
                    )
                    
                    # Check if timestamp matches
                    if first_df.index[0][1] == self.datafeeder_system_timestamp:
                        self.datafeeder_system_timestamp = first_df.index[0][1]
                        first_rows.append(first_df)
                        
                        # Remove the first row using the exact index
                        idx_to_drop = first_df.index[0]
                        self.market_data_df = self.market_data_df.drop(idx_to_drop)
                        # self.logger.debug({'market_data_df':len(self.market_data_df)})
                        
                    
            if len(first_rows) > 0:
                first_rows = pd.concat(first_rows)
            else:
                first_rows = None
        else:
                first_rows = None
        # self.logger.debug({'market_data_df':self.market_data_df})
        return first_rows