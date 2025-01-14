import json
import pandas as pd
import numpy as np
import logging
import datetime
from typing import List, Tuple, Dict, Optional
import os
import pytz
from datetime import datetime, timedelta
from systems.utils import create_logger, project_path
import pandas_market_calendars as mcal

# --- Constants ---
HARDCODED_DATA_DIR = "/mnt/VANDAN_DISK/code_stuff/projects/mathematricks_gagan/db/data/ibkr"


class ConfigManager:
    """Manages configuration settings for the trading strategy."""

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(
                f"Failed to load config file from {self.config_file_path}. Error: {e}. Using defaults.")
            return {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_all(self):
        return self.config

class IBKRConnect:
    """Manages the IBKR connection (not used in this local-only version)."""
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-connect', print_to_console=True)
        self.connected = False


class IBKR(IBKRConnect):
    """Main class for IBKR integration (data and execution)."""
    def __init__(self):
        super().__init__()
        self.data = Data()  # Modified to remove IBKR dependencies
        self.execute = IBKR_Execute()  # Order Execution



class Data:
    """Handles loading and processing historical data from local files."""
    def __init__(self):
        self.glogger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-data', print_to_console=True)
        self.interval_lookup = {"1m": "1 min", "2m": "2min", "5m": "5 min", "1d": "1 day"}
        self.market_open_bool = self.is_market_open(pd.Timestamp.now().tz_localize('UTC'))

    def update_price_data(self, stock_symbols, interval_inputs, data_folder=HARDCODED_DATA_DIR,
                         throttle_secs=1, start_date=None, end_date=None, lookback=None, update_data=False,
                         run_mode=4):
        """Fetches price data from local CSV files."""
        data_frames = []
        self.glogger.debug(f"Looking for data in folder: {data_folder}")

        for interval in interval_inputs:
            interval_folder = os.path.join(data_folder, interval)
            self.glogger.debug(f"Checking interval folder: {interval_folder}")

            if not os.path.exists(interval_folder):
                self.glogger.warning(f"Interval folder does not exist: {interval_folder}")
                continue
                
            
            for symbol in stock_symbols:
                csv_path = os.path.join(interval_folder, f"{symbol}.csv")
                self.glogger.debug(f"Checking for file: {csv_path}")

                if os.path.exists(csv_path):
                    self.glogger.debug(f"Found file: {csv_path}")
                    try:
                        df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
                        self.glogger.debug(f"Successfully read {csv_path} with {len(df)} rows")
                        df['symbol'] = symbol
                        df['interval'] = interval
                        data_frames.append(df)
                    except Exception as e:
                        self.glogger.error(f"Error reading {csv_path}: {str(e)}")
                        self.glogger.debug(f"File contents: {open(csv_path).read(100)}...")
                else:
                    self.glogger.debug(f"File not found: {csv_path}")
        if not data_frames:
           self.glogger.error("No data files found")
           return pd.DataFrame()
        
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False, inplace=True)
        combined_df.set_index(['interval', 'datetime', 'symbol'], inplace=True)
        combined_df.sort_index(inplace=True)
                
        # Filter by date range if specified
        if start_date is not None:
           combined_df = combined_df.loc[combined_df.index.get_level_values('datetime') >= start_date]
        if end_date is not None:
            combined_df = combined_df.loc[combined_df.index.get_level_values('datetime') <= end_date]

        return combined_df
    
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
        schedule = nyse.schedule(start_date=current_datetime - timedelta(days=4), end_date=current_datetime)
        current_date_str = current_date.strftime('%Y-%m-%d')
        if current_date_str in schedule.index:
            if current_datetime >= schedule.loc[current_date_str]['market_open'] and current_datetime <= schedule.loc[current_date_str]['market_close']:
                market_open_bool = True
            else:
                market_open_bool = False
        else:
            market_open_bool = False
        
        return market_open_bool


class IBKR_Execute:
    """Handles order execution (no IBKR calls in this version)."""

    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-execute', print_to_console=True)
        self.interval_lookup = {"1m": "1 min", "2m": "2min", "5m": "5 min", "1d": "1 day"}
        self.duration_lookup = {"1m": "1 W", "2m": "2 W", "5m": "1 M", "1d": "10 Y"}



if __name__ == '__main__':
    ibkr = IBKR()
    tickers = ['AAPL', 'MSFT']
    interval_inputs = {'1d': {'columns': ['open', 'high', 'low', 'close', 'volume']}}
    data = ibkr.data.update_price_data(tickers, interval_inputs)
    print(data.head())