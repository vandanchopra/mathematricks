
#from ib_insync import IB, Stock, util, MarketOrder, LimitOrder, StopOrder
# from ib_insync import *
from copy import deepcopy
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Order as IBOrder, util
from matplotlib.pyplot import bar
import nest_asyncio
import os
#from networkx import dfs_edges
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from vault.base_strategy import Order

# from datetime import datetime, timedelta
import asyncio
import logging
from datetime import date, datetime, timezone
from systems.utils import create_logger, generate_hash_id, sleeper, project_path

nest_asyncio.apply()

class IBKRConnect:
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-connect', print_to_console=True)
        self.ib = None
        self.connected = False
    
    def connect_to_IBKR(self, client_id=0):
        self.ib = IB()
        #NOTE: First start the TWS or Gateway software from IBKR

        # Connect to the IBKR TWS (Trader Workstation) or Gateway
        if client_id is None:
            client_id = 0  # Default to client_id 0
        retries = 3  # Number of retries in case of timeout
        for attempt in range(retries):
            try:
                print(f"Connecting to IBKR with clientId: {client_id}, Attempt {attempt + 1}")
                self.ib.connect('127.0.0.1', 7497, clientId=client_id)
                break  # Connection successful, exit the retry loop
            except TimeoutError:
                print(f"TimeoutError: Attempt {attempt + 1} failed.")
                # client_id += 1  # Increment clientId to avoid conflict
                if attempt == retries - 1:
                    raise  # Re-raise the exception after the last attempt
            except Exception as e:
                print(f"Error connecting to IBKR: {e}")
                client_id += 1
                if attempt == retries - 1:
                    raise

        # Check if the connection is successful
        if self.ib.isConnected():
            print('Connected to IBKR')
            self.connected = True
        else:
            print('Failed to connect to IBKR')
        return self.ib

class IBKR(IBKRConnect):
    def __init__(self):
        super().__init__()
        self.data = Data(self.ib, self.connect_to_IBKR)  # Yahoo Finance Data Fetcher
        self.execute = IBKR_Execute(self.ib, self.connect_to_IBKR)  # Order Execution
        
class Data:
    def __init__(self, ib, connect_to_IBKR):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-data', print_to_console=True)
        self.ib = ib
        self.connect_to_IBKR = connect_to_IBKR
        self.interval_lookup = {"1m": "1 min", "2m": "2min", "5m":"5 min","1d":"1 day"} # update this to include all provided interval from yahoo and ibkr

    def check_ib_connection(self):
        if not self.ib:
            self.logger.debug('Not connected to IBKR. Connecting now')
            self.ib = self.connect_to_IBKR()
            if self.ib:
                self.logger.debug('Connected to IBKR')
    
    async def update_price_data_batch_old(self, stock_symbols, interval, start_date=None, batch_size = 75):
        self.check_ib_connection()
        
        asset_data_df_dict = {}
        barSize = self.interval_lookup[interval]
        duration = self.duration_lookup[interval]
        asset_data_df_dict[interval] = {}
        task = [self.fetch_historical_data(ticker, barSize, duration, start=start_date) for ticker in stock_symbols]
        data = await asyncio.gather(*task, return_exceptions=True)
        for i, bars in enumerate(data):
            if isinstance(bars, Exception):
                raise Exception(f"Error fetching data for batch: {bars}. Are you sure TWS is Running?")

            # logging.debug(f'bars: {bars}, type: {type(bars)}')
            asset_data_df_dict[interval][stock_symbols[i]] = util.df(bars)
                        
        '''
        asset_data_df_dict = {}
        for interval in stock_symbols:
            asset_data_df_dict[interval] = {}
            for ticker in stock_symbols[interval]:
                if start_date is not None:
                    self.logger.debug({'start_date to yahoo': start_date})
                    barSize = self.interval_lookup[interval]
                    asset_data_df = self.fetch_historical_data(ticker,barSize,start=start_date)
                else:
                    asset_data_df = self.fetch_historical_data(ticker,barSize)
                asset_data_df_dict[interval][ticker]= asset_data_df
        return asset_data_df_dict'''
        
        return asset_data_df_dict
    
    def calculate_ibkr_duration_scaled(self, start_date: datetime) -> dict:
        
        """
        Converts a start_date into IBKR duration strings for given barSize types,
        scaling the duration in seconds to the appropriate IBKR-friendly granularity.

        Args:
            start_date (datetime): The start datetime (with timezone).
            bar_sizes (list[str]): A list of bar sizes, e.g., ['1m', '1d'].

        Returns:
            dict: A dictionary where keys are bar sizes and values are durations 
                with the appropriate IBKR-friendly units.
        """
        bar_sizes = ["1m", "1d"]  # Bar sizes to calculate durations for
        
        if start_date is None:
            return {"1m": "1 W", "2m": "2 W", "5m":"1 M","1d":"20 Y"}
        else:
            # Current datetime with timezone
            current_date_obj = datetime.now(timezone.utc)
            
            # Calculate total duration in seconds
            total_seconds = (current_date_obj - start_date).total_seconds()
            duration_dict = {}

            def scale_duration(seconds):
                # 1 year = 31,536,000 seconds (365 days)
                if seconds >= 31536000:  # More than 1 year
                    return f"{int(seconds // 31536000) + 1} Y"  # Display in years
                elif seconds >= 2419200:  # More than 1 month (30 days)
                    return f"{int(seconds // 2419200) + 1} M"  # Display in months
                elif seconds >= 604800:  # More than 1 week
                    return f"{int(seconds // 604800) + 1} W"  # Display in weeks
                elif seconds >= 86400:  # More than 1 day
                    return f"{int(seconds // 86400) + 1} D"  # Display in days
                else:  # Less than 1 hour
                    return f"{int(seconds) + 1} S"  # Display in minutes
            

            # Map bar sizes to appropriate duration calculations
            for bar_size in bar_sizes:
                if "m" in bar_size:  # Minute-level data
                    duration = scale_duration(total_seconds)
                    duration_dict[bar_size] = duration
                elif "h" in bar_size:  # Hour-level data
                    duration = scale_duration(total_seconds)
                    duration_dict[bar_size] = duration
                elif "d" in bar_size:  # Day-level data
                    duration = scale_duration(total_seconds)
                    duration_dict[bar_size] = duration
                elif "w" in bar_size:  # Week-level data
                    duration = scale_duration(total_seconds)
                    duration_dict[bar_size] = duration
                else:
                    raise ValueError(f"Unsupported bar size: {bar_size}")

            return duration_dict
    
    async def update_price_data_batch(self, stock_symbols, interval, start_date=None, batch_size=75):
        self.check_ib_connection()

        asset_data_df_dict = {}
        barSize = self.interval_lookup[interval]
        # self.logger.debug({'stock_symbols':stock_symbols})
        # self.logger.debug({'interval':interval, 'barSize':barSize, 'start_date':start_date})
        ibkr_duration_dict = self.calculate_ibkr_duration_scaled(start_date)
        duration = ibkr_duration_dict[interval]

        # Map symbols to their fetch tasks
        tasks = {
            ticker: self.fetch_historical_data(ticker, barSize, duration, start_date=start_date)
            for ticker in stock_symbols
        }

        # Initialize progress bar
        with tqdm_asyncio(total=len(stock_symbols), desc=f'IBKR - Updating data: Interval: {interval}', unit="stock") as pbar:
            data = {}
            for symbol, task in tasks.items():
                try:
                    bars = await task  # Await the task result
                    data[symbol] = util.df(bars)
                except Exception as e:
                    data[symbol] = e  # Capture the exception for that symbol
                finally:
                    pbar.update(1)  # Update progress bar for each task

        # Process and validate the data
        for symbol, bars in data.items():
            if not isinstance(bars, Exception):
                # raise Exception(f"Error fetching data for {symbol}: {bars}. Are you sure TWS is running?")
            
                asset_data_df_dict[symbol] = bars

        return asset_data_df_dict

    async def fetch_historical_data(self, ticker, barSize, duration, start_date=None, exchange:str = "SMART", currency:str = "USD"):
        try:
            contract = self.ib.qualifyContracts(Stock(ticker, exchange, currency))[0]
        
        # self.logger.debug({'Symbol':ticker, 'barSize': barSize, 'duration': duration})
        # duration = self.duration_lookup.get(barSize,"1 Y")
        # barSize = self.interval_lookup.get(barSize,"1 day")
        # duration = "20 Y"
        # barSize = "1 day"
        # Fetch historical data for max duration
            # self.logger.debug(f"Fetching data for {ticker}|{exchange}|{currency} with barSize: {barSize}, duration: {duration}")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                
            )
        except Exception as e:
            bars = Exception(f"Error fetching data for {ticker}. Error: {e}")
        
        return bars

    def restructure_asset_data_df(self, asset_data_df):
        # If asset_data_df is not None, then restructure it
        if asset_data_df is not None:
            if not asset_data_df.empty:
                asset_data_df.columns = asset_data_df.columns.str.lower()
                # Convert the date column to datetime in UTC timezone
                # Add time to the date column
                if 'date' in asset_data_df.columns:
                    asset_data_df['date'] = pd.to_datetime(asset_data_df['date'],utc=True)
                    # Set the date column as the index
                    asset_data_df.set_index(['date'],inplace=True)
                asset_data_df.index.names = ['datetime']
                asset_data_df.index = asset_data_df.index.tz_convert('UTC')
                cols = list(asset_data_df.columns)
                asset_data_df = asset_data_df.T.reset_index(drop=True).T
                asset_data_df = asset_data_df.set_axis(cols, axis=1)
        else:
            asset_data_df = pd.DataFrame()
        
        return asset_data_df
    
    def remove_unwanted_cols(self, interval_inputs, interval, asset_data_df):
        pass_cols = []
        useful_columns = interval_inputs[interval]['columns']
        cols = list(asset_data_df.columns)
        for col in useful_columns:
            if col in cols:
                pass_cols.append(col)
        to_use_cols = pass_cols+['symbol','interval']
        
        return asset_data_df[to_use_cols]
    
    def update_price_data(self, stock_symbols, interval_inputs, data_folder=project_path+'db/data/ibkr', throttle_secs=1, start_date=None, end_date=None, lookback=None, update_data=True, run_mode=4):
        data_frames = []
        batch_size = 75
        
        '''STEP 1: Bifurcate the list of stock symbols into three lists: 1) ones that have data 2) ones that don't have data 3) Ones that have partial data'''
        # Break the list into two lists. ones that don't have data and ones that have data
        stock_symbols_no_data = { k:[] for k in interval_inputs}
        stock_symbols_with_partial_data = { k:[] for k in interval_inputs}
        stock_symbols_with_full_data = { k:[] for k in interval_inputs}
        existing_data_dict = { k:{} for k in interval_inputs}
        
        for interval in interval_inputs:
            csv_loader_pbar = tqdm(stock_symbols, desc=f'Fetching {interval} CSV data & Bifurcating Batches: ')
            
            for symbol in stock_symbols:
                symbol = symbol.replace('/','-') if '/' in symbol else symbol
                csv_file_path = os.path.join(data_folder, interval, f"{symbol}.csv")

                if not os.path.exists(csv_file_path):
                    stock_symbols_no_data[interval].append(symbol)
                else:
                    try:
                        existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape})
                        
                        # Add timedelta 1 day to the index column
                        if not existing_data.empty:
                            # existing_data_first_date = existing_data.index.min().tz_convert('UTC')
                            existing_data_last_date = existing_data.index.max().tz_convert('UTC')
                            # self.logger.debug({'symbol':symbol, 'existing_data_last_date':existing_data_last_date.astimezone('US/Eastern')})
                            yday_date = pd.Timestamp.today().tz_localize('UTC').replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                            # replace hours, minutes, seconds with 0
                            # if interval == '1d':
                            # self.logger.debug({'symbol':symbol, 'interval':interval,'existing_data_last_date':existing_data_last_date, 'yday_date':yday_date, 'end_date':end_date, 'timestamp_test_bool':existing_data_last_date >= yday_date})
                            # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape})
                            
                            if (end_date is not None and existing_data_last_date >= end_date) or (existing_data_last_date >= yday_date and interval == '1d'):
                            # if end_date is not None and existing_data_last_date >= end_date:
                                # prune the data using the back_test_start_date and back_test_end_date
                                # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape})
                                existing_data = existing_data.loc[:end_date]
                                # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape})
                                existing_data_dict[interval][symbol] = existing_data
                                stock_symbols_with_full_data[interval].append(symbol)
                            else:
                                existing_data_dict[interval][symbol] = existing_data
                                stock_symbols_with_partial_data[interval].append(symbol)
                                
                        else: 
                            stock_symbols_no_data[interval].append(symbol)
                            
                    except Exception as e:
                        raise Exception(f"Error reading {csv_file_path}. Error: {e}")
                csv_loader_pbar.update(1)
            csv_loader_pbar.close()

        for interval in interval_inputs:
            self.logger.info({'interval':interval, 'stock_symbols_no_data':len(stock_symbols_no_data[interval]), 'stock_symbols_with_partial_data':len(stock_symbols_with_partial_data[interval]), 'stock_symbols_with_full_data':len(stock_symbols_with_full_data[interval])})
            # if interval == '1d':
                # self.logger.debug({'partial_data_symbols':stock_symbols_with_full_data[interval]})
        # sleeper(5, 'Giving you time to read the above Message')
        
        '''STEP 2: Get the data for the ones that don't have data'''
        for interval in interval_inputs:
            if len(stock_symbols_no_data[interval]) > 0:
                pbar = tqdm(stock_symbols_no_data[interval], desc=f'Updating NO data: Interval: {interval}')
                # pbar_ydownloader = tqdm(stock_symbols[interval], desc= f'Downloading data from Yahoo: Interval: {interval}: ')
                for i in range(0, len(stock_symbols_no_data[interval]), batch_size):
                    batch = stock_symbols_no_data[interval][i:i + batch_size]
                    # batch_asset_data_df_dict = self.batch_update÷_price_data(batch, interval, start_date=None)
                    # self.logger.info({'interval':interval, 'batch':batch})
                    loop = asyncio.get_event_loop()
                    batch_asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(batch, interval, start_date=None))
                    
                    # Extract the data from yahoo batch response and save it to csv
                    data_input_folder = os.path.join(data_folder, interval)
                    os.makedirs(data_input_folder, exist_ok=True)
                    for symbol in batch_asset_data_df_dict:
                        # Extract the data from the yahoo batch response
                        asset_data_df = batch_asset_data_df_dict[symbol]
                        # asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                        # self.logger.info({'symbol':symbol, 'asset_data_df':asset_data_df.shape})
                        asset_data_df = self.restructure_asset_data_df(asset_data_df)
                        # self.logger.debug({'symbol':symbol, 'asset_data_df':asset_data_df})
                        asset_data_df = asset_data_df.dropna(how='all')
                        asset_data_df['symbol'] = symbol
                        asset_data_df['interval'] = interval
                        
                        if interval == '1d':
                            today = pd.Timestamp(datetime.now()).tz_localize('UTC')
                            today = today.replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(minutes=1)
                            asset_data_df = asset_data_df.loc[:today]
                        
                        # Save it to the csv file
                        symbol = symbol.replace('/','-') if '/' in symbol else symbol
                        csv_file_path = os.path.join(data_folder, interval, f"{symbol}.csv")
                        
                        # self.logger.info({'symbol':symbol, 'asset_data_df':asset_data_df.shape})
                        # self.logger.info(f"Trying to save to {csv_file_path}")
                        if not asset_data_df.empty:
                            asset_data_df.to_csv(csv_file_path)
                            # self.logger.info({f"Saved {symbol}|{interval} data to {csv_file_path}"})
                        # else:
                            # self.logger.warning(f"Data for {symbol}|{interval} is empty.")

                        # Remove all cols not needed
                        asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                        
                        # Update it to the data_frames list
                        data_frames.append(asset_data_df)
                    pbar.update(len(batch))
                    if throttle_secs < 1:
                        self.logger.warning('Throttle seconds is less than 1 to give IBKR API time to breathe.')
                        time.sleep(throttle_secs)
                    else:
                        sleeper(throttle_secs, 'Giving IBKR API time to breathe.')  # To avoid hitting rate limits
                pbar.close()
        
        '''STEP 3: Get the data for the ones that have partial data'''
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        for interval in interval_inputs:
            if len(stock_symbols_with_partial_data[interval]) > 0:
                pbar = tqdm(stock_symbols_with_partial_data[interval], desc=f'Updating Partial data: Interval: {interval}')
                for i in range(0, len(stock_symbols_with_partial_data[interval]), batch_size):
                    batch = stock_symbols_with_partial_data[interval][i:i + batch_size]
                    # self.logger.info({'interval':interval, 'batch':batch})
                    batch_start_date = None
                    for symbol in batch:
                        existing_data = existing_data_dict[interval][symbol]
                        # last_date = existing_data.index.max().tz_localize('UTC') if not existing_data.empty else None
                        last_date = existing_data.index.max().tz_convert('UTC') if not existing_data.empty else None
                        batch_start_date = last_date if batch_start_date is None and last_date is not None else None if last_date is None else min(batch_start_date, last_date)
                    
                    loop = asyncio.get_event_loop()
                    batch_asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(batch, interval, start_date=batch_start_date))
                    ## Dummy code to emulate yfinance data download failure
                    # batch_asset_data_df_dict = {}
                    # for symbol in batch:
                    #     batch_asset_data_df_dict[symbol] = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                    #                                                     [('Adj Close', symbol), ('Close', symbol), ('High', symbol), 
                    #                                                         ('Low', symbol), ('Open', symbol), ('Volume', symbol)],
                    #                                                             names=['Metric', 'Ticker']), index=pd.DatetimeIndex([])
                    #                                                             )
                    
                    data_input_folder = os.path.join(data_folder, interval)
                    for symbol in batch:
                        # Extract the data from the yahoo batch response
                        asset_data_df = batch_asset_data_df_dict[symbol]
                        # self.logger.info({'symbol':symbol, 'asset_data_df':asset_data_df.shape})
                        # asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                        asset_data_df = self.restructure_asset_data_df(asset_data_df)
                        asset_data_df = asset_data_df.dropna(how='all')
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'asset_data_df':asset_data_df})
                        existing_data = existing_data_dict[interval][symbol]
                        # get the start date of asset_data_df
                        symbol_start_date = asset_data_df.index.min() if not asset_data_df.empty else start_date
                        # prune the existing_data to only include data before the start date
                        symbol_start_date = symbol_start_date.to_pydatetime()
                        # self.logger.debug({'symbol':symbol, 'symbol_start_date':symbol_start_date})
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.index})
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape})
                        
                        existing_data = existing_data[existing_data.index < symbol_start_date]
                        # new_data = asset_data_df[asset_data_df.index >= existing_data.index.max()]
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'existing_data':existing_data.shape, 'new_data':new_data.shape})
                        # concatenate the existing data and the new data
                        # self.logger.info({'symbol':symbol, 'start_date':start_date, 'interval':interval, 'existing_data':existing_data.shape, 'asset_data_df':asset_data_df.shape})
                        if not asset_data_df.empty:
                            updated_data = pd.concat([existing_data, asset_data_df])
                        else:
                            updated_data = existing_data
                        # Remove symbol and interval columns if they exist in updated_data
                        updated_data = updated_data.drop(columns=['symbol','interval'], errors='ignore')
                        updated_data = updated_data.dropna(how='all')
                        updated_data['symbol'] = symbol
                        updated_data['interval'] = interval
                        # self.logger.debug({'symbol':symbol, 'interval':interval, 'updated_data':updated_data})
                        if interval == '1d':
                            today = pd.Timestamp(datetime.now()).tz_localize('UTC')
                            today = today.replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(minutes=1)
                            updated_data = updated_data.loc[:today]
                        
                        # updated_data = updated_data.iloc[:-1] if interval == '1d' else updated_data
                        # self.logger.info({'symbol':symbol, 'existing_data':existing_data.shape, 'updated_data':updated_data.shape, 'asset_data_df':asset_data_df.shape})
                        # Save it to the csv file
                        # if update_data is not an empty DataFrame, then save it to the csv file
                        
                        if not updated_data.empty and not asset_data_df.empty:
                            symbol = symbol.replace('/','-') if '/' in symbol else symbol
                            csv_file_path = os.path.join(data_folder, interval, f"{symbol}.csv")
                            # self.logger.info(f"Trying to save to {csv_file_path}")
                            updated_data.to_csv(csv_file_path)
                            # self.logger.info({f"Saved {symbol}|{interval} data to {csv_file_path}: asset_data_df: {asset_data_df.tail(2)}"})
                        # else:
                        #     self.logger.warning(f"Data for {symbol}|{interval} is empty.")
                        
                        # if interval == '1m':
                        # self.logger.info({'asset_data_df':asset_data_df})
                        # self.logger.info({f"Saved {symbol}|{interval} data to {csv_file_path}"})
                        # sleeper(5, 'Sleeping for 5 seconds...')
                        
                        # Remove all cols not needed
                        updated_data = self.remove_unwanted_cols(interval_inputs, interval, updated_data)
                        
                        # Update it to the data_frames list
                        data_frames.append(updated_data)
                    pbar.update(len(batch))
                    if throttle_secs < 1:
                        self.logger.warning(f'Throttle seconds is less than 1 to give Yahoo API time to breathe. throttle_secs: {throttle_secs}')
                        time.sleep(throttle_secs)
                    else:
                        sleeper(throttle_secs, f'Giving IBKR API time to breathe. throttle_secs: {throttle_secs}')  # To avoid hitting rate limits
                        
                pbar.close()
        
        '''STEP 4: Get the data for the ones that have full data'''
        for interval in interval_inputs:
            if len(stock_symbols_with_full_data[interval]) > 0:
                pbar = tqdm(stock_symbols_with_full_data[interval], desc='Get the data for the ones that have full data: ')
                for symbol in existing_data_dict[interval]:
                    asset_data_df = existing_data_dict[interval][symbol]
                    asset_data_df = self.restructure_asset_data_df(asset_data_df)
                    asset_data_df['symbol'] = symbol
                    asset_data_df['interval'] = interval
                    # asset_data_df = asset_data_df.iloc[:-1] if interval == '1d' else asset_data_df
                    
                    # Remove all cols not needed
                    asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                    
                    # Update it to the data_frames list
                    # self.logger.debug({'symbol':symbol, 'interval':interval, 'asset_data_df':asset_data_df})
                    data_frames.append(asset_data_df)
                    self.logger.debug({'symbol':symbol, 'interval':interval, 'asset_data_df':asset_data_df.shape})
                    pbar.update(1)
                pbar.close()
        
        '''Step 5: Add 1d timedelta to date'''
        for dataframe in data_frames:
            if '1d' in dataframe['interval'].unique():
                dataframe.index = dataframe.index + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        '''Step 6: Trim the data to the back_test_start_date and back_test_end_date'''
        for count, data_frame in enumerate(data_frames):
            lookback_value = int(lookback['1d'] * 1.2) if '1d' in lookback else 0
            before = data_frame.loc[:start_date]
            try:
                after = data_frame.loc[start_date:end_date]
                after = after.iloc[1:]
                before_new = before.iloc[-lookback_value:]
            except Exception as e:
                raise Exception({'start_date':start_date, 'end_date':end_date})
            # self.logger.info({'interval':interval, 'before':before.shape, 'after':after.shape, 'before_new':before_new.shape, 'req_start_date':start_date, 'req_end_date':end_date, 'tail':data_frame.tail()})
            # self.logger.debug({'interval':interval, 'lookback_value':lookback_value, 'first_row':before_new.index[0], 'last_row':before_new.index[-1]}) 

            # join before and after dataframes
            if lookback_value > 0:
                joined = pd.concat([before_new, after])
            else:
                joined = after
            joined.index.name = 'date'
            data_frames[count] = joined        
        
        # self.logger.debug({'start_date':start_date, 'end_date':end_date, 'lookback_value':lookback_value, 'joined':joined.shape})
        # raise AssertionError('STOP')
        
        '''STEP 7: # Combine all DataFrames into a single DataFrame'''
        # self.logger.info('Combining all DataFrames into a single DataFrame...')
        # Remove empty dataframes from the list data_frames
        
        data_frames = [df for df in data_frames if not df.empty]
        # self.logger.info({'data_frame':data_frames[0]})
        if len(data_frames) > 0:
            combined_df = pd.concat(data_frames)
            combined_df.reset_index(drop=False,inplace=True)
            # Set multi-index
            combined_df.set_index(['date','symbol'],inplace=True)
            asset_data_df = data_frames[0]
            pass_cols = list(asset_data_df.columns)
            combined_df = combined_df.reset_index().pivot_table(values=pass_cols, index=['interval', 'date'], columns=['symbol'], aggfunc='mean')
            # combined_df = combined_df.unstack(level='symbol')
            # Sort the index
            combined_df.sort_index(inplace=True)
            # self.logger.info({'combined_df':combined_df})
            # raise AssertionError('STOP HERE')

            '''STEP 8: Add a column for data_source= 'yahoo' '''
            combined_df['data_source'] = 'ibkr'
        else:
            combined_df = pd.DataFrame()
        
        return combined_df

class IBKR_Execute:
    def __init__(self, ib, connect_to_IBKR):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-execute', print_to_console=True)
        self.interval_lookup = {"1m": "1 min", "2m": "2min", "5m":"5 min","1d":"1 day"} # update this to include all provided interval from yahoo and ibkr
        self.duration_lookup = {"1m": "1 W", "2m": "2 W", "5m":"1 M","1d":"10 Y"} #update this to get max duration fora each interval
        self.ib = ib
        self.connect_to_IBKR = connect_to_IBKR
        self.ibkr_open_orders = {'open_orders':[], 'updated_time':None}
    
    def check_ib_connection(self):
        if not self.ib:
            self.logger.debug('Not connected to IBKR. Connecting now')
            self.ib = self.connect_to_IBKR()
            if self.ib:
                self.logger.debug('Connected to IBKR')
    
    def create_system_order_from_ibkr_order(self, ibkr_order):
        order_template = {'symbol': None,
                                'timestamp': None, 
                                "orderDirection":None, 
                                'fill_price':None,
                                "orderType":None, 
                                "timeInForce":None,
                                "orderQuantity":None,
                                "strategy_name":None,
                                "broker": None,
                                'granularity': None,
                                "status": None,
                                "signal_id": None,
                                "symbol_ltp": {},
                                "broker_order_id": None
                                }
        
        order = deepcopy(order_template)
        order['symbol'] = ibkr_order.contract.symbol
        order['orderDirection'] = ibkr_order.order.action
        if ibkr_order.order.orderType.lower() in ['stp']:
            order['orderType'] = 'stoploss_abs'
            order['exitPrice'] = ibkr_order.order.auxPrice
        elif ibkr_order.order.orderType.lower() in ['mkt']:
            order['orderType'] = 'MARKET'
            price = ibkr_order.order.fillPrice if hasattr(ibkr_order.order, 'fillPrice') else ibkr_order.order.lmtPrice
            order['entryPrice'] = price
            order['fill_price'] = price
        order['timeInForce'] = ibkr_order.order.tif
        order['orderQuantity'] = ibkr_order.order.totalQuantity
        ibkr_status = ibkr_order.orderStatus.status if hasattr(ibkr_order.orderStatus, 'status') else None
        if ibkr_status:
            if ibkr_status in ['PendingSubmit']:
                status = 'submitted'
            elif ibkr_status in ['Submitted', 'PreSubmitted']:
                status = 'open'
            elif ibkr_status == 'Filled':
                status = 'closed'
            elif ibkr_status == 'Cancelled':
                status = 'cancelled'
            else:
                status = 'pending'
            order['status'] = status
        order['broker_order_id'] = ibkr_order.order.permId
        order['timestamp'] = ibkr_order.log[-1].time if ibkr_order.log else datetime.now()
        order['currency'] = ibkr_order.contract.currency
        order['broker'] = 'IBKR'
        return order
    
    def get_open_orders(self, market_data_df, system_timestamp, min_granularity):
        open_positions_ibkr = self.ib.positions()
        self.unfilled_orders_ibkr = self.ib.reqAllOpenOrders()
        trades = self.ib.trades()
        
        open_orders_ibkr = []

        for open_position in open_positions_ibkr:
            multi_leg_order = []
            symbol = open_position.contract.symbol
            position_size = open_position.position
            avg_cost = open_position.avgCost
            orderDirection = 'BUY' if position_size > 0 else 'SELL'
            entry_order = None
            current_price = None
            try:
                current_price = market_data_df.loc[min_granularity].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
            except Exception as e:
                current_price = self.ib.reqTickers(Stock(symbol, 'SMART', 'USD'))[0].marketPrice()
                self.logger.error(f"Error fetching current price for symbol from market_data_df: {symbol}. This needs to be fixed, Using IBKR to fetch the price as a backup.")
            finally:
                current_price = current_price if current_price else 0
                
            entry_order_leg = {'symbol': symbol,
                                'timestamp': 'Unknown', 
                                "orderDirection":orderDirection, 
                                "entryPrice":avg_cost,
                                'fill_price':avg_cost,
                                "orderType":'MARKET', 
                                "timeInForce":'DAY',
                                "orderQuantity":abs(position_size),
                                "strategy_name":'Unknown',
                                "broker": 'IBKR', 
                                'granularity': 'Unknown',
                                "status": 'closed',
                                "signal_id": 'Unknown',
                                "symbol_ltp": {str(system_timestamp):current_price},
                                "broker_order_id": entry_order.order.permId if entry_order else None
                                }
            multi_leg_order.append(entry_order_leg)
            # self.logger.debug(f"IBKR Open Position: Symbol: {symbol}, Position Size: {position_size}, Avg Cost: {avg_cost}, Current Price: {current_price}")
            
            # Find the exit order for the position
            for exit_order in self.unfilled_orders_ibkr:
                if exit_order.contract.symbol == symbol and exit_order.order.orderType.lower() in ['stp']:
                    # self.logger.debug({'exit_order':exit_order})
                    exit_order_leg = {'symbol': symbol, 
                                    'timestamp': exit_order.log[-1].time if exit_order.log else 'Unknown', 
                                    "orderDirection":exit_order.order.action, 
                                    "exitPrice":exit_order.order.lmtPrice, 
                                    "orderType":exit_order.order.orderType, 
                                    "timeInForce":exit_order.order.tif,
                                    "orderQuantity":exit_order.order.totalQuantity,
                                    "strategy_name":'Unknown',
                                    "broker": 'IBKR', 
                                    'granularity': 'Unknown',
                                    "status": 'open',
                                    "signal_id": 'Unknown',
                                    "symbol_ltp": {str(system_timestamp):current_price},
                                    "broker_order_id": exit_order.order.permId,
                                    "stoploss_abs":exit_order.order.lmtPrice
                                    }
                    multi_leg_order.append(exit_order_leg)
        
            open_orders_ibkr.append(multi_leg_order)
        
        for open_order_ibkr in open_orders_ibkr:
            msg = f"Open Order IBKR: Symbol: {open_order_ibkr[0]['symbol']}, Position Size: {open_order_ibkr[0]['orderQuantity']}, Entry Price: {open_order_ibkr[0]['entryPrice']}"
            if len(open_order_ibkr) > 1:
                msg += f", Exit Price: {open_order_ibkr[1]['exitPrice']}"
            self.logger.info(msg)
                
        return open_orders_ibkr, self.unfilled_orders_ibkr
    
    def place_order(self, order, market_data_df):
        # Create a contract for the stock
        system_timestamp = market_data_df.index.get_level_values(1)[-1]
        symbol = order.symbol
        currency = 'USD'
        exchange = 'SMART'
        orderDirection = order.orderDirection
        orderQuantity = order.orderQuantity
        
        self.check_ib_connection()
        response_order = None
        # Create an order for the stock
        if order.order_type == 'MARKET':
            contract = self.ib.qualifyContracts(Stock(symbol, exchange, currency))[0]
            ib_order = MarketOrder(orderDirection, orderQuantity)
            if not hasattr(order, 'order_id') or order.order_id is None:
                mathematricks_order_id = generate_hash_id(order.dict(), system_timestamp)
            else:
                mathematricks_order_id = order.order_id
                
            # ib_order.OrderRef = mathematricks_order_id
            # Place the order
            ib_order_response = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(1)
            # self.logger.debug({'IB_order_response':IB_order_response})
            response_order = deepcopy(order)
            response_order.order_id = mathematricks_order_id
            response_order.broker_order_id = ib_order_response.order.permId
            
            ibkr_order_status = ib_order_response.orderStatus.status if hasattr(ib_order_response.orderStatus, 'status') else None
            
            if ibkr_order_status in ['PendingSubmit']:
                response_order.status = 'submitted'
            elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
                response_order.status = 'open'
            elif ibkr_order_status == 'Filled':
                response_order.status = 'closed'
            elif ibkr_order_status == 'Cancelled':
                response_order.status = 'cancelled'
            else:
                response_order.status = 'pending'
            
            setattr(response_order, 'filled', ib_order_response.orderStatus.filled)
            setattr(response_order, 'remaining', ib_order_response.orderStatus.remaining)
            response_order.filled_price = ib_order_response.orderStatus.avgFillPrice
            setattr(response_order, 'tradeLogEntryTime', ib_order_response.log[0].time if ib_order_response.log else None)
            setattr(response_order, 'errorCode', ib_order_response.log[0].errorCode if ib_order_response.log else None)
            setattr(response_order, 'fresh_update', True)
            setattr(response_order, 'message', 'Market order placed on IBKR')
            
        elif order.order_type == 'LIMIT':
            contract = self.ib.qualifyContracts(Stock(symbol, exchange, currency))[0]
            ib_order = LimitOrder(orderDirection, orderQuantity, order.price)
            if not hasattr(order, 'order_id') or order.order_id is None:
                mathematricks_order_id = generate_hash_id(order.dict(), system_timestamp)
            else:
                mathematricks_order_id = order.order_id
                
            ib_order_response = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(1)
            response_order = deepcopy(order)
            response_order.order_id = mathematricks_order_id
            response_order.broker_order_id = ib_order_response.order.permId
            
            ibkr_order_status = ib_order_response.orderStatus.status if hasattr(ib_order_response.orderStatus, 'status') else None
            
            if ibkr_order_status in ['PendingSubmit']:
                response_order.status = 'submitted'
            elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
                response_order.status = 'open'
            elif ibkr_order_status == 'Filled':
                response_order.status = 'closed'
            elif ibkr_order_status == 'Cancelled':
                response_order.status = 'cancelled'
            else:
                response_order.status = 'pending'
            
            setattr(response_order, 'filled', ib_order_response.orderStatus.filled)
            setattr(response_order, 'remaining', ib_order_response.orderStatus.remaining)
            response_order.filled_price = ib_order_response.orderStatus.avgFillPrice
            setattr(response_order, 'tradeLogEntryTime', ib_order_response.log[0].time if ib_order_response.log else None)
            setattr(response_order, 'errorCode', ib_order_response.log[0].errorCode if ib_order_response.log else None)
            setattr(response_order, 'fresh_update', True)
            setattr(response_order, 'message', 'Limit order placed on IBKR')
            
        elif order.order_type == 'STOPLOSS':
            contract = self.ib.qualifyContracts(Stock(symbol, exchange, currency))[0]
            ib_order = StopOrder(orderDirection, orderQuantity, order.price)
            if not hasattr(order, 'order_id') or order.order_id is None:
                mathematricks_order_id = generate_hash_id(order.dict(), system_timestamp)
            else:
                mathematricks_order_id = order.order_id
            
            ib_order_response = self.ib.placeOrder(contract, ib_order)
            self.ib.sleep(1)
            response_order = deepcopy(order)
            response_order.order_id = mathematricks_order_id
            response_order.broker_order_id = ib_order_response.order.permId
            
            ibkr_order_status = ib_order_response.orderStatus.status if hasattr(ib_order_response.orderStatus, 'status') else None
            
            if ibkr_order_status in ['PendingSubmit']:
                response_order.status = 'submitted'
            elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
                response_order.status = 'open'
            elif ibkr_order_status == 'Filled':
                response_order.status = 'closed'
            elif ibkr_order_status == 'Cancelled':
                response_order.status = 'cancelled'
            else:
                response_order.status = 'pending'
            
            setattr(response_order, 'filled', ib_order_response.orderStatus.filled)
            setattr(response_order, 'remaining', ib_order_response.orderStatus.remaining)
            response_order.filled_price = ib_order_response.orderStatus.avgFillPrice
            setattr(response_order, 'tradeLogEntryTime', ib_order_response.log[0].time if ib_order_response.log else None)
            setattr(response_order, 'errorCode', ib_order_response.log[0].errorCode if ib_order_response.log else None)
            setattr(response_order, 'fresh_update', True)
            setattr(response_order, 'message', 'Stop-loss order placed on IBKR')

        else:
            response_order = deepcopy(order)
            response_order.status = 'rejected'
            setattr(response_order, 'fresh_update', True)
            setattr(response_order, 'message', f"Order type {order.order_type} not supported. Use MARKET, LIMIT, or STOPLOSS")
            self.logger.error(f"Order type {order.order_type} not supported. Use MARKET, LIMIT, or STOPLOSS")

        return response_order
    
    def ibkr_order_status_to_mathematricks_order_status(self, ibkr_order_status):
        if ibkr_order_status in ['PendingSubmit']:
            status = 'submitted'
        elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
            status = 'open'
        elif ibkr_order_status == 'Filled':
            status = 'closed'
        elif ibkr_order_status in ['Cancelled', 'ApiCancelled']:
            status = 'cancelled'
        # elif ibkr_order_status in ['Inactive']:
            # status = 'rejected'
        else:
            raise AssertionError(f"IBKR Order status {ibkr_order_status} not recognized.")
        
        return status
    
    def order_status_change_message(self, current_status, new_status):
        if current_status == 'pending' and new_status == 'submitted':
            message = 'Order submitted to IBKR'
        elif current_status == 'submitted' and new_status == 'open':
            message = 'Order opened on IBKR'
        elif current_status == 'open' and new_status == 'closed':
            message = 'Order closed on IBKR'
        elif new_status == 'cancelled':
            message = 'Order cancelled on IBKR'
        return message
    
    def update_order_status(self, order, system_timestamp):
        response_order = order
        
        if order.status in ['open', 'submitted', 'pendingsubmit']:
            if self.ibkr_open_orders['updated_time'] is None or system_timestamp != self.ibkr_open_orders['updated_time']:
                self.ibkr_open_orders['open_orders'] = self.ib.reqOpenOrders()
                self.ibkr_open_orders['updated_time'] = system_timestamp
            
            current_status = order.status
            for open_order in self.ibkr_open_orders['open_orders']:
                if open_order.order.permId == order.broker_order_id:
                    new_status = self.ibkr_order_status_to_mathematricks_order_status(open_order.orderStatus.status)
                    if new_status != current_status:
                        response_order = deepcopy(order)
                        response_order.status = new_status
                        setattr(response_order, 'filled', open_order.orderStatus.filled)
                        setattr(response_order, 'remaining', open_order.orderStatus.remaining)
                        response_order.filled_price = open_order.orderStatus.avgFillPrice
                        setattr(response_order, 'tradeLogEntryTime', open_order.log[0].time if open_order.log else None)
                        setattr(response_order, 'errorCode', open_order.log[0].errorCode if open_order.log else None)
                        setattr(response_order, 'message', self.order_status_change_message(current_status, new_status))
                        setattr(response_order, 'fresh_update', True)
                    else:
                        response_order = order
                    break
                
        return response_order
    
    def modify_order(self, order, system_timestamp):
        self.check_ib_connection()
        new_price = None
        new_quantity = None
        response_order = order
        self.unfilled_orders_ibkr = self.ib.reqAllOpenOrders()
        
        target_order = None
        # Fetch the open order by order ID
        for unfilled_order in self.unfilled_orders_ibkr:
            if unfilled_order.order.permId == order.broker_order_id:
                target_order = unfilled_order
                break
        if not target_order:
            for unfilled_order in self.unfilled_orders_ibkr:
                if unfilled_order.contract.symbol == order.symbol and unfilled_order.order.action == order.orderDirection:
                    target_order = unfilled_order
                    break
        
        if hasattr(order, 'modify_reason') and 'new_price' in order.modify_reason:
            new_price = order.price
        if hasattr(order, 'modify_reason') and 'new_quantity' in order.modify_reason:
            new_quantity = order.orderQuantity
            
        if target_order:
            order_id = target_order.order.orderId
            
            new_price = target_order.order.auxPrice if not new_price else new_price
            new_quantity = target_order.order.totalQuantity if not new_quantity else new_quantity
            # Check if the order is already filled or canceled
            order_status = target_order.orderStatus.status
            if order_status not in ["Filled", "Cancelled"]:
                # Cancel the existing order
                assert target_order.order.clientId == self.ib.client.clientId, f"Order ID {order_id} does not belong to the current client ID: {self.client.clientId}."
                self.ib.cancelOrder(target_order.order)
                self.logger.info(f"Order {order_id} canceled successfully.")
                
                # Wait until the cancellation is confirmed
                self.ib.sleep(1)

                # Place a new order with updated details
                new_order = IBOrder(
                    action=target_order.order.action,  # e.g., 'BUY' or 'SELL'
                    orderType='STP',                   # Assuming a limit order, can be modified if needed
                    totalQuantity=new_quantity,
                    auxPrice=new_price, 
                    tif='GTC'
                )
                trade = self.ib.placeOrder(target_order.contract, new_order)
                response_order = deepcopy(order)
                response_order.price = new_price
                response_order.orderQuantity = new_quantity
                setattr(response_order, 'fresh_update', True)
                if trade.orderStatus.status in ['PendingSubmit']:
                    response_order.status = 'submitted'
                    self.logger.info(f"New order placed with updated price: {new_price} and quantity: {new_quantity}. Order Status: {trade.orderStatus.status}")
            else:
                self.logger.error("Order is already done (filled or canceled); no modification needed.")
        else:
            self.logger.error(f"Order for symbol {order.symbol} not found among open orders.")

        return response_order

        return response_order

    def execute_order(self, order, market_data_df, system_timestamp):
        response_order = order
        # Extract slippage and fees from order if present
        slippage = getattr(order, 'slippage', 0.001)  # Default 0.1%
        brokerage_fee = getattr(order, 'brokerage_fee', 0.0035)  # Default 0.35%
        
        if order.status == 'pending':
            # Execute the order in the simulator
            response_order = self.place_order(order, market_data_df=market_data_df)
        elif order.status.lower() in ['open', 'submitted', 'pendingsubmit']:
            # Update the order status in the simulator and check if it's filled
            response_order = self.update_order_status(order, system_timestamp) 
            
            # Apply slippage and fees when order is filled
            if hasattr(response_order, 'fresh_update') and response_order.status == 'closed':
                direction = 1 if response_order.orderDirection == 'BUY' else -1
                if hasattr(response_order, 'filled_price') and response_order.filled_price is not None:
                    # Apply slippage to fill price
                    response_order.filled_price *= (1 + direction * slippage)
                    # Apply brokerage fee
                    response_order.filled_price *= (1 + direction * brokerage_fee)
            
        elif order.status == 'modify':
            response_order = self.modify_order(order, system_timestamp=system_timestamp)
        
        return response_order

    def get_account_summary(self, trading_currency, base_currency, account_number):
        self.check_ib_connection()
        account_summary = self.ib.accountSummary()

        account_balance_dict = {trading_currency:{}}
        account_balance_dict_temp = {}
        for count, item in enumerate(account_summary):
            if item.account == account_number:
                account_balance_dict_temp[item.tag] = {'value': item.value, 'currency': item.currency}
            if item.tag == 'ExchangeRate' and item.currency == trading_currency:
                ExchangeRate = float(item.value)
        
        for key in account_balance_dict_temp:
            if key in ['NetLiquidation', 'BuyingPower', 'GrossPositionValue', 'Cushion']:
                value = float(account_balance_dict_temp[key]['value'])
                currency = account_balance_dict_temp[key]['currency']
                if currency != trading_currency and currency != '':
                    value = value / ExchangeRate
                if key == 'NetLiquidation':
                    account_balance_dict[trading_currency]['total_account_value'] = value
                elif key == 'BuyingPower':
                    account_balance_dict[trading_currency]['buying_power_available'] = value
                elif key == 'GrossPositionValue':
                    account_balance_dict[trading_currency]['buying_power_used'] = value
                elif key == 'Cushion':
                    account_balance_dict[trading_currency]['cushion'] = value

        account_balance_dict[trading_currency]['margin_multiplier'] = (float(account_balance_dict[trading_currency]['buying_power_available']) + float(account_balance_dict[trading_currency]['buying_power_used'])) / (float(account_balance_dict[trading_currency]['pledge_to_margin_used']) + float(account_balance_dict[trading_currency]['pledge_to_margin_availble']))
        account_balance_dict[trading_currency]['total_buying_power'] = float(account_balance_dict[trading_currency]['buying_power_available']) + float(account_balance_dict[trading_currency]['buying_power_used'])
        account_balance_dict[trading_currency]['pct_of_margin_used'] = float(account_balance_dict[trading_currency]['pledge_to_margin_used']) / float(account_balance_dict[trading_currency]['total_account_value'])
        
        # self.logger.debug({'IBKR: account_balance_dict':account_balance_dict})
        
        return account_balance_dict
    
if __name__ == '__main__':
    import time
    
    ibkr = IBKR()
    if not ibkr.connected:
        print('Not connected to IBKR')
        ibkr.connect_to_IBKR()
        if ibkr.connected:
            print('Connection established.')
            acc_summary = ibkr.ib.accountSummary()
            # print({'acc_summary':acc_summary})
    
    tickers = ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'NFLX', 'NVDA', 'PYPL', 'ADBE']
    exchange = 'SMART'
    currency = 'USD'
    while True:
        for ticker in tickers:
            print(f"Fetching historical data for {ticker}")
            time.sleep(5)
            
            contract = ibkr.ib.qualifyContracts(Stock(ticker, exchange, currency))[0]
            barSize = '1d'
            # duration = ibkr.duration_lookup.get(barSize,"1 Y")
            # barSize = ibkr.interval_lookup.get(barSize,"1 day")
            duration = "1 D"
            barSize = "1 min"
            # Fetch historical data for max duration
            bars = ibkr.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            new_data = util.df(bars)
            if not new_data.empty:
                new_data['symbol'] = ticker
                new_data = new_data.rename(columns={
                    'date': 'dateTime',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })[['symbol', 'dateTime', 'open', 'high', 'low', 'close', 'volume']]
            else:
                print(f"No data found for {ticker} at {str(datetime.now())}")
        
            print(new_data)