from math import e
import os, hashlib, time, json, logging, sys
from numpy import isin
import pandas as pd
import yfinance as yf
from systems.utils import create_logger, generate_hash_id, sleeper, project_path
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from colorama import Fore, Style
from systems.utils import MarketDataExtractor

# Main Simulation Class
class Sim():
    def __init__(self):
        self.data = Yahoo()  # Yahoo Finance Data Fetcher
        self.execute = SIM_Execute()  # Order Execution

# Order Execution Class
class SIM_Execute():
    def __init__(self):
        self.logger = create_logger(log_level='DEBUG', logger_name='SIM_Execute', print_to_console=True)
        self.granularity_lookup_dict = {"1m":60,"2m":120,"5m":300,"1d":86400}
        self.market_data_extractor = MarketDataExtractor()
        
    def place_order(self, order, market_data_df, system_timestamp):
        """
        Place a market or stop-loss-limit order. All other order types are rejected.
        """
        symbol = order['symbol']
        granularity = order['granularity']
        current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
        current_system_timestamp = market_data_df.index.get_level_values(1)[-1]
        
        if order['orderType'].lower() in ['market', 'market_exit']:
            response_order = deepcopy(order)
            response_order['status'] = 'closed'
            response_order['fill_price'] = current_price
            response_order['filled_timestamp'] = system_timestamp
            if 'order_id' not in order or order['order_id'] is None:
                order_ = deepcopy(order)
                if 'history' in order_:
                    del order_['history']
                response_order['order_id'] = generate_hash_id(order_, system_timestamp)
            response_order['broker_order_id'] = response_order['order_id']
            response_order['fresh_update'] = True
            response_order['message'] = 'Market Order filled at market price.' if order['orderType'].lower() == 'market' else 'Market Exit Order filled at market price.'
            # if order['orderType'].lower() == 'market_exit':
                # raise AssertionError(f"MARKET EXIT ORDER FILLED. symbol: {response_order['symbol']}")
            
        elif order['orderType'].lower() in ['stoploss_abs', 'stoploss_pct']:
            response_order = deepcopy(order)
            response_order['status'] = 'open'
            response_order['order_id'] = generate_hash_id(order, system_timestamp)
            response_order['broker_order_id'] = response_order['order_id']
            response_order['message'] = 'Stop-loss order placed.'
            response_order['fresh_update'] = True
            
        else:
            response_order = deepcopy(order)
            response_order['status'] = 'rejected'
            response_order['order_id'] = generate_hash_id(order, system_timestamp)
            response_order['broker_order_id'] = response_order['order_id']
            response_order['message'] = 'Order Rejected: Order type not supported.'
            self.logger.warning(f"ORDER REJECTED: Order type '{order['orderType']}' not supported.")
            response_order['fresh_update'] = True
        
        return response_order
    
    def update_order_status(self, order, market_data_df):
        '''if the order is open, check if it's filled'''
        system_timestamp = market_data_df.index.get_level_values(1)[-1]
        symbol = order['symbol']
        # granularity = order['granularity'] if ('granularity' in order and order['granularity'].lower() != 'unknown') else '1d'
        # current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
        min_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
        # close_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'close')
        low_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'low')
        high_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'high')
        # close_prices.dropna(inplace=True)
        # close_prices = close_prices.tolist()
        high_prices.dropna(inplace=True)
        low_prices.dropna(inplace=True)
        high_prices = high_prices.tolist()
        low_prices = low_prices.tolist()
        # current_close_price = close_prices[-1]
        current_low_price = low_prices[-1]
        current_high_price = high_prices[-1]
        
        if order['orderType'].lower() in ['stoploss_abs', 'stoploss_pct']:
            if order['orderDirection'].lower() == 'buy' and current_high_price >= order['exitPrice'] or order['orderDirection'].lower() == 'sell' and current_low_price <= order['exitPrice']:
                response_order = deepcopy(order)
                response_order['status'] = 'closed'
                self.available_granularities = market_data_df.index.get_level_values(0).unique()
                self.min_granularity_val = min([self.granularity_lookup_dict[granularity] for granularity in self.available_granularities])
                self.min_granularity = list(self.granularity_lookup_dict.keys())[list(self.granularity_lookup_dict.values()).index(self.min_granularity_val)]
                fill_price = order['exitPrice'] if self.min_granularity == '1d' else order['exitPrice']
                response_order['fill_price'] = fill_price
                response_order['filled_timestamp'] = system_timestamp
                response_order['fresh_update'] = True
                response_order['message'] = 'Stop-loss order filled.'
                # raise AssertionError(f"STOP-LOSS ORDER FILLED. symbol: {response_order['symbol']}")
            else:
                response_order = deepcopy(order)
                response_order['status'] = 'open'
                response_order['fresh_update'] = False
        else:
            response_order = deepcopy(order)
            response_order['status'] = 'rejected'
            response_order['fresh_update'] = True
            response_order['message'] = 'Order Rejected: Order type not supported.'
            self.logger.warning(f"ORDER REJECTED: Order type {order['orderType']} not supported.")
        
        return response_order
    
    def modify_order(self, order, system_timestamp):
        # if order['modify_reason'] in ['new_price', 'new_quantity', 'new_price+new_quantity']:
        for modify_reason in order['modify_reason']:
            assert modify_reason.lower() in ['new_price', 'new_quantity']
        order['status'] = 'open'
        order['modified_timestamp'] = system_timestamp
        order['fresh_update'] = True
        
        return order
    
    def cancel_order(self, order, system_timestamp):
        pass
           
    def execute_order(self, order, market_data_df, system_timestamp):
        if order['status'] == 'pending':
            # Execute the order in the simulator
            response_order = self.place_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
        elif order['status'] == 'open':
            # Update the order status in the simulator and check if it's filled
            response_order = self.update_order_status(order, market_data_df=market_data_df)
        elif order['status'] == 'modify':
            response_order = self.modify_order(order, system_timestamp=system_timestamp)
        # elif order['status'] == 'cancel':
        #     response_order = self.cancel_order(order, system_timestamp=system_timestamp)
        else:
            self.logger.debug({'order':order})
            raise ValueError(f"Order status '{order['status']}' not supported.")
        
        return response_order
    
    def create_account_summary(self, trading_currency, base_currency, base_currency_to_trading_currency_exchange_rate, starting_account_inputs):
        account_balance_dict = {trading_currency:{}}
        for currency in starting_account_inputs:
            if base_currency != trading_currency and currency == base_currency:
                base_currency_account_balance_dict = {}
                base_currency_account_balance_dict[currency] = starting_account_inputs[currency]
                base_currency_account_balance_dict[currency]['margin_multiplier'] = (float(base_currency_account_balance_dict[currency]['buying_power_available']) + float(base_currency_account_balance_dict[currency]['buying_power_used'])) / (float(base_currency_account_balance_dict[currency]['pledge_to_margin_used']) + float(base_currency_account_balance_dict[currency]['pledge_to_margin_available']))
                base_currency_account_balance_dict[currency]['total_buying_power'] = float(base_currency_account_balance_dict[currency]['buying_power_available']) + float(base_currency_account_balance_dict[currency]['buying_power_used'])
                base_currency_account_balance_dict[currency]['pct_of_margin_used'] = float(base_currency_account_balance_dict[currency]['pledge_to_margin_used']) / float(base_currency_account_balance_dict[currency]['total_account_value'])
                
                for key, value in base_currency_account_balance_dict[currency].items():
                    if key not in ['cushion', 'margin_multipler', 'pct_of_margin_used']:
                        account_balance_dict[trading_currency][key] = round(value * base_currency_to_trading_currency_exchange_rate, 2)
                    else:
                        account_balance_dict[trading_currency][key] = round(value, 2)
        
        # account_balance_dict = starting_account_inputs
        # account_balance_dict['margin_multipler'] = {'value': (float(account_balance_dict['buying_power_available']['value']) + float(account_balance_dict['buying_power_used']['value'])) / (float(account_balance_dict['pledge_to_margin_used']['value']) + float(account_balance_dict['pledge_to_margin_availble']['value'])), 'currency': ''}
        # account_balance_dict['total_buying_power'] = {'value': float(account_balance_dict['buying_power_available']['value']) + float(account_balance_dict['buying_power_used']['value']), 'currency': 'CAD'}
        # account_balance_dict['pct_of_margin_used'] = {'value': float(account_balance_dict['pledge_to_margin_used']['value']) / float(account_balance_dict['total_account_value']['value']), 'currency': ''}
        
        return account_balance_dict
    
class Yahoo():
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
        self.asset_data_df_dict = {}
    
    def get_nasdaq_stock_symbols(self, nasdaq_csv_filepath, min_market_cap=10 * 1 * 1000 * 1000 * 1000):
        '''
        nasdaq_csv_filepath = '/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/stocksymbolslists/nasdaq_screener_1725835471552.csv'
        pruned_df, stock_symbols = get_nasdaq_stock_symbols(nasdaq_csv_filepath, min_market_cap=10 * 1 * 1000 * 1000 * 1000)
        print({'stock_symbols':stock_symbols})
        print({'stock_symbols':len(stock_symbols)})
        '''
        # load the CSV file into a DataFrame
        nasdaq_df = pd.read_csv(nasdaq_csv_filepath)
        # get the filename of the CSV file
        filename = os.path.basename(nasdaq_csv_filepath)
        # get the timestamp of the CSV file
        timestamp = int(filename.split('_')[-1].split('.')[0])
        age = (time.time() * 1000) - timestamp
        # Convert age from milliseconds to days
        age_days = age / (1000 * 60 * 60 * 24)
        # Print the age of the file in days
        print(f'The file {filename} is {age_days:.2f} days old.')
        if age_days > 7:
            print('The file is more than 7 days old. Consider updating the data.')
        
        # extract the stock symbols from the DataFrame where the 'Market Cap' is greater than $10 billion
        pruned_df = nasdaq_df[nasdaq_df['Market Cap'] > min_market_cap].copy()

        # Sort the DataFrame by 'Market Cap' in descending order
        pruned_df.sort_values('Market Cap', ascending=False, inplace=True)
        # Get the list of stock symbols
        stock_symbols = pruned_df['Symbol'].tolist()
        # save the stock symbols to a json file
        with open('stock_symbols.json', 'w') as file:
            json.dump(stock_symbols, file)
        
        return pruned_df, stock_symbols
        
    def update_price_data_single_asset(self, symbol, data_folder=project_path+'/db/data/yahoo', throttle_secs=0.25):
        os.makedirs(data_folder, exist_ok=True)
        time.sleep(throttle_secs)
        csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
        # If CSV file does not exist, download all available data
        if not os.path.exists(csv_file_path):
            data = yf.download(symbol, period="max", interval='1d', progress=False)
            if not data.empty:
                data.to_csv(csv_file_path)
            else:
                return None
        # If CSV file exists, update the data
        else:
            # print(f"Updating data for {symbol}...")
            existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
            last_date = existing_data.index.max()
            # self.logger.debug(f"Data for {symbol} was only available till {last_date}. Updating till today now...")

            # Download data from the day after the last date in the CSV until today
            if last_date + pd.Timedelta(days=1) < pd.Timestamp.today():
                # self.logger.debug({'last_date': last_date, 'today': pd.Timestamp.today()})
                new_data = yf.download(symbol, interval='1d',start=last_date + pd.Timedelta(days=1), progress=False)
                if not new_data.empty:
                    updated_data = pd.concat([existing_data, new_data])
                    updated_data.to_csv(csv_file_path)
        
        asset_data_df = pd.read_csv(csv_file_path)
        asset_data_df['symbol'] = symbol
        asset_data_df['date'] = pd.to_datetime(asset_data_df['Date'])
        
        return asset_data_df
    
    def restructure_asset_data_df(self, asset_data_df):
        if not asset_data_df.empty:
            asset_data_df.columns = asset_data_df.columns.str.lower()
            asset_data_df.index.names = ['datetime']
            asset_data_df.index = asset_data_df.index.tz_convert('UTC')
            cols = list(asset_data_df.columns)
            asset_data_df = asset_data_df.T.reset_index(drop=True).T
            asset_data_df = asset_data_df.set_axis(cols, axis=1)
        
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
    
    def batch_update_price_data(self, stock_symbols, interval, start_date):
        asset_data_df_dict = {}
        if len(stock_symbols) == 1:
            stock_symbols.append('AAPL')
            dummy_addition = True
        if len(stock_symbols) > 1:
            # sleeper(5, "Giving you 5 seconds to read the above message...")
            # self.logger.info(f"÷Downloading Yahoo data for 'Start Date': {start_date}, 'Interval': {interval}, Batch Size: {stock_symbols}")
            
            if start_date is not None and ((interval == '1m' and time.time() - start_date.timestamp() < 60*60*24*7) or (interval == '1d')):
                # convert Timestamp to _datetime
                # start_date = start_date.to_pydatetime()
                data = yf.download(stock_symbols, start=start_date, progress=False,interval=interval)
            else:
                data = yf.download(stock_symbols, period="max", progress=False,interval=interval)
            # self.logger.info({'data':data})
            for ticker in stock_symbols:
                if ticker in data.columns.get_level_values(1):
                    asset_data_df = data.loc[:, data.columns.get_level_values(1) == ticker]
                    asset_data_df_dict[ticker] = asset_data_df

        return asset_data_df_dict
        
    def update_price_data(self, stock_symbols, interval_inputs, data_folder=project_path+'/db/data/yahoo', throttle_secs=1, start_date=None, end_date=None, lookback=None, update_data=True, run_mode=4):
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
                        # Add timedelta 1 day to the index column
                        if interval == '1d':
                            existing_data.index = existing_data.index + pd.Timedelta(days=1)
                        if not existing_data.empty:
                            # existing_data_first_date = existing_data.index.min().tz_convert('UTC')
                            existing_data_last_date = existing_data.index.max().tz_convert('UTC')
                            # self.logger.debug({'symbol':symbol, 'existing_data_last_date':existing_data_last_date.astimezone('US/Eastern')})
                            yday_date = pd.Timestamp.today().tz_localize('UTC').replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
                            # replace hours, minutes, seconds with 0
                            # if interval == '1d':
                                # self.logger.debug({'symbol':symbol, 'interval':interval,'existing_data_last_date':existing_data_last_date, 'yday_date':yday_date, 'end_date':end_date, 'timestamp_test_bool':existing_data_last_date >= yday_date})
                            if (end_date is not None and existing_data_last_date >= end_date) or (existing_data_last_date >= yday_date and interval == '1d'):
                            # if end_date is not None and existing_data_last_date >= end_date:
                                # prune the data using the back_test_start_date and back_test_end_date
                                existing_data = existing_data.loc[:end_date]
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
            #     self.logger.debug({'partial_data_symbols':stock_symbols_with_partial_data[interval]})
        
        '''STEP 2: Get the data for the ones that don't have data'''
        for interval in interval_inputs:
            if len(stock_symbols_no_data[interval]) > 0:
                pbar = tqdm(stock_symbols_no_data[interval], desc=f'Yahoo - Updating NO data: Interval: {interval}')
                # pbar_ydownloader = tqdm(stock_symbols[interval], desc= f'Downloading data from Yahoo: Interval: {interval}: ')
                for i in range(0, len(stock_symbols_no_data[interval]), batch_size):
                    batch = stock_symbols_no_data[interval][i:i + batch_size]
                    batch_asset_data_df_dict = self.batch_update_price_data(batch, interval, start_date=None)
                    # Extract the data from yahoo batch response and save it to csv
                    data_input_folder = os.path.join(data_folder, interval)
                    os.makedirs(data_input_folder, exist_ok=True)
                    for symbol in batch_asset_data_df_dict:
                        # Extract the data from the yahoo batch response
                        asset_data_df = batch_asset_data_df_dict[symbol]
                        asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                        asset_data_df = self.restructure_asset_data_df(asset_data_df)
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
                        # self.logger.info(f"Trying to save to {csv_file_path}")
                        if not asset_data_df.empty:
                            # self.logger.info({f"Saved {symbol}|{interval} data to {csv_file_path}"})
                            asset_data_df.to_csv(csv_file_path)
                        # else:
                            # self.logger.warning(f"Data for {symbol} is empty.")

                        # Remove all cols not needed
                        asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                        
                        # Update it to the data_frames list
                        data_frames.append(asset_data_df)
                    pbar.update(len(batch))
                    if throttle_secs < 1:
                        self.logger.warning('Throttle seconds is less than 1 to give Yahoo API time to breathe.')
                        time.sleep(throttle_secs)
                    else:
                        sleeper(throttle_secs, 'Giving Yahoo API time to breathe.')  # To avoid hitting rate limits
                pbar.close()
        
        '''STEP 3: Get the data for the ones that have partial data'''
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        for interval in interval_inputs:
            if len(stock_symbols_with_partial_data[interval]) > 0:
                pbar = tqdm(stock_symbols_with_partial_data[interval], desc=f'Updating Partial data: Interval: {interval}')
                for i in range(0, len(stock_symbols_with_partial_data[interval]), batch_size):
                    batch = stock_symbols_with_partial_data[interval][i:i + batch_size]
                    batch_start_date = None
                    for symbol in batch:
                        existing_data = existing_data_dict[interval][symbol]
                        # last_date = existing_data.index.max().tz_localize('UTC') if not existing_data.empty else None
                        last_date = existing_data.index.max().tz_convert('UTC') if not existing_data.empty else None
                        batch_start_date = last_date if batch_start_date is None and last_date is not None else None if last_date is None else min(batch_start_date, last_date)
                    
                    batch_asset_data_df_dict = self.batch_update_price_data(batch, interval, start_date=batch_start_date)
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
                        asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                        asset_data_df = self.restructure_asset_data_df(asset_data_df)
                        asset_data_df = asset_data_df.dropna(how='all')
                        existing_data = existing_data_dict[interval][symbol]
                        # get the start date of asset_data_df
                        symbol_start_date = asset_data_df.index.min() if not asset_data_df.empty else start_date
                        # prune the existing_data to only include data before the start date
                        # symbol_start_date = symbol_start_date.to_pydatetime()
                        # self.logger.debug({'symbol':symbol, 'symbol_start_date':symbol_start_date})
                        existing_data = existing_data[existing_data.index < str(symbol_start_date)]
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
                        
                        if interval == '1d':
                            today = pd.Timestamp(datetime.now()).tz_localize('UTC')
                            today = today.replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(minutes=1)
                            updated_data = updated_data.loc[:today]
                        
                        # updated_data = updated_data.iloc[:-1] if interval == '1d' else updated_data
                        # self.logger.info({'symbol':symbol, 'existing_data':existing_data.shape, 'updated_data':updated_data.shape, 'asset_data_df':asset_data_df.shape})
                        # Save it to the csv file
                        # if update_data is not an empty DataFrame, then save it to the csv file
                        
                        # self.logger.info(f"Trying to save to {csv_file_path}")
                        if not updated_data.empty and not asset_data_df.empty:
                            symbol = symbol.replace('/','-') if '/' in symbol else symbol
                            csv_file_path = os.path.join(data_folder, interval, f"{symbol}.csv")
                            # self.logger.info({f"Saved {symbol}|{interval} data to {csv_file_path}: asset_data_df: {asset_data_df.tail(2)}"})
                            updated_data.to_csv(csv_file_path)
                        # else:
                            # self.logger.warning(f"Data for {symbol} is empty.")
                        
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
                        sleeper(throttle_secs, f'Giving Yahoo API time to breathe. throttle_secs: {throttle_secs}')  # To avoid hitting rate limits
                        
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
                    asset_data_df = asset_data_df.iloc[:-1] if interval == '1d' else asset_data_df
                    
                    # Remove all cols not needed
                    asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                    
                    # Update it to the data_frames list
                    data_frames.append(asset_data_df)
                    pbar.update(1)
                pbar.close()
        
        '''Step 6: Add 1d timedelta to date'''
        for dataframe in data_frames:
            if '1d' in dataframe['interval'].unique():
                dataframe.index = dataframe.index + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        '''Step 7: Trim the data to the back_test_start_date and back_test_end_date'''
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
        
        '''STEP 8: # Combine all DataFrames into a single DataFrame'''
        # self.logger.info('Combining all DataFrames into a single DataFrame...')
        # self.logger.info({'data_frame':data_frames[0]})
        
        # Remove empty dataframes from the list data_frames
        data_frames = [df for df in data_frames if not df.empty]
        if len(data_frames) > 0:
            combined_df = pd.concat(data_frames)
            combined_df.reset_index(drop=False,inplace=True)
            # Set multi-index
            combined_df.set_index(['date','symbol'],inplace=True)
            asset_data_df = data_frames[0]
            pass_cols = list(asset_data_df.columns)
            # self.logger.info({'combined_df':combined_df})
            combined_df = combined_df.reset_index().pivot_table(values=pass_cols, index=['interval', 'date'], columns=['symbol'], aggfunc='mean')
            # combined_df = combined_df.unstack(level='symbol')
            # Sort the index
            combined_df.sort_index(inplace=True)
            '''STEP 9: Add a column for data_source= 'yahoo' '''
            combined_df['data_source'] = 'yahoo'
        else:
            combined_df = pd.DataFrame()
        # self.logger.info({'combined_df':combined_df})
        # raise AssertionError('STOP HERE')
        
        
        # self.logger.info({'combined_df':combined_df})
        # raise AssertionError('STOP HERE')
        
        return combined_df