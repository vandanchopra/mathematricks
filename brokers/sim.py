from re import A
import os, hashlib, time, json, logging, sys
from turtle import back
import pandas as pd
import yfinance as yf
from systems.utils import create_logger, generate_hash_id
from tqdm import tqdm
from copy import deepcopy

# Main Simulation Class
class Sim():
    def __init__(self):
        self.data = Yahoo()  # Yahoo Finance Data Fetcher
        self.execute = SIM_Execute()  # Order Execution

# Order Execution Class
class SIM_Execute():
    def __init__(self):
        self.logger = create_logger(log_level='DEBUG', logger_name='SIM_Execute', print_to_console=True)

    def place_order(self, order, market_data_df, system_timestamp):
        """
        Place a market or stop-loss-limit order. All other order types are rejected.
        """
        symbol = order['symbol']
        granularity = order['granularity']
        current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]
        current_system_timestamp = market_data_df.index.get_level_values(1)[-1]
        
        if order['orderType'].lower() == 'market':
            response_order = deepcopy(order)
            response_order['status'] = 'closed'
            response_order['fill_price'] = current_price
            response_order['filled_timestamp'] = system_timestamp
            if 'order_id' not in order or order['order_id'] is None:
                order_without_history = deepcopy(order)
                if 'history' in order_without_history:
                    del order_without_history['history']
                response_order['order_id'] = generate_hash_id(order_without_history, system_timestamp)
            response_order['broker_order_id'] = response_order['order_id']
            response_order['fresh_update'] = True
            response_order['message'] = 'Order filled at market price.'
            
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
        granularity = order['granularity']
        current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]
        
        if order['orderType'].lower() in ['stoploss_abs', 'stoploss_pct']:
            if order['orderDirection'].lower() == 'buy' and current_price >= order['exitPrice'] or order['orderDirection'].lower() == 'sell' and current_price <= order['exitPrice']:
                response_order = deepcopy(order)
                response_order['status'] = 'closed'
                response_order['fill_price'] = current_price
                response_order['filled_timestamp'] = system_timestamp
                response_order['fresh_update'] = True
                response_order['message'] = 'Stop-loss order filled.'
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
        order['status'] = 'open'
        order['modified_timestamp'] = system_timestamp
        order['message'] = 'Order modified.'
        order['fresh_update'] = True
        
        return order
       
    def execute_order(self, order, market_data_df, system_timestamp):
        if order['status'] == 'pending':
            # Execute the order in the simulator
            response_order = self.place_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
        elif order['status'] == 'open':
            # Update the order status in the simulator and check if it's filled
            response_order = self.update_order_status(order, market_data_df=market_data_df)
        elif order['status'] == 'modify':
            response_order = self.modify_order(order, system_timestamp=system_timestamp)
        
        return response_order
    
class Yahoo():
    def __init__(self):
        self.logger = create_logger(log_level=logging.INFO, logger_name='datafetcher', print_to_console=True)
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
        
    def update_price_data_single_asset(self, symbol, data_folder='data/yahoo', throttle_secs=0.25):
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
    
    def update_price_data_batch(self, stock_symbols, start_date, pbar, batch_size=75, throttle_secs=0.25):
        asset_data_df_dict = {}
        for interval in stock_symbols:
            asset_data_df_dict[interval] = {}
            for i in range(0, len(stock_symbols[interval]), batch_size):
                batch = stock_symbols[interval][i:i + batch_size]
                if len(batch) == 1:
                    batch.append('AAPL')
                    dummy_addition = True
                if len(batch) > 1:
                    if start_date is not None:
                        # convert Timestamp to _datetime
                        # start_date = start_date.to_pydatetime()
                        data = yf.download(batch, start=start_date, progress=False,interval=interval)
                    else:
                        data = yf.download(batch, period="max", progress=False,interval=interval)
                    
                    for ticker in stock_symbols[interval]:
                        if ticker in data.columns.get_level_values(1):
                            asset_data_df = data.loc[:, data.columns.get_level_values(1) == ticker]
                            asset_data_df_dict[interval][ticker]= asset_data_df
                            pbar.update(1)
                            
                time.sleep(throttle_secs)  # To avoid hitting rate limits
                        
        return asset_data_df_dict
    
    def restructure_asset_data_df(self, asset_data_df):
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
    
    def update_price_data(self, stock_symbols, interval_inputs, data_folder='db/data/yahoo', throttle_secs=0.25, back_test_start_date=None, back_test_end_date=None, lookback=None, update_data=True):
        data_frames = []

        # Break the list into two lists. ones that don't have data and ones that have data
        stock_symbols_no_data = { k:[] for k in interval_inputs}
        stock_symbols_with_partial_data = { k:[] for k in interval_inputs}
        stock_symbols_with_full_data = { k:[] for k in interval_inputs}
        existing_data_dict = { k:{} for k in interval_inputs}
        '''STEP 1: Bifurcate the list of stock symbols into two lists: 1) ones that have data 2) ones that don't have data 3) Ones that have partial data'''
        for interval in interval_inputs:
            csv_loader_pbar = tqdm(stock_symbols, desc=f'Fetching {interval} CSV data: ')
            
            for symbol in stock_symbols:
                csv_file_path = os.path.join(data_folder, interval, f"{symbol}.csv")
                if not os.path.exists(csv_file_path):
                    stock_symbols_no_data[interval].append(symbol)
                else:
                    try:
                        existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                        if not existing_data.empty:
                            # existing_data_first_date = existing_data.index.min().tz_convert('UTC')
                            existing_data_last_date = existing_data.index.max().tz_convert('UTC')
                            if back_test_end_date is not None and existing_data_last_date >= back_test_end_date:
                                # prune the data using the back_test_start_date and back_test_end_date
                                existing_data = existing_data.loc[:back_test_end_date]
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
        
        pbar = tqdm(stock_symbols, desc='Updating data: ')
        '''STEP 2: Get the data for the ones that don't have data'''
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_no_data, start_date=None, pbar=pbar, batch_size=75, throttle_secs=throttle_secs)

        # Extract the data from yahoo batch response and save it to csv
        for interval in asset_data_df_dict:
            data_input_folder = os.path.join(data_folder, interval)
            os.makedirs(data_input_folder, exist_ok=True)
            for symbol in asset_data_df_dict[interval]:
                # Extract the data from the yahoo batch response
                asset_data_df = asset_data_df_dict[interval][symbol]
                asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                asset_data_df = self.restructure_asset_data_df(asset_data_df)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval
                asset_data_df = asset_data_df.dropna(how='all')
                
                # Save the data to a csv file
                csv_file_path = os.path.join(data_input_folder, f"{symbol}.csv")
                asset_data_df.to_csv(csv_file_path)
                
                # Remove all cols not needed
                asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                
                # Update it to the data_frames list
                data_frames.append(asset_data_df)

        '''STEP 3: Get the data for the ones that have partial data'''
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        start_date = None
        for interval in stock_symbols_with_partial_data:
            for symbol in stock_symbols_with_partial_data[interval]:
                existing_data = existing_data_dict[interval][symbol]
                # last_date = existing_data.index.max().tz_localize('UTC') if not existing_data.empty else None
                last_date = existing_data.index.max().tz_convert('UTC') if not existing_data.empty else None
                start_date = last_date if start_date is None and last_date is not None else None if last_date is None else min(start_date, last_date)
                # self.logger.debug({'interval':interval, 'symbol': symbol, 'last_date': last_date, 'start_date': start_date})
        # self.logger.debug({'start_date': start_date})
                
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_with_partial_data, start_date=start_date,  pbar=pbar, batch_size=75)

        for interval in asset_data_df_dict:
            for symbol in asset_data_df_dict[interval]:
                
                # Extract the data from the yahoo batch response
                asset_data_df = asset_data_df_dict[interval][symbol]
                asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
                asset_data_df = self.restructure_asset_data_df(asset_data_df)
                
                existing_data = existing_data_dict[interval][symbol]
                # get the start date of asset_data_df
                symbol_start_date = asset_data_df.index.min()
                # prune the existing_data to only include data before the start date
                symbol_start_date = symbol_start_date.to_pydatetime()
                existing_data = existing_data[existing_data.index < symbol_start_date]
                # concatenate the existing data and the new data
                updated_data = pd.concat([existing_data, asset_data_df])
                updated_data = updated_data.dropna(how='all')
                updated_data['symbol'] = symbol
                updated_data['interval'] = interval
                
                # Save it to the csv file
                csv_file_path = os.path.join(data_input_folder, f"{symbol}.csv")
                updated_data.to_csv(csv_file_path)
                
                # Remove all cols not needed
                updated_data = self.remove_unwanted_cols(interval_inputs, interval, updated_data)
                
                # Update it to the data_frames list
                data_frames.append(updated_data)
        
        '''STEP 4: Get the data for the ones that have full data'''
        for interval in interval_inputs:
            for symbol in existing_data_dict[interval]:
                asset_data_df = existing_data_dict[interval][symbol]
                asset_data_df = self.restructure_asset_data_df(asset_data_df)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval
                
                # Remove all cols not needed
                asset_data_df = self.remove_unwanted_cols(interval_inputs, interval, asset_data_df)
                
                # Update it to the data_frames list
                data_frames.append(asset_data_df)
                pbar.update(1)
        pbar.close()
        
        '''STEP 5: # Combine all DataFrames into a single DataFrame'''
        self.logger.debug('Combining all DataFrames into a single DataFrame...')
        start = time.time()
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False,inplace=True)
        
        # Set multi-index
        combined_df.set_index(['datetime','symbol'],inplace=True)
        
        pass_cols = list(asset_data_df.columns)
        combined_df = combined_df.reset_index().pivot_table(values=pass_cols, index=['interval', 'datetime'], columns=['symbol'], aggfunc='mean')
        # combined_df = combined_df.unstack(level='symbol')

        # Sort the index
        combined_df.sort_index(inplace=True)
        self.logger.debug(f'DONE:  Combining all DataFrames into a single DataFrame....Time Taken: {(time.time() - start)/60} minutes.')

        '''STEP 6: Trim the data to the back_test_start_date and back_test_end_date'''
        if lookback is not None:
            joined_dict = {}
            for interval, lookback_value in lookback.items():
                lookback_value = int(lookback_value*1.5)
                before = combined_df.loc[interval].loc[:back_test_start_date]
                after = combined_df.loc[interval].loc[back_test_start_date:back_test_end_date]
                before_new = before.iloc[-lookback_value:]
                # self.logger.debug({'interval':interval, 'lookback_value':lookback_value, 'first_row':before_new.index[0], 'last_row':before_new.index[-1]}) 
                
                # join before and after dataframes
                joined = pd.concat([before_new, after])
                
                joined.sort_index(inplace=True)
                joined_dict[interval] = joined
            combined_df = pd.concat(joined_dict.values(), keys=joined_dict.keys(), names=['interval', 'date'])
            # raise AssertionError('STOP HERE')
        
        return combined_df
    