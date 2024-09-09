import os, sys, time, json
import pandas as pd
import yfinance as yf
from utils import create_logger
import logging
from tqdm import tqdm

class Yahoo():
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
    
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
        
    def update_price_data_single_asset(self, symbol, data_folder='data/yahoo', throttle_secs=1):
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
    
    def update_price_data_batch(self, stock_symbols, start_date, batch_size=75):
        asset_data_df_dict = {}
        for i in range(0, len(stock_symbols), batch_size):
            batch = stock_symbols[i:i + batch_size]
            if len(batch) > 1:
                if start_date is not None:
                    # convert Timestamp to _datetime
                    start_date = start_date.to_pydatetime()
                    self.logger.debug({'start_date to yahoo': start_date})
                    data = yf.download(batch, start=start_date, progress=True)
                else:
                    data = yf.download(batch, period="max", progress=True)
                for ticker in stock_symbols:
                    # if the ticker is not in the combined_data DataFrame, skip it
                    if ticker in data.columns.get_level_values(1):
                        asset_data_df = data.loc[:, data.columns.get_level_values(1) == ticker]
                        asset_data_df_dict[ticker]= asset_data_df
            else:
                asset_data_df = self.update_price_data_single_asset(batch[0])
                asset_data_df_dict[batch[0]] = asset_data_df
            time.sleep(1)  # To avoid hitting rate limits
                    
        return asset_data_df_dict
    
    def update_price_data(self, stock_symbols, data_folder='db/data/yahoo', throttle_secs=1):
        data_frames = []
        pbar = tqdm(stock_symbols, desc='Updating data: ')
        
        # Break the list into two lists. ones that don't have data and ones that have data
        stock_symbols_no_data = []
        stock_symbols_with_data = []
        for symbol in stock_symbols:
            csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
            if not os.path.exists(csv_file_path):
                stock_symbols_no_data.append(symbol)
            else:
                stock_symbols_with_data.append(symbol)
                
        self.logger.debug({'stock_symbols_no_data': stock_symbols_no_data})
        self.logger.debug({'stock_symbols_with_data': stock_symbols_with_data})
            
        # Get the data for the ones that don't have data
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_no_data, start_date=None, batch_size=75)

        for symbol in asset_data_df_dict:
            asset_data_df = asset_data_df_dict[symbol]
            csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
            asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
            cols = list(asset_data_df.columns)
            asset_data_df = asset_data_df.T.reset_index(drop=True).T
            asset_data_df = asset_data_df.set_axis(cols,axis=1)
            asset_data_df.to_csv(csv_file_path)
            data_frames.append(asset_data_df)
            pbar.update(1)
        
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        start_date = None
        for symbol in stock_symbols_with_data:
            csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
            existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
            last_date = existing_data.index.max()
            start_date = last_date if start_date is None else min(start_date, last_date)
        self.logger.debug({'start_date': start_date})
                
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_with_data, start_date=start_date, batch_size=75)
        
        for symbol in asset_data_df_dict:
            asset_data_df = asset_data_df_dict[symbol]
            asset_data_df = asset_data_df.xs(symbol, axis=1, level='Ticker')
            cols = list(asset_data_df.columns)
            asset_data_df = asset_data_df.T.reset_index(drop=True).T
            asset_data_df = asset_data_df.set_axis(cols,axis=1)

            csv_file_path = os.path.join(data_folder, f"{symbol}.csv")
            existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
            # get the start date of asset_data_df
            symbol_start_date = asset_data_df.index.min()
            self.logger.debug({'symbol_start_date': symbol_start_date})
            # prune the existing_data to only include data before the start date
            self.logger.debug(type(existing_data.index[0]))
            self.logger.debug(type(symbol_start_date))
            symbol_start_date = symbol_start_date.to_pydatetime()
            existing_data = existing_data[existing_data.index < symbol_start_date]
            self.logger.debug({'asset_data_df-shape': asset_data_df.shape})
            self.logger.debug({'existing_data-shape': existing_data.shape})
            # concatenate the existing data and the new data
            updated_data = pd.concat([existing_data, asset_data_df])
            updated_data['symbol'] = symbol
            updated_data.to_csv(csv_file_path)
            data_frames.append(updated_data)
            pbar.update(1)
        
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False,inplace=True)
        
        # Set multi-index
        combined_df.set_index(['Date','symbol'],inplace=True)

        combined_df = combined_df.reset_index().pivot_table(values=cols, index=['Date'], columns=['symbol'], aggfunc='mean')
        # combined_df = combined_df.unstack(level='symbol')

        # Sort the index
        combined_df.sort_index(inplace=True)       
        
        return combined_df
    
    def update_price_data_old(self, stock_symbols, data_folder='data/yahoo', throttle_secs=1):
        # Create the data directory if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)
        
        # Initialize an empty list to store DataFrames
        data_frames = []
        
        for symbol in stock_symbols:
            time.sleep(throttle_secs)
            csv_file_path = os.path.join(data_folder, f"{symbol}.csv")

            # If CSV file does not exist, download all available data
            if not os.path.exists(csv_file_path):
                data = yf.download(symbol, period="max", progress=False)
                if not data.empty:
                    data.to_csv(csv_file_path)
                    # add the data to the dataframe
                    df = pd.read_csv(csv_file_path)
                    df['symbol'] = symbol
                    df['date'] = pd.to_datetime(df['Date'])
                    data_frames.append(df)

            # If CSV file exists, update the data
            else:
                # print(f"Updating data for {symbol}...")
                existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
                last_date = existing_data.index.max()
                # self.logger.debug(f"Data for {symbol} was only available till {last_date}. Updating till today now...")

                # Download data from the day after the last date in the CSV until today
                if last_date + pd.Timedelta(days=1) < pd.Timestamp.today():
                    # self.logger.debug({'last_date': last_date, 'today': pd.Timestamp.today()})
                    new_data = yf.download(symbol, start=last_date + pd.Timedelta(days=1), progress=False)
                    if not new_data.empty:
                        updated_data = pd.concat([existing_data, new_data])
                        updated_data.to_csv(csv_file_path)
                        # add the data to the dataframe
                        df = pd.read_csv(csv_file_path)
                        df['symbol'] = symbol
                        df['date'] = pd.to_datetime(df['Date'])
                        data_frames.append(df)
                        
                else:
                    # add the data to the dataframe
                    df = existing_data
                    df['symbol'] = symbol
                    df['date'] = pd.to_datetime(df['Date'])
                    data_frames.append(df)

                    # print(f"Data for {symbol} updated in {csv_file_path}.")
                # else:
                    # print(f"No new data available for {symbol}.")
       
       
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(data_frames)

        # Set multi-index
        combined_df.set_index(['symbol', 'date'], inplace=True)

        # Sort the index
        combined_df.sort_index(inplace=True)       
        
        return combined_df