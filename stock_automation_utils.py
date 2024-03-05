import os
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
import time
from tqdm import tqdm

class StockAutomationUtils:
    def __init__(self):
        pass

    def add_stock_symbols_to_pickle(self, symbols, filename):
        # Check if the pickle file already exists
        if os.path.exists(filename):
            # Load the existing list of symbols from the file
            with open(filename, 'rb') as file:
                existing_symbols = pickle.load(file)
            # Update the list with new symbols, avoiding duplicates
            updated_symbols = list(set(existing_symbols + symbols))
        else:
            # If the file doesn't exist, create a new list of symbols
            updated_symbols = symbols
        
        # Save the updated list of symbols to the pickle file
        with open(filename, 'wb') as file:
            pickle.dump(updated_symbols, file)

        print(f"Stock symbols updated in {filename}.")
    
    def get_stock_data(self, symbol, start_date_d, end_date_d):
        # Load data from 'data' folder
        csv_path = f'data/{symbol}.csv'
        df = pd.read_csv(csv_path)
        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        return df[(df['Date'] >= start_date_d) & (df['Date'] <= end_date_d)]

    # Load all csv files into a dictionary
    def load_all_stock_data(self, symbols, start_date, end_date):
        stock_data = {}
        for symbol in symbols:
            try:
                stock_data[symbol] = self.get_stock_data(symbol, start_date, end_date)
            except Exception as e:
                print(f'Error loading data for {symbol}: {e}')
        return stock_data
        
    def update_stock_data(self, pickle_file='stock_symbols.pkl', data_folder='data', throttle_secs=1):
        # Check if the pickle file exists
        if not os.path.exists(pickle_file):
            print(f"No pickle file found at {pickle_file}. Please make sure the file exists.")
            return

        # Create the data directory if it doesn't exist
        os.makedirs(data_folder, exist_ok=True)

        # Load the list of stock symbols from the pickle file
        with open(pickle_file, 'rb') as file:
            stock_symbols = pickle.load(file)

        with tqdm(total=len(stock_symbols)) as pbar:
            for symbol in stock_symbols:
                pbar.set_description(f"Downloading data for {symbol}: ")
                csv_file_path = os.path.join(data_folder, f"{symbol}.csv")

                # If CSV file does not exist, download all available data
                if not os.path.exists(csv_file_path):
                
                    data = yf.download(symbol, period="max")
                    
                    if not data.empty:
                        time.sleep(throttle_secs)
                        data.to_csv(csv_file_path)
                        print(f"Data for {symbol} saved to {csv_file_path}.")
                    else:
                        print(f"No data found for {symbol}.")

                # If CSV file exists, update the data
                else:
                    print(f"Updating data for {symbol}...")
                    existing_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)
                    last_date = existing_data.index.max()
                    # pbar.set_postfix(f"Data for {symbol} was only available till {last_date}. Updating...")
                    if datetime.now().date() == last_date + pd.offsets.BDay():
                        pass
                    else:
                        # Download data from the day after the last date in the CSV until today
                        new_data = yf.download(symbol, start=last_date + pd.offsets.BDay())

                        if not new_data.empty:
                            time.sleep(throttle_secs)
                            updated_data = pd.concat([existing_data, new_data])
                            updated_data.iloc[:-1].to_csv(csv_file_path)
                            print(f"Data for {symbol} updated in {csv_file_path}.")
                        else:
                            print(f"No new data available for {symbol}.")

                    # ############################################################
                    # # Alternative solution, does not take weekends into account:
                    #
                    # # Send an API request to download new data regardless of today's date
                    # new_data = yf.download(symbol, start=last_date)
                    #
                    # # Should always pass this and never get to else, but just for safety:
                    # if not data.empty:
                    #     time.sleep(throttle_secs)
                    #
                    #     # Check to see if last_date is today's date,
                    #     # which means multiple downloads in the same day,
                    #     # so we need to drop the last date's row from existing data
                    #     if last_date == datetime.now().date():
                    #         existing_data = existing_data.drop(last_date)
                    #     else:
                    #         # last_date is not today's date, therefore we need to drop
                    #         # it from new data, to avoid changing data we already may have
                    #         # used to make trading decisions
                    #         new_data = new_data.drop(last_date)
                    #
                    #     # Concatenate the two dataframes
                    #     updated_data = pd.concat([existing_data, new_data])
                    #     updated_data.to_csv(csv_file_path)
                    #     print(f"Data for {symbol} updated in {csv_file_path}.")
                    # else:   # Should never get here as we are downloading data regardless of today's date
                    #     print(f"No new data available for {symbol}.")
                pbar.update(1)

    def get_data_for_interval(self, symbols, symbols_df, start_date, end_date):
        pruned_data = {}
        for symbol in symbols:
            try:
                pruned_data[symbol] = symbols_df[symbol][(symbols_df[symbol]['Date'] >= start_date) & (symbols_df[symbol]['Date'] <= end_date)]
            except Exception as e:
                print(f'Error loading data for {symbol}: {e}')
        return pruned_data

