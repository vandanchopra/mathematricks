from ib_insync import IB, Stock, util
import nest_asyncio
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

nest_asyncio.apply()

class IBKR():
    def __init__(self, ib):
        if ib is None:
            self.ib = IB()
        else:
            self.ib = ib
        self.interval_lookup = {"1m": "1 min", "2m": "2min", "5m":"5 min","1d":"1 day"}

    def connect_to_IBKR(self):
        '''NOTE: First start the TWS or Gateway software from IBKR'''

        ib = self.ib

        # Connect to the IBKR TWS (Trader Workstation) or Gateway
        ib.connect('127.0.0.1', 7496, clientId=0)

        # Check if the connection is successful
        if ib.isConnected():
            print('Connected to IBKR')
        else:
            print('Failed to connect to IBKR')
        return ib

    def get_current_portfolio(self):
        # Request the current portfolio from IBKR
        portfolio = self.ib.portfolio()

        # Return the portfolio
        return portfolio

    def get_current_price(self, ticker: str, exchange: str, currency: str):
        # Request the current market data for AAPL
        contract = self.ib.qualifyContracts(Stock(ticker, exchange, currency))
        ticker = self.ib.reqTickers(contract)[0]

        # Get the current price
        current_price = ticker.marketPrice()

        # Return the current price
        return current_price

    def get_current_stop_loss_orders(self):
        # Request the current stop loss orders from IBKR
        all_open_orders = self.ib.openOrders()
        stop_loss_orders = []
        for open_order in all_open_orders:
            if open_order.orderType == 'STP':
                stop_loss_orders.append(open_order)

        # Return the stop loss orders
        return stop_loss_orders

    def place_order(self, ticker: str, exchange: str, currency: str, action: str, quantity: int, order_type: str, limit_price: float = 0, stop_price: float = 0):
        # Create a contract for the stock
        ib = self.ib
        contract = ib.qualifyContracts(Stock(ticker, exchange, currency))

        # Create an order for the stock
        if order_type == 'MKT':
            order = ib.marketOrder(action, quantity)
        elif order_type == 'LMT':
            order = ib.LimitOrder(action, quantity, limit_price)
        elif order_type == 'STP':
            order = ib.StopOrder(action, quantity, stop_price)

        # Place the order
        trade = ib.placeOrder(contract, order)

        # Return the trade
        return trade

    def cancel_order(self, trade):
        # Cancel the order
        self.ib.cancelOrder(trade)

        # Return the trade
        return trade

    def modify_order(self, trade, new_quantity: int, new_limit_price: float, new_stop_price: float):
        # Modify the order
        self.ib.modifyOrder(trade, new_quantity, new_limit_price, new_stop_price)

        # Return the trade
        return trade
    
    def update_price_data_batch(self,stock_symbols, start_date=None):
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
        return asset_data_df_dict

    def fetch_historical_data(self,ticker,barSize,start=None,exchange:str = "SMART", currency:str = "USD"):
        ib = self.ib
        contract = ib.qualifyContracts(Stock(ticker, exchange, currency))
        ib.qualifyContracts(contract)
        if start is not None:
            start_time_dt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            duration = (datetime.now() - start_time_dt).days
            duration = f'{duration} D'
        else:
            duration = "max"
            
        # Fetch historical data for max duration
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        # Convert to DataFrame
        df = util.df(bars)
        return df

    def update_price_data(self, stock_symbols,interval_inputs=['1d'], data_folder='db/data/ibkr', throttle_secs=1):
        data_frames = []
        pbar = tqdm(stock_symbols, desc='Updating data: ')

        # Break the list into two lists. ones that don't have data and ones that have data
        stock_symbols_no_data = { k:[] for k in interval_inputs}
        stock_symbols_with_data = { k:[] for k in interval_inputs}
        for interval in interval_inputs:
            for symbol in stock_symbols:
                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                if not os.path.exists(csv_file_path):
                    stock_symbols_no_data[interval].append(symbol)
                else:
                    stock_symbols_with_data[interval].append(symbol)
                
        self.logger.debug({'stock_symbols_no_data': stock_symbols_no_data})
        self.logger.debug({'stock_symbols_with_data': stock_symbols_with_data})
            
        # Get the data for the ones that don't have data
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_no_data, start_date=None)

        for interval in asset_data_df_dict:
            data_input_folder = os.path.join(data_folder,interval)
            if not os.path.exists(data_input_folder):
                os.makedirs(data_input_folder)
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                csv_file_path = os.path.join(data_input_folder, f"{symbol}.csv")

                asset_data_df.set_index(['date'],inplace=True)
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval
                asset_data_df.to_csv(csv_file_path)
                data_frames.append(asset_data_df)
                pbar.update(1)
        
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        start_date = None
        for interval in stock_symbols_with_data:
            for symbol in stock_symbols_with_data[interval]:
                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='date', parse_dates=True)
                last_date = existing_data.index.max()
                start_date = last_date if start_date is None else min(start_date, last_date)
                
        self.logger.debug({'start_date': start_date})
                
        asset_data_df_dict = self.update_price_data_batch(stock_symbols_with_data, start_date=start_date)

        for interval in asset_data_df_dict:
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                asset_data_df.set_index(['date'],inplace=True)
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval

                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='date', parse_dates=True)
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
                updated_data['interval'] = interval
                updated_data.to_csv(csv_file_path)
                data_frames.append(updated_data)
                pbar.update(1)
        
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False,inplace=True)
        
        # Set multi-index
        combined_df.set_index(['date','symbol'],inplace=True)
        cols = asset_data_df
        combined_df = combined_df.reset_index().pivot_table(values=cols, index=['interval','Datetime'], columns=['symbol'], aggfunc='mean')
        # combined_df = combined_df.unstack(level='symbol')

        # Sort the index
        combined_df.sort_index(inplace=True)       
        
        return combined_df