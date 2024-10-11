
#from ib_insync import IB, Stock, util, MarketOrder, LimitOrder, StopOrder
# from ib_insync import *
from copy import deepcopy
from urllib import response
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Order, util
from matplotlib.pyplot import bar
import nest_asyncio
import os
#from networkx import dfs_edges
import pandas as pd
from tqdm import tqdm
# from datetime import datetime, timedelta
import asyncio
import logging
from systems.utils import create_logger, generate_hash_id, sleeper

nest_asyncio.apply()

class IBKRConnect:
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='IBKR-connect', print_to_console=True)
        self.ib = None
        self.connected = False
    
    def connect_to_IBKR(self, client_id=None):
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
                client_id += 1  # Increment clientId to avoid conflict
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
        self.duration_lookup = {"1m": "1 W", "2m": "2 W", "5m":"1 M","1d":"20 Y"} #update this to get max duration fora each interval

    def check_ib_connection(self):
        if not self.ib:
            self.logger.debug('Not connected to IBKR. Connecting now')
            self.ib = self.connect_to_IBKR()
            if self.ib:
                self.logger.debug('Connected to IBKR')
    
    async def update_price_data_batch(self, stock_symbols, start_date=None, batch_size = 75):
        self.check_ib_connection()
        
        asset_data_df_dict = {}
        for interval in stock_symbols:
            barSize = self.interval_lookup[interval]
            duration = self.duration_lookup[interval]
            asset_data_df_dict[interval] = {}
            for i in range(0, len(stock_symbols[interval]), batch_size):
                batch = stock_symbols[interval][i:i + batch_size]
                task = [self.fetch_historical_data(ticker, barSize, duration, start=start_date) for ticker in batch]
                data = await asyncio.gather(*task, return_exceptions=True)
                for i, bars in enumerate(data):
                    if isinstance(bars, Exception):
                        raise Exception(f"Error fetching data for batch: {bars}. Are you sure TWS is Running?")

                    logging.debug(f'bars: {bars}, type: {type(bars)}')
                    asset_data_df_dict[interval][batch[i]] = util.df(bars)
                        
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

    async def fetch_historical_data(self, ticker, barSize, duration, start=None, exchange:str = "SMART", currency:str = "USD"):
        contract = self.ib.qualifyContracts(Stock(ticker, exchange, currency))[0]
        self.logger.debug({'barSize': barSize, 'duration': duration})
        # duration = self.duration_lookup.get(barSize,"1 Y")
        # barSize = self.interval_lookup.get(barSize,"1 day")
        # duration = "20 Y"
        # barSize = "1 day"
        # Fetch historical data for max duration
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=barSize,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        return bars

    def update_price_data(self, stock_symbols,interval_inputs=['1d'], data_folder='db/data/ibkr', throttle_secs=1,back_test_start_date=None,back_test_end_date=None, lookback=None):
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
        loop = asyncio.get_event_loop()
        asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(stock_symbols_no_data, start_date=None))
        
        for interval in asset_data_df_dict:
            data_input_folder = os.path.join(data_folder,interval)
            if not os.path.exists(data_input_folder):
                os.makedirs(data_input_folder)
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                csv_file_path = os.path.join(data_input_folder, f"{symbol}.csv")

                asset_data_df['date'] = pd.to_datetime(asset_data_df['date'],utc=True)
                asset_data_df.set_index(['date'],inplace=True)
                asset_data_df.index = asset_data_df.index.tz_convert('UTC')
                asset_data_df.index.names = ['datetime']
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval
                asset_data_df = asset_data_df.dropna(how='all')
                asset_data_df.to_csv(csv_file_path)
                data_frames.append(asset_data_df)
                pbar.update(1)
        
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        start_date = None
        for interval in stock_symbols_with_data:
            for symbol in stock_symbols_with_data[interval]:
                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                last_date = existing_data.index.max()
                start_date = last_date if start_date is None else min(start_date, last_date)
                
        self.logger.debug({'start_date': start_date})
        
        loop = asyncio.get_event_loop()
        asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(stock_symbols_with_data, start_date=start_date))
        
        for interval in asset_data_df_dict:
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                asset_data_df['date'] = pd.to_datetime(asset_data_df['date'],utc=True)
                asset_data_df.set_index(['date'],inplace=True)
                asset_data_df.index = asset_data_df.index.tz_convert('UTC')
                asset_data_df.index.names = ['datetime']
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval

                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                # get the start date of asset_data_df
                symbol_start_date = asset_data_df.index.min()
                # self.logger.debug({'symbol_start_date': symbol_start_date})
                # prune the existing_data to only include data before the start date
                # self.logger.debug(type(existing_data.index[0]))
                # self.logger.debug(type(symbol_start_date))
                symbol_start_date = symbol_start_date.to_pydatetime()
                existing_data = existing_data[existing_data.index < symbol_start_date]
                # self.logger.debug({'asset_data_df-shape': asset_data_df.shape})
                # self.logger.debug({'existing_data-shape': existing_data.shape})
                # concatenate the existing data and the new data
                updated_data = pd.concat([existing_data, asset_data_df])
                updated_data['symbol'] = symbol
                updated_data['interval'] = interval
                updated_data = updated_data.dropna(how='all')
                updated_data.to_csv(csv_file_path)
                data_frames.append(updated_data)
                pbar.update(1)
        
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False,inplace=True)
        
        # Set multi-index
        combined_df.set_index(['datetime','symbol'],inplace=True)
        cols = asset_data_df
        combined_df = combined_df.reset_index().pivot_table(values=cols, index=['interval','datetime'], columns=['symbol'], aggfunc='mean')
        # combined_df = combined_df.unstack(level='symbol')

        # Sort the index
        combined_df.sort_index(inplace=True)  
        if back_test_start_date is None and back_test_end_date is None:
            return combined_df

        if back_test_start_date is not None and back_test_end_date is not None:
            combined_df = combined_df.loc[(combined_df.index.get_level_values(1) >= back_test_start_date) & (combined_df.index.get_level_values(1) <= back_test_end_date),:]
        
        if back_test_start_date is not None:
            combined_df = combined_df.loc[combined_df.index.get_level_values(1) >= back_test_start_date,:]
    
        if back_test_end_date is not None:
            combined_df = combined_df.loc[combined_df.index.get_level_values(1) <= back_test_end_date,:]      
        
        return combined_df
    
    def update_price_data_old(self, stock_symbols,interval_inputs=['1d'], data_folder='db/data/ibkr', throttle_secs=1,back_test_start_date=None,back_test_end_date=None, lookback=None):
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
        loop = asyncio.get_event_loop()
        asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(stock_symbols_no_data, start_date=None))
        
        for interval in asset_data_df_dict:
            data_input_folder = os.path.join(data_folder,interval)
            if not os.path.exists(data_input_folder):
                os.makedirs(data_input_folder)
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                csv_file_path = os.path.join(data_input_folder, f"{symbol}.csv")

                asset_data_df['date'] = pd.to_datetime(asset_data_df['date'],utc=True)
                asset_data_df.set_index(['date'],inplace=True)
                asset_data_df.index = asset_data_df.index.tz_convert('UTC')
                asset_data_df.index.names = ['datetime']
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval
                asset_data_df = asset_data_df.dropna(how='all')
                asset_data_df.to_csv(csv_file_path)
                data_frames.append(asset_data_df)
                pbar.update(1)
        
        # Update the existing data. Get the minimum start date for the ones that have data. Then update the new downloaded data to the existing data
        start_date = None
        for interval in stock_symbols_with_data:
            for symbol in stock_symbols_with_data[interval]:
                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                last_date = existing_data.index.max()
                start_date = last_date if start_date is None else min(start_date, last_date)
                
        self.logger.debug({'start_date': start_date})
        
        loop = asyncio.get_event_loop()
        asset_data_df_dict = loop.run_until_complete(self.update_price_data_batch(stock_symbols_with_data, start_date=start_date))
        
        for interval in asset_data_df_dict:
            for symbol in asset_data_df_dict[interval]:
                asset_data_df = asset_data_df_dict[interval][symbol]
                asset_data_df['date'] = pd.to_datetime(asset_data_df['date'],utc=True)
                asset_data_df.set_index(['date'],inplace=True)
                asset_data_df.index = asset_data_df.index.tz_convert('UTC')
                asset_data_df.index.names = ['datetime']
                cols = list(asset_data_df.columns)
                asset_data_df['symbol'] = symbol
                asset_data_df['interval'] = interval

                csv_file_path = os.path.join(data_folder,interval, f"{symbol}.csv")
                existing_data = pd.read_csv(csv_file_path, index_col='datetime', parse_dates=True)
                # get the start date of asset_data_df
                symbol_start_date = asset_data_df.index.min()
                # self.logger.debug({'symbol_start_date': symbol_start_date})
                # prune the existing_data to only include data before the start date
                # self.logger.debug(type(existing_data.index[0]))
                # self.logger.debug(type(symbol_start_date))
                symbol_start_date = symbol_start_date.to_pydatetime()
                existing_data = existing_data[existing_data.index < symbol_start_date]
                # self.logger.debug({'asset_data_df-shape': asset_data_df.shape})
                # self.logger.debug({'existing_data-shape': existing_data.shape})
                # concatenate the existing data and the new data
                updated_data = pd.concat([existing_data, asset_data_df])
                updated_data['symbol'] = symbol
                updated_data['interval'] = interval
                updated_data = updated_data.dropna(how='all')
                updated_data.to_csv(csv_file_path)
                data_frames.append(updated_data)
                pbar.update(1)
        
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(data_frames)
        combined_df.reset_index(drop=False,inplace=True)
        
        # Set multi-index
        combined_df.set_index(['datetime','symbol'],inplace=True)
        cols = asset_data_df
        combined_df = combined_df.reset_index().pivot_table(values=cols, index=['interval','datetime'], columns=['symbol'], aggfunc='mean')
        # combined_df = combined_df.unstack(level='symbol')

        # Sort the index
        combined_df.sort_index(inplace=True)  
        if back_test_start_date is None and back_test_end_date is None:
            return combined_df

        if back_test_start_date is not None and back_test_end_date is not None:
            combined_df = combined_df.loc[(combined_df.index.get_level_values(1) >= back_test_start_date) & (combined_df.index.get_level_values(1) <= back_test_end_date),:]
        
        if back_test_start_date is not None:
            combined_df = combined_df.loc[combined_df.index.get_level_values(1) >= back_test_start_date,:]
    
        if back_test_end_date is not None:
            combined_df = combined_df.loc[combined_df.index.get_level_values(1) <= back_test_end_date,:]      
        
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
        order = []
        
        return order
    
    def get_open_orders(self):
        self.check_ib_connection()
        open_orders = []
        open_orders = self.ib.reqOpenOrders()
        trades = self.ib.trades()
        
        

        # for count, open_order in enumerate(open_orders):
            # print(f"Order ID: {open_order.order.orderId}, Symbol: {open_order.contract.symbol}, Status: {open_order.orderStatus.status}, Action: {open_order.order.action}, Quantity: {open_order.order.totalQuantity}, Order Type: {open_order.order.orderType}, Limit Price: {open_order.order.lmtPrice}, Aux Price: {open_order.order.auxPrice}")
    
    def place_order(self, order, market_data_df):
        # Create a contract for the stock
        # ticker: str, exchange: str, currency: str, orderSide: str, orderQuantity: int, orderType: str, limit_price: float = 0, stop_price: float = 0
        system_timestamp = market_data_df.index.get_level_values(1)[-1]
        symbol = order['symbol']
        currency = 'USD'
        exchange = 'SMART'
        orderDirection = order['orderDirection']
        orderQuantity = order['orderQuantity']
        
        self.check_ib_connection()
        response_order = None
        # Create an order for the stock
        if order['orderType'].lower() == 'market':
            contract = self.ib.qualifyContracts(Stock(symbol, exchange, currency))[0]
            IB_order = MarketOrder(orderDirection, orderQuantity)
            if 'order_id' not in order or order['order_id'] is None:
                mathematricks_order_id = generate_hash_id(order, system_timestamp)
            else:
                mathematricks_order_id = order['order_id']
                
            # IB_order.OrderRef = mathematricks_order_id
            # Place the order
            # self.logger.debug({'symbol':symbol, 'IB_order':IB_order})
            IB_order_response = self.ib.placeOrder(contract, IB_order)
            self.ib.sleep(1)
            # self.logger.debug({'IB_order_response':IB_order_response})
            response_order = deepcopy(order)
            response_order['order_id'] = mathematricks_order_id
            response_order['broker_order_id'] = IB_order_response.order.permId
            
            ibkr_order_status = IB_order_response.orderStatus.status if hasattr(IB_order_response.orderStatus, 'status') else None
            
            if ibkr_order_status in ['PendingSubmit']:
                status = 'submitted'
            elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
                status = 'open'
            elif ibkr_order_status == 'Filled':
                status = 'closed'
            elif ibkr_order_status == 'Cancelled':
                status = 'cancelled'
            else:
                status = 'pending'
            
            response_order['status'] = status
            response_order['filled'] = IB_order_response.orderStatus.filled
            response_order['remaining'] = IB_order_response.orderStatus.remaining
            response_order['avgFillPrice'] = IB_order_response.orderStatus.avgFillPrice
            response_order['tradeLogEntryTime'] = IB_order_response.log[0].time if IB_order_response.log else None
            response_order['errorCode'] = IB_order_response.log[0].errorCode if IB_order_response.log else None
            response_order['fresh_update'] = True
            response_order['message'] = 'Market order placed on IBKR'
        elif order['orderType'].lower() in ['stoploss_abs', 'stoploss_pct']:
            exitPrice = order['exitPrice']
            # contract = self.ib.qualifyContracts(Stock(symbol, exchange, currency))[0]
            # IB_order = StopOrder(orderDirection, orderQuantity, exitPrice)
            
            # Place a stoploss order on IBKR
            contract = Stock(symbol, exchange, currency)  # Example: Apple stock
            minTick = self.ib.reqContractDetails(contract)[0].minTick
            decimalPlaces = len(str(minTick).split('.')[1])
            IB_order = Order(action=orderDirection, orderType='STP', totalQuantity=orderQuantity, auxPrice=round(exitPrice, decimalPlaces))
            # self.logger.debug({'symbol':symbol, 'IB_order':IB_order})
            # Place the order
            IB_order_response = self.ib.placeOrder(contract, IB_order)
            self.ib.sleep(1)
            # self.logger.debug({'IB_order_response':IB_order_response})
            
            # self.logger.debug({'IB_order_response':IB_order_response})
            response_order = deepcopy(order)
            if 'order_id' not in order or order['order_id'] is None:
                mathematricks_order_id = generate_hash_id(order, system_timestamp)
            else:
                mathematricks_order_id = order['order_id']
            response_order['order_id'] = mathematricks_order_id
            response_order['broker_order_id'] = IB_order_response.order.orderId if hasattr(IB_order_response.order, 'orderId') else None
            
            ibkr_order_status = IB_order_response.orderStatus.status if hasattr(IB_order_response.orderStatus, 'status') else None
            
            if ibkr_order_status in ['PendingSubmit']:
                status = 'pending'
            elif ibkr_order_status in ['Submitted', 'PreSubmitted']:
                status = 'open'
            elif ibkr_order_status == 'Filled':
                status = 'closed'
            elif ibkr_order_status == 'Cancelled':
                status = 'cancelled'
            else:
                status = 'pending'
            
            response_order['status'] = IB_order_response.orderStatus.status if hasattr(IB_order_response.orderStatus, 'status') else None
            response_order['filled'] = IB_order_response.orderStatus.filled if hasattr(IB_order_response.orderStatus, 'filled') else 0
            response_order['remaining'] = IB_order_response.orderStatus.remaining if hasattr(IB_order_response.orderStatus, 'remaining') else orderQuantity
            response_order['avgFillPrice'] = IB_order_response.orderStatus.avgFillPrice if hasattr(IB_order_response.orderStatus, 'avgFillPrice') else None
            response_order['tradeLogEntryTime'] = IB_order_response.log[-1].time if hasattr(IB_order_response.log[-1], 'time') and IB_order_response.log else None
            response_order['errorCode'] = IB_order_response.log[-1].errorCode if hasattr(IB_order_response.log[-1], 'errorCode') and IB_order_response.log else None
            response_order['fresh_update'] = True
            response_order['message'] = 'Stoploss order placed on IBKR'
        else:
            raise Exception(f"Order type {order['orderType']} not supported. Use 'market', 'stoploss_abs', 'stoploss_pct")
        # Return the response_order
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
    
    def update_order_status(self, order, market_data_df):
        self.logger.warning('NOTE: NEED TO CHECK IF UPDATE ORDER IBKR IS WORKING CORRECTLY')
        '''if the order is open, check if it's filled'''
        system_timestamp = market_data_df.index.get_level_values(1)[-1]
        symbol = order['symbol']
        granularity = order['granularity']
        current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'][-1]

        # Check if the order is open
        if order['status'] in ['open', 'submitted']:
            if self.ibkr_open_orders['updated_time'] is None or system_timestamp != self.ibkr_open_orders['updated_time']:
                self.ibkr_open_orders['open_orders'] = self.ib.reqOpenOrders()
                self.ibkr_open_orders['updated_time'] = system_timestamp
            
            current_status = order['status']
            # Check if the order is filled
            # if order['orderType'].lower() == 'market':
            for open_order in self.ibkr_open_orders['open_orders']:
                if open_order.order.permId == order['broker_order_id']:
                    new_status = self.ibkr_order_status_to_mathematricks_order_status(open_order.orderStatus.status)
                    if new_status != current_status:
                        response_order = deepcopy(order)
                        response_order['status'] = new_status
                        response_order['filled'] = open_order.orderStatus.filled
                        response_order['remaining'] = open_order.orderStatus.remaining
                        response_order['avgFillPrice'] = open_order.orderStatus.avgFillPrice
                        response_order['tradeLogEntryTime'] = open_order.log[0].time
                        response_order['errorCode'] = open_order.log[0].errorCode
                        response_order['message'] = self.order_status_change_message(self, current_status, new_status)
                        response_order['fresh_update'] = True
                    else:
                        response_order = order
                    break
                
        return response_order
    
    def modify_stoploss_price(self, order, system_timestamp):
        # Cancel the existing order
        contract = Stock(order['symbol'], 'SMART', 'USD')
        target_order_id = order['broker_order_id']
        open_orders = self.ib.reqOpenOrders()
        for order in open_orders:
            if order.orderId == target_order_id:
                new_stop_price = order['exitPrice']  # Set your new stop-loss price here
                order.auxPrice = new_stop_price  # Update the stop-loss price
                
                # Place the updated order
                self.ib.placeOrder(contract, order)
        order['status'] = 'open'
        order['modified_timestamp'] = system_timestamp

        return order
    
    def modify_order(self, order, system_timestamp):
        '''Modify an existing order with new parameters'''
        if order['modify_reason'] == 'stoploss_update':
            response_order = self.modify_stoploss_price(order, system_timestamp)
        else:
            raise NotImplementedError(f"Modify order reason {order['modify_reason']} not implemented.")
        return response_order
    
    def execute_order(self, order, market_data_df, system_timestamp):
        response_order = order
        if order['status'] == 'pending':
            # Execute the order in the simulator
            response_order = self.place_order(order, market_data_df=market_data_df)
        elif order['status'] in ['open', 'submitted']:
            # Update the order status in the simulator and check if it's filled
            response_order = self.update_order_status(order, market_data_df=market_data_df)
        elif order['status'] == 'modify':
            response_order = self.modify_order(order, system_timestamp=system_timestamp)
        
        return response_order

if __name__ == '__main__':
    ibkr = IBKR()
    if not ibkr.connected:
        print('Not connected to IBKR')
        ibkr.connect_to_IBKR()
        if ibkr.connected:
            print('Connection established.')
            acc_summary = ibkr.ib.accountSummary()
            print({'acc_summary':acc_summary})
    
    ticker = 'AAPL'
    exchange = 'SMART'
    currency = 'USD'
    contract = ibkr.ib.qualifyContracts(Stock(ticker, exchange, currency))[0]
    barSize = '1d'
    # duration = ibkr.duration_lookup.get(barSize,"1 Y")
    # barSize = ibkr.interval_lookup.get(barSize,"1 day")
    duration = "1 Y"
    barSize = "1 day"
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
    print(bars)