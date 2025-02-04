from copy import deepcopy
from tracemalloc import start
import pandas as pd
import warnings

from sqlalchemy import over
warnings.filterwarnings('ignore', category=FutureWarning, message="'T' is deprecated and will be removed in a future version")
# Suppress inplace operation warnings from pykrakenapi
warnings.filterwarnings('ignore', category=FutureWarning,
                       message=".*through chained assignment using an inplace method.*")
import logging
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import time
import krakenex
from pykrakenapi import KrakenAPI
import numpy as np
from systems.utils import create_logger, generate_hash_id, project_path, sleeper
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
import requests, pytz

class CustomKrakenAPI(KrakenAPI):
    """Custom KrakenAPI class to fix DataFrame warnings and timestamp handling"""
    def get_recent_trades(self, pair, since=None):
        if since is not None:
            # Ensure since is a proper integer
            try:
                since = int(since)
            except (TypeError, ValueError):
                since = int(float(since))
        
        trades, last = super().get_recent_trades(pair, since)
        
        if trades is not None and not trades.empty:
            # Create a new DataFrame instead of modifying in place
            trades = pd.DataFrame(trades)
            # First ensure the columns exist with default values
            if 'buy_sell' not in trades.columns:
                trades['buy_sell'] = 'unknown'
            if 'market_limit' not in trades.columns:
                trades['market_limit'] = 'unknown'
            
            # Then apply mapping only to non-null values
            trades['buy_sell'] = trades['buy_sell'].map({'b': 'buy', 's': 'sell'}).fillna('unknown')
            trades['market_limit'] = trades['market_limit'].map({'l': 'limit', 'm': 'market'}).fillna('unknown')
        return trades, last

    def get_ohlc_data(self, pair, interval, since=None):
        if since is not None:
            # Ensure since is a proper integer
            try:
                since = int(since)
            except (TypeError, ValueError):
                since = int(float(since))
        
        return super().get_ohlc_data(pair, interval, since)

class KrakenConnect:
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='Kraken-connect', print_to_console=True)
        self.api = None
        self.k = None
        self.connected = False
    
    def connect_to_kraken(self):
        try:
            load_dotenv()
            api_key = os.getenv('KRAKEN_API_KEY')
            api_secret = os.getenv('KRAKEN_API_SECRET')
            if not api_key or not api_secret:
                raise ValueError("Kraken API credentials not found in environment variables")
            self.k = krakenex.API(key=api_key, secret=api_secret)
            self.api = CustomKrakenAPI(self.k)  # Use CustomKrakenAPI instead
            self.api.retry = 5  # Set retry time gap to 2
            self.connected = True
            self.logger.info('Connected to Kraken')
            return self.api
        except Exception as e:
            self.logger.error(f'Error connecting to Kraken: {e}')
            self.connected = False
            return None

class Data:
    def __init__(self, api, connect_to_kraken):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='Kraken-data', print_to_console=True)
        self.api = api
        self.connect_to_kraken = connect_to_kraken
        self.interval_lookup = {
            "1m": 1,
            "5m": 5,
            '1h': 60,
            "1d": 1440,
            
        }

    def check_connection(self):
        if not self.api:
            self.logger.debug('Not connected to Kraken. Connecting now')
            self.api = self.connect_to_kraken()

    def fetch_historical_data(self, kraken_symbol, interval, start_date, end_date, kraken_interval):
        """
        Fetch historical data using OHLC for recent data and trades for older data
        
        Args:
            kraken_symbol: Trading pair symbol
            interval: Time interval (1m, 5m, 1d)
            start_date: Start date for data fetching
            end_date: End date for data fetching
            kraken_interval: Kraken API specific interval
        
        Returns:
            pd.DataFrame: OHLCV data or None if no data retrieved
        """
        max_retries = 8
        backoff_factor = 1.5
        now = pd.Timestamp.now(tz='UTC')
        
        # Define time limits for different intervals
        interval_limits = {
            "1m": pd.Timedelta(hours=1),
            "5m": pd.Timedelta(days=1),
            "1d": pd.Timedelta(days=30),
        }
        # Ensure all datetime objects are timezone aware
        def ensure_tz_aware(dt):
            if dt is None:
                return None
            if not isinstance(dt, pd.Timestamp):
                dt = pd.Timestamp(dt)
            if dt.tz is None:
                dt = pd.Timestamp(dt).tz_localize(pytz.UTC)
            return dt
        
        if end_date is None:
            end_date = now
        
        # Ensure dates are timezone-aware
        start_date = pd.Timestamp(start_date).tz_localize('UTC') if not isinstance(start_date, pd.Timestamp) else start_date
        end_date = pd.Timestamp(end_date).tz_localize('UTC') if not isinstance(end_date, pd.Timestamp) else end_date
        if start_date.tz is None:
            start_date = start_date.tz_localize('UTC')
        if end_date.tz is None:
            end_date = end_date.tz_localize('UTC')
        
        data_chunks = []
        remaining_start = start_date
        remaining_end = end_date
        # self.logger.info({'remaining_start': remaining_start, 'remaining_end': remaining_end})
        ## Calculate recent data window with explicit timezone
        recent_data_start = ensure_tz_aware(now - interval_limits.get(interval, pd.Timedelta(days=1)))
        recent_data_end = ensure_tz_aware(now)
        
        # Check for overlap with recent data window
        has_recent_overlap = (
            (remaining_start <= recent_data_end and remaining_end >= recent_data_start) or
            (remaining_end >= recent_data_start and remaining_start <= recent_data_end)
        )
        # print the time zones of the start and end dates, recent data start and end dates, remaining start and end dates
        # Fetch recent data using OHLC endpoint if there's overlap
        if has_recent_overlap:
            overlap_start = ensure_tz_aware(max(remaining_start, recent_data_start))
            overlap_end = ensure_tz_aware(min(remaining_end, recent_data_end))
            
            # self.logger.info(f"Fetching OHLC data for {kraken_symbol} from {overlap_start} to {overlap_end}")
            ohlc_data = self._fetch_ohlc_data(kraken_symbol, interval, overlap_start, kraken_interval)
            ohlc_data_last_timestamp = ohlc_data.index[-1] if ohlc_data is not None and not ohlc_data.empty else None
            # self.logger.info({'ohlc_data_last_timestamp':ohlc_data_last_timestamp})
            if ohlc_data is not None and not ohlc_data.empty:
                ohlc_data = ohlc_data[overlap_start:overlap_end]
                if not ohlc_data.empty:
                    data_chunks.append(ohlc_data)
                    # Update the remaining period to fetch
                    if ohlc_data_last_timestamp:
                        remaining_end = overlap_start
        
        # Fetch remaining data using trades endpoint if needed
        # self.logger.info({'remaining_start': remaining_start, 'remaining_end': remaining_end})
        if remaining_start < remaining_end:
            retry_count = 0
            current_start = remaining_start
            all_trades = []
            if has_recent_overlap:
                sleeper(1)
            # First collect all trades
            self.logger.info({'remaining_start': remaining_start, 'remaining_end': remaining_end})
            while retry_count < max_retries and current_start < remaining_end:
                # try:
                    self.logger.info(f"Fetching trades for {kraken_symbol} from {current_start}")
                    granularity_int = 60
                    trades, last = self.api.get_recent_trades(kraken_symbol, since=int(current_start.timestamp()) + granularity_int)
                    if trades is None or trades.empty:
                        self.logger.warning('No trades returned')
                        break
                    
                    # Process trades timestamps
                    trades.index = pd.to_datetime(trades.index).tz_localize('UTC') if trades.index.tz is None else trades.index
                    all_trades.append(trades)
                    
                    # Move forward using the last timestamp from trades
                    current_start = trades.index[-1]
                    retry_count = 0  # Reset on success
                    
                    if current_start >= remaining_end:
                        break
                    else:
                        sleeper(1)
                        
                # except Exception as e:
                #     retry_count += 1
                #     wait_time = int(5 * (backoff_factor ** retry_count)) if 'public call frequency exceeded' in str(e).lower() else int(backoff_factor ** retry_count)
                #     self.logger.warning(f"Error fetching trades: {str(e)}. Waiting {wait_time}s. Retry {retry_count}/{max_retries}")
                #     sleeper(wait_time)
                #     if retry_count >= max_retries:
                #         self.logger.error(f"Max retries ({max_retries}) exceeded for {kraken_symbol}")
                #         break
            
            # Convert all collected trades to OHLCV
            if all_trades:
                trades_df = pd.concat(all_trades)
                trades_df = trades_df.sort_index()
                trades_df = trades_df[~trades_df.index.duplicated(keep='last')]
                
                # Convert trades to OHLCV format
                freq_map = {"1m": "1Min", "5m": "5Min", "1d": "1D"}
                ohlc = trades_df.groupby(pd.Grouper(freq=freq_map.get(interval, "1Min"))).agg({
                    'price': ['first', 'max', 'min', 'last'],
                    'volume': 'sum'
                }).dropna()
                
                ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
                ohlc['count'] = trades_df.groupby(pd.Grouper(freq=freq_map.get(interval, "1Min"))).size()
                
                if not ohlc.empty:
                    data_chunks.append(ohlc)
        
        # Combine and clean data
        if data_chunks:
            combined_data = pd.concat(data_chunks)
            combined_data = combined_data.sort_index()
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            return combined_data
        
        self.logger.error(f"No data retrieved for {kraken_symbol}")
        return None
    
    def _fetch_ohlc_data(self, kraken_symbol, interval, current_start, kraken_interval):
        """Helper method to fetch data using OHLC endpoint for recent data"""
        try:
            data, last = self.api.get_ohlc_data(
                kraken_symbol,
                interval=kraken_interval,
                since=int(current_start.timestamp()),
            )
            # sort data by dtime in ascending order
            data = data.sort_values(by='dtime', ascending=True)
            if data is None:
                self.logger.warning(f"No OHLC data returned for {kraken_symbol}")
                return None
            if data is not None and not data.empty:
                # self.logger.info(f"Retrieved {len(data)} rows using OHLC endpoint")
                # self.logger.info(f"Time range: {data.index.min()} to {data.index.max()}")
                if data.index.tz is None:
                    data.index = pd.to_datetime(data.index).tz_localize('UTC')
            return data
        except Exception as e:
            self.logger.error(f"Error fetching OHLC data: {e}")
            return None

    def restructure_asset_data_df(self, asset_data_df, symbol=None, interval=None):
        """Restructure the data to match the expected format"""
        if asset_data_df is not None and not asset_data_df.empty:
            # Convert index to proper datetime format and preserve values
            asset_data_df = asset_data_df.reset_index(drop=False)
            asset_data_df = asset_data_df.rename(columns={'dtime': 'datetime'})
            # Convert to datetime first, then ensure UTC timezone
            asset_data_df['datetime'] = pd.to_datetime(asset_data_df['datetime'], unit='s')
            asset_data_df['datetime'] = asset_data_df['datetime'].dt.tz_localize('UTC')
            
            # Create a copy to avoid SettingWithCopyWarning
            asset_data_df = asset_data_df.copy()
            
            # Add required columns with default values
            asset_data_df['adj close'] = asset_data_df['close']
            asset_data_df['average'] = (asset_data_df['high'] + asset_data_df['low']) / 2
            asset_data_df['barcount'] = asset_data_df['count'] if 'count' in asset_data_df.columns else 0
            asset_data_df['symbol'] = symbol
            asset_data_df['interval'] = interval
            
            # Select and order columns as required
            columns = ['datetime', 'adj close', 'close', 'high', 'low', 'open', 'volume', 
                      'average', 'barcount', 'symbol', 'interval']
            asset_data_df = asset_data_df[columns]
            
            # Convert numeric columns to float
            numeric_cols = ['adj close', 'close', 'high', 'low', 'open', 'volume', 'average', 'barcount']
            asset_data_df[numeric_cols] = asset_data_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        else:
            # Create empty DataFrame with all required columns
            asset_data_df = pd.DataFrame(columns=[
                'datetime', 'adj close', 'close', 'high', 'low', 'open', 'volume', 
                'average', 'barcount', 'symbol', 'interval'
            ])
        return asset_data_df

    def fetch_price_data(self, stock_symbols, interval_inputs, data_folder=project_path+'db/data/kraken',
                     throttle_secs=1, run_mode=3, start_date=None, end_date=None, lookback=None,
                     update_data=True):
        data_frames = []
        interval_dict = {'1d':pd.Timedelta(days=1), '1h':pd.Timedelta(hours=1), '1m':pd.Timedelta(minutes=1)}
        # Define chunking sizes for backward updates (in days)
        chunking_dict = {
            "1m": 1,      # 24 hours
            "1h": 5,      # 5 days
            "4h": 5,      # 5 days
            "12h": 5,     # 5 days
            "1d": 180     # 180 days
        }
        
        # Track remaining dates for each symbol and interval
        remaining_dates = {}
        
        for interval in interval_inputs:
            remaining_dates[interval] = {}
            kraken_interval = self.interval_lookup[interval]
            for symbol in tqdm(stock_symbols, desc=f'Updating {interval} data'):
                # try:
                    # Convert symbol to Kraken format
                    kraken_symbol = f"{symbol}"
                    self.logger.info(f"Processing {kraken_symbol} at interval {interval}")
                    
                    # Setup paths
                    csv_path = os.path.join(data_folder, interval, f"{symbol}.csv")
                    os.makedirs(os.path.join(data_folder, interval), exist_ok=True)
                    
                    # Load existing data
                    symbol_data = pd.DataFrame()
                    if os.path.exists(csv_path):
                        symbol_data = pd.read_csv(csv_path, parse_dates=['datetime'])
                        symbol_data['symbol'] += 'USD'
                        if not symbol_data.empty:
                            symbol_data.set_index('datetime', inplace=True)
                            if not symbol_data.index.tz:
                                symbol_data.index = pd.to_datetime(symbol_data.index).tz_localize('UTC')
                                
                            # now lets trim the loaded data to only be between the start and end dates
                            if start_date:
                                symbol_data = symbol_data[symbol_data.index >= start_date]
                            if end_date:
                                symbol_data = symbol_data[symbol_data.index <= end_date]
                    if update_data == False and run_mode not in [1,2]:
                        if not symbol_data.empty:
                            data_frames.append(symbol_data)
                        continue
                    else:
                        # Get timestamp boundaries
                        current_timestamp = pd.Timestamp.now(tz='UTC')
                        earliest_timestamp = symbol_data.index.min() if not symbol_data.empty else None
                        latest_timestamp = symbol_data.index.max() if not symbol_data.empty else None
                        forward_update = True if current_timestamp - latest_timestamp > interval_dict[interval] else False
                        
                        # Forward update
                        if forward_update and latest_timestamp is not None:
                            self.logger.info(f"Forward updating {kraken_symbol} from {latest_timestamp}")
                            forward_data = self.fetch_historical_data(
                                kraken_symbol=kraken_symbol,
                                interval=interval,
                                start_date=latest_timestamp,
                                end_date=None,
                                kraken_interval=kraken_interval
                            )
                            
                            if forward_data is not None and not forward_data.empty:
                                symbol_data = pd.concat([symbol_data, forward_data])
                                symbol_data = symbol_data[~symbol_data.index.duplicated(keep='last')]
                                # update remaining_date_dict
                                
                        # Backward update
                        backward_update = False if earliest_timestamp < start_date else True
                        
                        if backward_update:
                            # Initialize remaining dates if not already set
                            if symbol not in remaining_dates[interval]:
                                if start_date and end_date:
                                    remaining_dates[interval][symbol] = {
                                        'start': pd.Timestamp(start_date).tz_localize('UTC'),
                                        'end': pd.Timestamp(end_date).tz_localize('UTC')
                                    }
                                elif earliest_timestamp is not None:
                                    remaining_dates[interval][symbol] = {
                                        'start': earliest_timestamp - pd.Timedelta(days=chunking_dict[interval]),
                                        'end': earliest_timestamp
                                    }
                            
                            # Process one chunk
                            if symbol in remaining_dates[interval]:
                                chunk_end = remaining_dates[interval][symbol]['end']
                                chunk_start = max(
                                    remaining_dates[interval][symbol]['start'],
                                    chunk_end - pd.Timedelta(days=chunking_dict[interval])
                                )
                                
                                self.logger.info(f"Backward updating {kraken_symbol} chunk {chunk_start} to {chunk_end}")
                                backward_data = self.fetch_historical_data(
                                    kraken_symbol=kraken_symbol,
                                    interval=interval,
                                    start_date=chunk_start,
                                    end_date=chunk_end,
                                    kraken_interval=kraken_interval
                                )
                                
                                if backward_data is not None and not backward_data.empty:
                                    symbol_data = pd.concat([backward_data, symbol_data])
                                    symbol_data = symbol_data[~symbol_data.index.duplicated(keep='last')]
                                    
                                    # Update remaining dates
                                    remaining_dates[interval][symbol]['end'] = chunk_start
                                    if chunk_start <= remaining_dates[interval][symbol]['start']:
                                        del remaining_dates[interval][symbol]
                        
                        # Clean up and save data
                        if not symbol_data.empty:
                            symbol_data = symbol_data.sort_index()
                            symbol_data['symbol'] = symbol
                            symbol_data['interval'] = interval
                            data_frames.append(symbol_data)
                            symbol_data.to_csv(csv_path)

        # Combine all data frames
        if data_frames:
            data_frames = [df for df in data_frames if not df.empty]
            if data_frames:
                combined_df = pd.concat(data_frames)
                combined_df.reset_index(inplace=True)
                combined_df.rename(columns={'index': 'datetime'}, inplace=True)
                # Create multi-index DataFrame
                pass_cols = ['open', 'high', 'low', 'close', 'volume']
                combined_df = combined_df.pivot_table(
                    values=pass_cols,
                    index=['interval', 'datetime'],
                    columns=['symbol'],
                    aggfunc='mean'
                )
                
                combined_df = combined_df.sort_index()
                combined_df['data_source'] = 'kraken'
                self.logger.info({'combined_df':combined_df.shape})
                self.logger.info({'combined_df':combined_df.head()})
                
                return combined_df
        
        return pd.DataFrame()
    
class Kraken(KrakenConnect):
    def __init__(self):
        super().__init__()
        self.data = Data(self.api, self.connect_to_kraken)
        self.execute = Kraken_Execute(self.api, self.k, self.connect_to_kraken)

class Kraken_Execute:
    def __init__(self, api, k, connect_to_kraken):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='Kraken-execute', print_to_console=True)
        self.api = api
        self.k = k  # Direct krakenex API for some operations
        self.connect_to_kraken = connect_to_kraken

    def check_connection(self):
        if not self.api:
            self.logger.debug('Not connected to Kraken. Connecting now')
            self.api = self.connect_to_kraken()

    def place_order(self, order, market_data_df):
        self.check_connection()
        
        # Convert traditional symbol to Kraken pair format
        kraken_symbol = f"{order.symbol}"
        quantity = order.orderQuantity
        
        try:
            order_params = {
                'pair': kraken_symbol,
                'volume': str(quantity),
                'type': 'buy' if order.orderDirection == 'BUY' else 'sell',
            }
            
            if order.order_type == 'MARKET':
                order_params['ordertype'] = 'market'
                response = self.k.query_private('AddOrder', order_params)
                
                if response.get('error'):
                    raise Exception(response['error'][0])
                
                response_order = deepcopy(order)
                response_order.status = 'submitted'
                response_order.broker_order_id = response['result']['txid'][0]
                
            elif order.order_type == 'STOPLOSS':
                order_params['ordertype'] = 'stop-loss'
                order_params['price'] = str(order.price)
                response = self.k.query_private('AddOrder', order_params)
                
                if response.get('error'):
                    raise Exception(response['error'][0])
                
                response_order = deepcopy(order)
                response_order.status = 'open'
                response_order.broker_order_id = response['result']['txid'][0]
                
            else:
                response_order = deepcopy(order)
                response_order.status = 'rejected'
                setattr(response_order, 'message', f'Order type {order.order_type} not supported')
                
        except Exception as e:
            response_order = deepcopy(order)
            response_order.status = 'rejected'
            setattr(response_order, 'message', f'Order failed: {str(e)}')
            
        return response_order

    def update_order_status(self, order):
        self.check_connection()
        
        try:
            response = self.k.query_private('QueryOrders', {'txid': order.broker_order_id})
            
            if response.get('error'):
                raise Exception(response['error'][0])
            
            order_info = response['result'][order.broker_order_id]
            response_order = deepcopy(order)
            
            status_mapping = {
                'closed': 'closed',
                'canceled': 'cancelled',
                'expired': 'cancelled',
                'open': 'open',
                'pending': 'submitted'
            }
            
            response_order.status = status_mapping.get(order_info['status'], 'rejected')
            
            if order_info['status'] == 'closed':
                response_order.filled_price = float(order_info['price'])
                response_order.filled_timestamp = datetime.fromtimestamp(float(order_info['closetm']), tz=timezone.utc)
            
            setattr(response_order, 'fresh_update', True)
            
        except Exception as e:
            self.logger.error(f"Error updating order status: {e}")
            response_order = order
            
        return response_order

    def execute_order(self, order, market_data_df, system_timestamp):
        if order.status == 'pending':
            response_order = self.place_order(order, market_data_df)
        elif order.status.lower() in ['open', 'submitted']:
            response_order = self.update_order_status(order)
        else:
            response_order = order
            
        return response_order

    def get_account_summary(self, trading_currency='ZUSD', base_currency='USD'):
        self.check_connection()
        
        try:
            # Get account balance
            balance = self.k.query_private('Balance')
            if balance.get('error'):
                raise Exception(balance['error'][0])
                
            # Get trade balance for additional info
            trade_balance = self.k.query_private('TradeBalance', {'asset': trading_currency})
            if trade_balance.get('error'):
                raise Exception(trade_balance['error'][0])
            
            # Extract relevant values
            available = float(balance['result'].get(trading_currency, 0))
            total = float(trade_balance['result']['eb'])  # Equivalent balance
            used = total - available
            
            return {
                'USD': {
                    'total_account_value': total,
                    'buying_power_available': available,
                    'buying_power_used': used,
                    'total_buying_power': total,
                    'cushion': available/total if total > 0 else 0,
                    'margin_multiplier': float(trade_balance['result']['m']),
                    'pct_of_margin_used': used/total if total > 0 else 0
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {'USD': {}}