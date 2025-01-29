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
from vault.base_strategy import Order

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
        """Place a market or stop-loss-limit order. All other order types are rejected."""
        symbol = order.symbol
        granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
        current_price = market_data_df.loc[granularity].xs(symbol, axis=1, level='symbol')['close'].iloc[-1]
        current_system_timestamp = market_data_df.index.get_level_values(1)[-1]
        if order.order_type == 'MARKET':
            response_order = deepcopy(order)
            response_order.status = 'closed'
            
            # Apply slippage and fees to fill price
            fill_price = current_price
            direction = 1 if order.orderDirection == 'BUY' else -1
            response_order.filled_price = fill_price
            
            response_order.filled_timestamp = system_timestamp
            if not hasattr(order, 'order_id') or order.order_id is None:
                order_ = deepcopy(order)
            
        elif order.order_type == 'STOPLOSS':
            response_order = deepcopy(order)
            response_order.status = 'open'
            response_order.broker_order_id = generate_hash_id(order.dict(), system_timestamp)
            setattr(response_order, 'message', 'Stop-loss order placed.')
            setattr(response_order, 'fresh_update', True)
            
        else:
            response_order = deepcopy(order)
        
        return response_order
    
    def update_order_status(self, order, market_data_df):
        '''if the order is open, check if it's filled'''
        system_timestamp = market_data_df.index.get_level_values(1)[-1]
        slippage = getattr(order, 'slippage', 0.001)  # Default 0.1%
        brokerage_fee = getattr(order, 'brokerage_fee', 0.0035)  # Default 0.35%
        symbol = order.symbol
        min_granularity = self.market_data_extractor.get_market_data_df_minimum_granularity(market_data_df)
        close_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'close')
        high_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'high')
        low_prices = self.market_data_extractor.get_market_data_df_symbol_prices(market_data_df, min_granularity, symbol, 'low')
        
        close_prices.dropna(inplace=True)
        close_prices = close_prices.tolist()
        current_close_price = close_prices[-1]
        
        high_prices.dropna(inplace=True)
        high_prices = high_prices.tolist()
        current_high_price = high_prices[-1]

        low_prices.dropna(inplace=True)
        low_prices = low_prices.tolist()
        current_low_price = low_prices[-1]
                
        if order.order_type == 'STOPLOSS':
            if order.orderDirection == 'BUY' and current_high_price >= order.price or order.orderDirection == 'SELL' and current_low_price <= order.price:
                response_order = deepcopy(order)
                response_order.status = 'closed'
                self.available_granularities = market_data_df.index.get_level_values(0).unique()
                self.min_granularity_val = min([self.granularity_lookup_dict[granularity] for granularity in self.available_granularities])
                self.min_granularity = list(self.granularity_lookup_dict.keys())[list(self.granularity_lookup_dict.values()).index(self.min_granularity_val)]
                fill_price = order.price if self.min_granularity == '1d' else current_close_price
                
                # Apply slippage and fees
                direction = 1 if order.orderDirection == 'BUY' else -1
                fill_price *= (1 + direction * slippage)  # Apply slippage
                fill_price *= (1 + direction * brokerage_fee)  # Apply brokerage fee
                
                response_order.filled_price = fill_price
                response_order.filled_timestamp = system_timestamp
                setattr(response_order, 'fresh_update', True)
                setattr(response_order, 'message', 'Stop-loss order filled.')
            else:
                response_order = deepcopy(order)
                response_order.status = 'open'
                setattr(response_order, 'fresh_update', False)
        else:
            response_order = deepcopy(order)
            response_order.status = 'rejected'
            setattr(response_order, 'fresh_update', True)
            setattr(response_order, 'message', 'Order Rejected: Order type not supported.')
            self.logger.warning(f"ORDER REJECTED: Order type {order.order_type} not supported.")
        
        return response_order
    
    def modify_order(self, order, system_timestamp):
        for modify_reason in getattr(order, 'modify_reason', []):
            assert modify_reason.lower() in ['new_price', 'new_quantity']
        response_order = deepcopy(order)
        response_order.status = 'open'
        setattr(response_order, 'modified_timestamp', system_timestamp)
        setattr(response_order, 'fresh_update', True)
        
        return response_order
    
    def cancel_order(self, order, system_timestamp):
        pass
           
    def execute_order(self, order, market_data_df, system_timestamp):
        if order.status == 'pending':
            # Execute the order in the simulator
            response_order = self.place_order(order, market_data_df=market_data_df, system_timestamp=system_timestamp)
        elif order.status == 'open':
            # Update the order status in the simulator and check if it's filled
            response_order = self.update_order_status(order, market_data_df=market_data_df)
        elif order.status == 'modify':
            response_order = self.modify_order(order, system_timestamp=system_timestamp)
        else:
            self.logger.debug({'order':order})
            raise ValueError(f"Order status '{order.status}' not supported.")
        
        return response_order
    
    def create_account_summary(self, trading_currency, base_currency, base_currency_to_trading_currency_exchange_rate, starting_account_inputs):
        account_balance_dict = {trading_currency:{}}
        for currency in starting_account_inputs:
            if base_currency != trading_currency and currency == base_currency:
                base_currency_account_balance_dict = {}
                base_currency_account_balance_dict[currency] = starting_account_inputs[currency]
                base_currency_account_balance_dict[currency]['total_buying_power'] = float(base_currency_account_balance_dict[currency]['buying_power_available']) + float(base_currency_account_balance_dict[currency]['buying_power_used'])
                base_currency_account_balance_dict[currency]['pct_of_margin_used'] = float(base_currency_account_balance_dict[currency]['buying_power_used']) / float(base_currency_account_balance_dict[currency]['buying_power_available'])
                
                for key, value in base_currency_account_balance_dict[currency].items():
                    if key not in ['cushion', 'margin_multipler', 'pct_of_margin_used']:
                        account_balance_dict[trading_currency][key] = round(value * base_currency_to_trading_currency_exchange_rate, 2)
                    else:
                        account_balance_dict[trading_currency][key] = round(value, 2)
        
        return account_balance_dict

class Yahoo():
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
        self.asset_data_df_dict = {}
    
    def get_nasdaq_stock_symbols(self, nasdaq_csv_filepath, min_market_cap=10 * 1 * 1000 * 1000 * 1000):
        # load the CSV file into a DataFrame
        nasdaq_df = pd.read_csv(nasdaq_csv_filepath)
        filename = os.path.basename(nasdaq_csv_filepath)
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
        pruned_df.sort_values('Market Cap', ascending=False, inplace=True)
        stock_symbols = pruned_df['Symbol'].tolist()
        # save the stock symbols to a json file
        with open('stock_symbols.json', 'w') as file:
            json.dump(stock_symbols, file)
        
        return pruned_df, stock_symbols