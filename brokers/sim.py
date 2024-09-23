import os
import sys
import time
import json
import pandas as pd
import yfinance as yf
from systems.utils import create_logger
import logging
from tqdm import tqdm

# Main Simulation Class
class Sim():
    def __init__(self):
        self.data = Yahoo()  # Yahoo Finance Data Fetcher
        self.execute = Execute()  # Order Execution
    
    def run_simulation(self, symbol, order_type, quantity, stop_loss=None, limit_price=None):
        """
        Runs the simulation by fetching data, placing an order, and executing it.
        """
        # Fetch current price data
        current_data = self.data.update_price_data_single_asset(symbol)
        
        if current_data is None or current_data.empty:
            print(f"No data available for {symbol}.")
            return
        
        current_price = current_data['Close'].iloc[-1]
        
        # If the current price is available, place an order
        if order_type == 'market':
            order = self.execute.place_order(symbol=symbol, order_type='market', quantity=quantity, price=current_price)
        elif order_type == 'stop_loss_limit':
            order = self.execute.place_order(symbol=symbol, order_type='stop_loss_limit', quantity=quantity, stop_loss=stop_loss, limit_price=limit_price)
        
        # Execute the order based on current prices
        self.execute.execute_orders({symbol: current_price})
        
        # Get open orders and positions after execution
        open_orders = self.execute.get_open_orders()
        open_positions = self.execute.get_open_positions()
        
        # Print results
        print(f"Open Orders: {open_orders}")
        print(f"Open Positions: {open_positions}")

# Order Execution Class
class Execute():
    def __init__(self):
        self.open_orders = []  # List of open orders
        self.open_positions = {}  # Dictionary of open positions by stock symbol

    def place_order(self, symbol, order_type, quantity, price=None, stop_loss=None, limit_price=None):
        """
        Place a market or stop-loss-limit order.
        """
        order = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'limit_price': limit_price,
            'status': 'open'  # 'open', 'executed', or 'canceled'
        }
        self.open_orders.append(order)
        return order
    
    def execute_orders(self, current_prices):
        """
        Execute open orders based on current stock prices.
        """
        for order in self.open_orders:
            symbol = order['symbol']
            current_price = current_prices.get(symbol)

            if order['status'] == 'open':
                # Market order: execute immediately
                if order['type'] == 'market':
                    self._execute_order(order, current_price)

                # Stop-loss-limit order: execute if conditions are met
                elif order['type'] == 'stop_loss_limit':
                    if order['stop_loss'] >= current_price or current_price <= order['limit_price']:
                        self._execute_order(order, current_price)

        # Clean up executed orders
        self.open_orders = [order for order in self.open_orders if order['status'] == 'open']

    def _execute_order(self, order, execution_price):
        """
        Execute the given order and update positions.
        """
        symbol = order['symbol']
        quantity = order['quantity']

        # Add position to open_positions
        if symbol in self.open_positions:
            self.open_positions[symbol] += quantity
        else:
            self.open_positions[symbol] = quantity

        order['status'] = 'executed'
        order['execution_price'] = execution_price
        print(f"Executed {order['type']} order for {symbol} at {execution_price}")

    def get_open_orders(self):
        """Return the list of open orders."""
        return self.open_orders

    def get_open_positions(self):
        """Return the current open positions."""
        return self.open_positions

class Yahoo():
    def __init__(self):
        self.logger = create_logger(log_level=logging.DEBUG, logger_name='datafetcher', print_to_console=True)
    
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
        
        # Load data
        asset_data_df = pd.read_csv(csv_file_path)
        asset_data_df['symbol'] = symbol
        asset_data_df['date'] = pd.to_datetime(asset_data_df['Date'])
        
        return asset_data_df

# Example of running the simulation
sim = Sim()
#orders = retOrders()
#sim.run_simulation(orders)
