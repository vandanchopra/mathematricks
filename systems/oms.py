from ib_insync import *
import pandas as pd
from brokers.brokers import Brokers
import json


ib = IB()

class OMS:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.brokers = Brokers()
        self.open_orders = self.load_open_orders()
        
        sync_direction = 'broker-to-oms' if config_dict['run_mode'] == 1 else 'oms-to-broker'
        self.sync_open_orders(sync_direction)

    def load_open_orders(self):
        """Open the JSON file and load all the open orders into oms.orders"""
        open_orders = []
        try:
            with open('open_orders.json', 'r') as file:
                open_orders = json.load(file)
        except FileNotFoundError:
            print("No open orders file found, starting with an empty list.")
        except json.JSONDecodeError:
            print("Error decoding the JSON file, please check the format.")
        return open_orders

    def save_open_orders(self):
        """Update the open orders in the JSON file."""
        with open('open_orders.json', 'w') as file:
            json.dump(self.open_orders, file)   

    def sync_open_orders(self, sync_direction):
        if sync_direction == 'broker-to-oms':
            # Get all the open orders from IBKR and update self.open_orders
            open_orders = ib.openOrders()
            
            if open_orders:
                for order in open_orders:
                    print(f"Order ID: {order.orderId}")
                    print(f"Action: {order.action}, Quantity: {order.totalQuantity}, Order Type: {order.orderType}")
                    print(f"Status: {order.status}")
                    print("-" * 50)
            else:
                print("No open orders found.")
            
        elif sync_direction == 'oms-to-broker':
            # Send all open orders from self.open_orders to the broker
            for order in self.open_orders:
                self.brokers.ib.update_order(order)
        else:
            raise Exception("sync_direction not supported. Use 'broker-to-oms' or 'oms-to-broker'.")

        # Now that the orders have been synced, update the open orders in the JSON file
        self.save_open_orders()

    def execute_orders(self, new_orders):
        """Execute a list of multi-leg orders."""
        for multi_leg_order in new_orders:
            for order in multi_leg_order:
                run_mode = self.config_dict['run_mode']

                # Override the broker to SIM if the run_mode is 3 (for simulated trades)
                if run_mode == 3:
                    order['broker'] = 'SIM'

                # Push the order to the broker
                if order['broker'] == 'IBKR':
                # Unpack dictionary and call place_order with appropriate arguments
                    self.brokers.ib.place_order(
                    ticker=order['symbol'],
                    exchange='SMART',  # Adjust if necessary
                    currency='USD',   # Adjust if necessary
                    orderSide=order['orderSide'],
                    orderQuantity=order['orderQuantity'],
                    orderType=order['orderType'],
                    limit_price=order.get('entryPrice', 0),
                    stop_price=order.get('exitPrice', 0)
                )
                elif order['broker'] == 'SIM':
                # Implement similar unpacking for SIM if required
                    self.brokers.sim.place_order(
                    ticker=order['symbol'],
                    exchange='SMART',
                    currency='USD',
                    orderSide=order['orderSide'],
                    orderQuantity=order['orderQuantity'],
                    orderType=order['orderType'],
                    limit_price=order.get('entryPrice', 0),
                    stop_price=order.get('exitPrice', 0)
                )
                else:
                    raise Exception(f"Broker {order['broker']} not supported. Use 'IBKR' or 'SIM'.")

            # Sync orders after execution
            sync_direction = 'broker-to-oms'
            self.sync_open_orders(sync_direction)

        # After sending orders to the broker, check the status of each order and update CSV
        self.update_order_status()

    def update_order_status(self):
        """Update the status of orders by checking with the broker."""
        # This function will check the broker for updates on open orders and reflect them in the CSV file.
        # Placeholder for future implementation.
        pass

# Usage example
if __name__ == '__main__':
    from config import config_dict
    oms = OMS(config_dict)
    
    # Load orders from the JSON file
    orders = [[
        # Entry order
        {
            'symbol': "INTC",
            'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
            'orderSide': "BUY",
            'entryPrice': 151.43,
            'orderType': "MKT",
            'timeInForce': 'DAY',
            'orderQuantity': 10,
            'strategy_name': "SMA15-SMA30",
            'broker': 'IBKR'
        },
        # Exit Order / Trail Stop Order
        {
            'symbol': "INTC",
            'timestamp': pd.Timestamp('2023-01-01 01:39:00'),
            'orderSide': 'SELL',
            'exitPrice': 149.43,
            'orderType': 'STP',
            'timeInForce': 'DAY',
            'orderQuantity': 10,
            'strategy_name': 'SMA15-SMA30',
            'broker': 'IBKR'
        }
    ]]
    
    # Execute orders
    oms.execute_orders(orders)