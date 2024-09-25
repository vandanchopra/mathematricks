#from ib_insync import *
import pandas as pd
#from brokers import Brokers
import json
from brokers.ibkr import *
from vault1 import *
from brokers.sim import *
from vault import *
from config import config_dict
import pandas as pd  # Assuming you are using pandas Timestamps
ib = IB()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable types like pd.Timestamp."""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()  # Convert Timestamp to ISO format
        return super().default(obj)

class OMS:
    def __init__(self, config):
        self.config_dict = config
        self.open_orders = self.load_json('open_orders.json')
        self.closed_orders = self.load_json('closed_orders.json')
        
        sync_direction = 'broker-to-oms' if config_dict['run_mode'] == 1 else 'oms-to-broker'
        self.sync_open_orders(sync_direction)

    def load_json(self, file_name):
        """Load JSON data from the file, or return an empty list if not found."""
        data = []
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"No {file_name} file found, starting with an empty list.")
        except json.JSONDecodeError:
            print(f"Error decoding the {file_name}, please check the format.")
        return data

    def save_json(self, file_name, data):
        """Save the data to a JSON file using the custom encoder."""
        with open(file_name, 'w') as file:
            json.dump(data, file, cls=CustomJSONEncoder)

    def sync_open_orders(self, sync_direction):
        if sync_direction == 'broker-to-oms':
            # Fetch open orders from IBKR or SIM broker and update self.open_orders
            open_orders = instance1.ib.openOrders() if self.config_dict['broker'] == 'IBKR' else sim_instance.get_open_orders()

            if open_orders:
                for order in open_orders:
                    print(f"Order ID: {order.orderId}, Action: {order.action}, Status: {order.status}")
                    self.open_orders.append(order)
            else:
                print("No open orders found.")
        
        elif sync_direction == 'oms-to-broker':
            # Send all open orders from self.open_orders to the broker
            for order in self.open_orders:
                self.ib.update_order(order) if order['broker'] == 'IBKR' else self.brokers.sim.update_order(order)
        
        self.save_json('open_orders.json', self.open_orders)

    def execute_orders(self, new_orders):
        """Execute a list of multi-leg orders."""
        ibkr_trader = IBKR(None)
        sim_trader = Sim()
    
        for multi_leg_order in new_orders:
            for order in multi_leg_order:
                run_mode = self.config_dict['run_mode']
                if run_mode == 3:
                    order['broker'] = 'SIM'

                if order['broker'] == 'IBKR':
                    ibkr_trader.place_order(
                        ticker=order['symbol'],
                        exchange='SMART',
                        currency='USD',
                        orderSide=order['orderSide'],
                        orderQuantity=order['orderQuantity'],
                        orderType=order['orderType'],
                        limit_price=order.get('entryPrice', 0),
                        stop_price=order.get('exitPrice', 0)
                    )
                    self.open_orders.append(order)
                
                elif order['broker'] == 'SIM':
                    sim_trader.run_simulation(
                        symbol=order['symbol'], 
                        order_type=order['orderType'],
                        quantity=order['orderQuantity'], 
                        limit_price=order.get('entryPrice', 0), 
                        stop_loss=order.get('exitPrice', 0)
                    )
                    self.open_orders.append(order)
                else:
                    raise Exception(f"Broker {order['broker']} not supported.")
        
        self.sync_open_orders('broker-to-oms')
        self.save_json('open_orders.json', self.open_orders)
        self.update_order_status()

    def update_order_status(self):
        """Check the broker for updates on open orders and move closed orders."""
        updated_open_orders = []
        
        for order in self.open_orders:
            status = instance1.get_order_status(order['OrderID'])  # Fetch order status from IBKR
            
            if status == 'Filled' or status == 'Cancelled':
                print(f"Order {order['orderId']} is {status}. Moving to closed orders.")
                self.closed_orders.append(order)
            else:
                updated_open_orders.append(order)
        
        self.open_orders = updated_open_orders
        self.save_json('open_orders.json', self.open_orders)
        self.save_json('closed_orders.json', self.closed_orders)


# Usage example
if __name__ == '__main__':    
    ibkr_trader = IBKR(None)
    sim_trader = Sim()
    instance1 = ibkr.IBKR(ib)
    sim_instance = sim.Execute()
    SIM = sim.Sim()
    
    
    orders = retOrders()  # Example orders to execute

    instance = OMS(config_dict)
    instance.execute_orders(orders)
